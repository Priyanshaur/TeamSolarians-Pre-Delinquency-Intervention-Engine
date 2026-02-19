"""
Synthetic data generator for Financial Triage AI.
Generates realistic customer financial behavior with temporal patterns,
employer cohort correlation, seasonality, and macro shock events.
"""
import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
from utils.logger import get_logger
from utils import config

logger = get_logger("data_generator")
fake = Faker()
Faker.seed(config.RANDOM_STATE)
np.random.seed(config.RANDOM_STATE)


def _assign_employer_stress(employers: list) -> dict:
    """Mark a fraction of employers as distressed."""
    n_distressed = max(1, int(len(employers) * config.DISTRESSED_EMPLOYER_FRACTION))
    distressed = np.random.choice(employers, size=n_distressed, replace=False)
    return {emp: emp in distressed for emp in employers}


def _generate_customer_profile(customer_id: int, employer: str, is_employer_distressed: bool):
    """Generate a single customer's base profile."""
    base_salary = np.random.lognormal(mean=10.5, sigma=0.5)  # ~30K-80K range
    tenure = np.random.randint(6, 180)  # months

    # Stress probability influenced by employer distress
    stress_base = 0.12 if not is_employer_distressed else 0.40
    is_stressed = np.random.random() < stress_base

    return {
        "customer_id": f"CUST_{customer_id:05d}",
        "employer_name": employer,
        "customer_tenure": tenure,
        "base_salary": base_salary,
        "is_stressed": is_stressed,
        "stress_severity": np.random.uniform(0.3, 1.0) if is_stressed else 0.0,
        "emi_base": base_salary * np.random.uniform(0.15, 0.55),
        "outstanding_balance": base_salary * np.random.uniform(5, 30),
        "savings_base": base_salary * np.random.uniform(0.5, 6.0),
        # Intervention response features
        "past_response_rate": np.clip(np.random.normal(0.6, 0.25), 0, 1),
        "prior_restructuring_success": int(np.random.random() < 0.3) if is_stressed else 0,
        "autodebit_reliability": np.clip(np.random.normal(0.8, 0.2), 0, 1),
    }


def _generate_weekly_snapshots(profile: dict, num_weeks: int, macro_shock_weeks: set):
    """Generate weekly time-series for one customer."""
    rows = []
    base_date = datetime(2025, 1, 6)  # Start Monday
    severity = profile["stress_severity"]
    is_stressed = profile["is_stressed"]

    # Temporal decay curves for stressed customers
    savings = profile["savings_base"]
    salary_delay = 0  # days late

    for week in range(num_weeks):
        current_date = base_date + timedelta(weeks=week)
        month = current_date.month

        # Seasonality: higher stress in Jan (post-holiday) and Aug-Sep (school fees)
        seasonal_stress = 1.0
        if month in [1, 8, 9]:
            seasonal_stress = 1.3
        elif month in [6, 12]:
            seasonal_stress = 1.15

        # Macro shock amplifier
        macro_mult = 1.4 if week in macro_shock_weeks else 1.0

        # ── Salary credit ──
        if is_stressed:
            salary_delay += np.random.uniform(0, 1.5) * severity * seasonal_stress
            salary_jitter = np.random.normal(0, 0.05)
        else:
            salary_delay = np.random.uniform(-0.5, 0.5)
            salary_jitter = np.random.normal(0, 0.02)

        salary_credit = profile["base_salary"] * (1 + salary_jitter)
        salary_credit = max(salary_credit, profile["base_salary"] * 0.5)

        # ── Savings balance (exponential decay for stressed) ──
        if is_stressed:
            decay_rate = 0.03 + 0.08 * severity * seasonal_stress * macro_mult
            savings *= (1 - decay_rate)
            savings = max(savings, 0)
        else:
            savings *= (1 + np.random.uniform(-0.01, 0.03))

        # ── EMI amount (constant with small drift) ──
        emi_amount = profile["emi_base"] * (1 + np.random.normal(0, 0.01))

        # ── Utility payment delays ──
        util_delay_days = 0
        if is_stressed:
            util_delay_days = int(np.random.exponential(3) * severity * macro_mult)

        utility_payment_date = current_date + timedelta(days=util_delay_days)

        # ── ATM withdrawals (spike for stressed) ──
        atm_base = np.random.poisson(3)
        if is_stressed:
            atm_base = int(atm_base * (1 + 1.5 * severity * macro_mult))

        # ── Lending app usage (spike for stressed) ──
        lending_base = np.random.poisson(1)
        if is_stressed:
            week_frac = week / num_weeks
            lending_base = int(lending_base + np.random.poisson(4 * severity * (0.5 + week_frac)))

        # ── Failed autodebits ──
        fail_prob = 0.02
        if is_stressed:
            fail_prob = 0.10 + 0.35 * severity * macro_mult * seasonal_stress
        failed_autodebit = int(np.random.random() < fail_prob)

        row = {
            "customer_id": profile["customer_id"],
            "date": current_date.strftime("%Y-%m-%d"),
            "week_number": week + 1,
            "salary_credit": round(salary_credit, 2),
            "savings_balance": round(savings, 2),
            "emi_amount": round(emi_amount, 2),
            "outstanding_balance": round(
                profile["outstanding_balance"] * (1 + np.random.normal(0, 0.01)), 2
            ),
            "utility_payment_date": utility_payment_date.strftime("%Y-%m-%d"),
            "atm_withdrawals": atm_base,
            "lending_app_upi_count": lending_base,
            "failed_autodebit_flag": failed_autodebit,
            "salary_delay_days": round(salary_delay, 2),
            "employer_name": profile["employer_name"],
            "customer_tenure": profile["customer_tenure"],
            # Intervention response features
            "past_response_rate": round(profile["past_response_rate"], 4),
            "prior_restructuring_success": profile["prior_restructuring_success"],
            "autodebit_reliability": round(profile["autodebit_reliability"], 4),
        }
        rows.append(row)

    return rows


def _assign_targets(df: pd.DataFrame, profiles: list) -> pd.DataFrame:
    """Assign default and recovery targets based on stress severity."""
    profile_map = {p["customer_id"]: p for p in profiles}

    # Use only the latest week per customer for target assignment
    latest = df.groupby("customer_id")["week_number"].max().reset_index()
    latest.columns = ["customer_id", "max_week"]
    df = df.merge(latest, on="customer_id")

    targets = []
    for cid in df["customer_id"].unique():
        p = profile_map[cid]
        if p["is_stressed"]:
            # Higher severity → higher default probability
            default_prob = 0.3 + 0.5 * p["stress_severity"]
            default_flag = int(np.random.random() < default_prob)
            # Recovery time correlates with severity
            recovery_days = int(20 + 100 * p["stress_severity"] + np.random.normal(0, 15))
            recovery_days = max(20, min(recovery_days, 180))
        else:
            default_flag = int(np.random.random() < 0.03)
            recovery_days = int(np.random.uniform(10, 30))

        targets.append({
            "customer_id": cid,
            "default_next_30_days": default_flag,
            "days_to_recovery": recovery_days,
        })

    targets_df = pd.DataFrame(targets)
    df = df.merge(targets_df, on="customer_id")
    df.drop(columns=["max_week"], inplace=True)
    return df


def generate_dataset(
    num_customers: int = None,
    num_weeks: int = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Generate the full synthetic financial triage dataset.

    Returns:
        pd.DataFrame with customer_id × week time-series, stress features, and targets.
    """
    num_customers = num_customers or config.NUM_CUSTOMERS
    num_weeks = num_weeks or config.NUM_WEEKS

    logger.info(f"Generating dataset: {num_customers} customers × {num_weeks} weeks")

    # Employer stress assignment
    employer_stress = _assign_employer_stress(config.EMPLOYER_LIST)
    logger.info(
        f"Distressed employers: "
        f"{[e for e, s in employer_stress.items() if s]}"
    )

    # Macro shock weeks (random 2-week windows)
    macro_shock_weeks = set()
    for _ in range(max(1, int(num_weeks * config.MACRO_SHOCK_PROBABILITY))):
        start = np.random.randint(0, num_weeks - 1)
        macro_shock_weeks.update([start, min(start + 1, num_weeks - 1)])
    logger.info(f"Macro shock weeks: {sorted(macro_shock_weeks)}")

    # Generate profiles
    profiles = []
    all_rows = []
    for i in range(num_customers):
        employer = np.random.choice(config.EMPLOYER_LIST)
        profile = _generate_customer_profile(i, employer, employer_stress[employer])
        profiles.append(profile)
        rows = _generate_weekly_snapshots(profile, num_weeks, macro_shock_weeks)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df = _assign_targets(df, profiles)

    # Summary stats
    n_defaults = df.groupby("customer_id")["default_next_30_days"].first().sum()
    default_rate = n_defaults / num_customers
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Default rate: {default_rate:.2%} ({n_defaults}/{num_customers})")

    if save:
        output_path = config.DATA_DIR / "transactions.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved to {output_path}")

    return df


if __name__ == "__main__":
    df = generate_dataset()
    print(df.head())
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nShape: {df.shape}")
    print(f"\nDefault rate: {df.groupby('customer_id')['default_next_30_days'].first().mean():.2%}")
