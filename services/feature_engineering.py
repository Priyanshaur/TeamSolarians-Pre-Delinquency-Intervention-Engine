"""
Feature engineering pipeline for Financial Triage AI.
Computes behavioral stress features from raw transaction data.
"""
import numpy as np
import pandas as pd
from utils.logger import get_logger
from utils import config

logger = get_logger("feature_engineering")


def compute_salary_delay_proxy(group: pd.DataFrame) -> pd.Series:
    """Rolling z-score of salary delay relative to customer's historical pattern."""
    delays = group["salary_delay_days"]
    rolling_mean = delays.rolling(window=config.SALARY_DELAY_WINDOW, min_periods=1).mean()
    rolling_std = delays.rolling(window=config.SALARY_DELAY_WINDOW, min_periods=1).std().fillna(1)
    rolling_std = rolling_std.replace(0, 1)
    return (delays - rolling_mean) / rolling_std


def compute_savings_decline_pct(group: pd.DataFrame) -> pd.Series:
    """Exponentially weighted week-over-week percentage drop in savings."""
    savings = group["savings_balance"]
    ewma = savings.ewm(span=config.SAVINGS_EWMA_SPAN, min_periods=1).mean()
    pct_change = ewma.pct_change()
    # Negative = decline
    return pct_change.fillna(0)


def compute_emi_to_income_ratio(df: pd.DataFrame) -> pd.Series:
    """EMI-to-income ratio, capped at EMI_RATIO_CAP."""
    ratio = df["emi_amount"] / df["salary_credit"].clip(lower=1)
    return ratio.clip(upper=config.EMI_RATIO_CAP)


def compute_spike_zscore(group: pd.DataFrame, column: str) -> pd.Series:
    """Z-score of a column vs its rolling mean (spike detection)."""
    values = group[column].astype(float)
    rolling_mean = values.rolling(window=config.SPIKE_LOOKBACK, min_periods=1).mean()
    rolling_std = values.rolling(window=config.SPIKE_LOOKBACK, min_periods=1).std().fillna(1)
    rolling_std = rolling_std.replace(0, 1)
    return (values - rolling_mean) / rolling_std


def compute_failed_autodebit_streak(group: pd.DataFrame) -> pd.Series:
    """Count consecutive failed autodebits."""
    flags = group["failed_autodebit_flag"].values
    streaks = []
    current_streak = 0
    for flag in flags:
        if flag == 1:
            current_streak += 1
        else:
            current_streak = 0
        streaks.append(current_streak)
    return pd.Series(streaks, index=group.index)


def compute_payment_regularity_score(group: pd.DataFrame) -> pd.Series:
    """
    Entropy-based regularity score for payment timing.
    Lower entropy = more regular = lower risk.
    Normalized to [0, 1] where 1 = most irregular.
    """
    delays = group["salary_delay_days"].values
    # Bin delays into categories
    bins = np.array([-np.inf, 0, 2, 5, 10, np.inf])
    regularity_scores = []
    for i in range(len(delays)):
        window_start = max(0, i - config.SPIKE_LOOKBACK + 1)
        window = delays[window_start:i + 1]
        if len(window) < 2:
            regularity_scores.append(0.0)
            continue
        hist, _ = np.histogram(window, bins=bins)
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        max_entropy = np.log2(len(bins) - 1)
        regularity_scores.append(entropy / max_entropy if max_entropy > 0 else 0)
    return pd.Series(regularity_scores, index=group.index)


def compute_stress_velocity(df: pd.DataFrame) -> pd.Series:
    """
    Weighted combination of stress indicators.
    Uses acceleration (2nd derivative) where available.
    """
    weights = config.STRESS_VELOCITY_WEIGHTS
    score = pd.Series(0.0, index=df.index)

    for feature, weight in weights.items():
        if feature in df.columns:
            vals = df[feature].fillna(0).astype(float)
            score += weight * vals

    return score


def compute_employer_cluster_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Percentile rank of stress velocity within employer groups.
    High rank = many colleagues are also stressed.
    """
    # Get latest stress velocity per customer per employer
    employer_stats = df.groupby("employer_name")["stress_velocity_score"].transform(
        lambda x: x.rank(pct=True)
    )
    return employer_stats


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main feature engineering pipeline.
    Takes raw transaction data and produces engineered features.

    Args:
        df: Raw transaction DataFrame (validated).

    Returns:
        DataFrame with all original columns + engineered features.
    """
    logger.info(f"Engineering features for {df['customer_id'].nunique()} customers")

    df = df.sort_values(["customer_id", "week_number"]).copy()

    # ── Per-customer rolling features ──
    logger.info("Computing per-customer rolling features...")
    df["salary_delay_proxy"] = df.groupby("customer_id", group_keys=False).apply(
        compute_salary_delay_proxy
    ).reset_index(level=0, drop=True)

    df["savings_decline_pct"] = df.groupby("customer_id", group_keys=False).apply(
        compute_savings_decline_pct
    ).reset_index(level=0, drop=True)

    df["lending_app_spike"] = df.groupby("customer_id", group_keys=False).apply(
        lambda g: compute_spike_zscore(g, "lending_app_upi_count")
    ).reset_index(level=0, drop=True)

    df["atm_withdrawal_spike"] = df.groupby("customer_id", group_keys=False).apply(
        lambda g: compute_spike_zscore(g, "atm_withdrawals")
    ).reset_index(level=0, drop=True)

    df["failed_autodebit_streak"] = df.groupby("customer_id", group_keys=False).apply(
        compute_failed_autodebit_streak
    ).reset_index(level=0, drop=True)

    df["payment_regularity_score"] = df.groupby("customer_id", group_keys=False).apply(
        compute_payment_regularity_score
    ).reset_index(level=0, drop=True)

    # ── Non-grouped features ──
    logger.info("Computing global features...")
    df["emi_to_income_ratio"] = compute_emi_to_income_ratio(df)

    df["liquidity_buffer_ratio"] = (
        df["savings_balance"] / df["emi_amount"].clip(lower=1)
    ).clip(lower=0)

    # ── Stress velocity (needs other features computed first) ──
    df["stress_velocity_score"] = compute_stress_velocity(df)

    # ── Employer cluster risk ──
    df["employer_cluster_risk"] = compute_employer_cluster_risk(df)

    # ── Interaction features ──
    logger.info("Computing interaction features...")
    df["low_liquidity_high_emi"] = (
        (df["liquidity_buffer_ratio"] < 2.0).astype(float) *
        (df["emi_to_income_ratio"] > config.EMI_RATIO_DANGER_ZONE).astype(float) *
        (df["savings_decline_pct"] < -0.05).astype(float)
    )

    # ── Fill NaN with 0 for features, clip infinities ──
    feature_cols = [
        "salary_delay_proxy", "savings_decline_pct", "emi_to_income_ratio",
        "lending_app_spike", "atm_withdrawal_spike", "stress_velocity_score",
        "liquidity_buffer_ratio", "employer_cluster_risk", "failed_autodebit_streak",
        "payment_regularity_score", "low_liquidity_high_emi",
    ]
    for col in feature_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


def get_feature_columns() -> list:
    """Return list of feature column names used for model training."""
    return [
        "salary_credit", "savings_balance", "emi_amount", "outstanding_balance",
        "atm_withdrawals", "lending_app_upi_count", "failed_autodebit_flag",
        "salary_delay_days", "customer_tenure",
        "past_response_rate", "prior_restructuring_success", "autodebit_reliability",
        # Engineered features
        "salary_delay_proxy", "savings_decline_pct", "emi_to_income_ratio",
        "lending_app_spike", "atm_withdrawal_spike", "stress_velocity_score",
        "liquidity_buffer_ratio", "employer_cluster_risk", "failed_autodebit_streak",
        "payment_regularity_score", "low_liquidity_high_emi",
    ]


if __name__ == "__main__":
    from data.generate_data import generate_dataset
    df = generate_dataset(num_customers=100, num_weeks=12, save=False)
    df = engineer_features(df)
    print(df[get_feature_columns()].describe())
