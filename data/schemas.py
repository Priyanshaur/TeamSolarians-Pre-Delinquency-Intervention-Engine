"""
Pandera validation schemas for Financial Triage AI data.
Catches silent data corruption before it poisons models.
"""
import pandera as pa
from pandera import Column, Check, DataFrameSchema


# ── Raw Transaction Schema ─────────────────────────────────────
raw_transaction_schema = DataFrameSchema(
    {
        "customer_id": Column(str, Check.str_startswith("CUST_"), nullable=False),
        "date": Column(str, nullable=False),
        "week_number": Column(int, Check.in_range(1, 52), nullable=False),
        "salary_credit": Column(float, Check.greater_than(0), nullable=False),
        "savings_balance": Column(float, Check.greater_than_or_equal_to(0), nullable=False),
        "emi_amount": Column(float, Check.greater_than(0), nullable=False),
        "outstanding_balance": Column(float, Check.greater_than(0), nullable=False),
        "utility_payment_date": Column(str, nullable=False),
        "atm_withdrawals": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
        "lending_app_upi_count": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
        "failed_autodebit_flag": Column(int, Check.isin([0, 1]), nullable=False),
        "salary_delay_days": Column(float, nullable=False),
        "employer_name": Column(str, nullable=False),
        "customer_tenure": Column(int, Check.greater_than(0), nullable=False),
        "past_response_rate": Column(float, Check.in_range(0, 1), nullable=False),
        "prior_restructuring_success": Column(int, Check.isin([0, 1]), nullable=False),
        "autodebit_reliability": Column(float, Check.in_range(0, 1), nullable=False),
        "default_next_30_days": Column(int, Check.isin([0, 1]), nullable=False),
        "days_to_recovery": Column(int, Check.greater_than(0), nullable=False),
    },
    coerce=True,
)


# ── Engineered Features Schema ─────────────────────────────────
engineered_features_schema = DataFrameSchema(
    {
        "customer_id": Column(str, nullable=False),
        "salary_delay_proxy": Column(float, nullable=False),
        "savings_decline_pct": Column(float, nullable=True),  # NaN for first period
        "emi_to_income_ratio": Column(float, Check.in_range(0, 2.0), nullable=False),
        "lending_app_spike": Column(float, nullable=False),
        "atm_withdrawal_spike": Column(float, nullable=False),
        "stress_velocity_score": Column(float, nullable=False),
        "liquidity_buffer_ratio": Column(float, Check.greater_than_or_equal_to(0), nullable=False),
        "employer_cluster_risk": Column(float, nullable=False),
        "failed_autodebit_streak": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
        "payment_regularity_score": Column(float, nullable=False),
    },
    coerce=True,
)


def validate_raw_data(df):
    """Validate raw transaction data against schema."""
    return raw_transaction_schema.validate(df, lazy=True)


def validate_features(df):
    """Validate engineered features against schema."""
    return engineered_features_schema.validate(df, lazy=True)
