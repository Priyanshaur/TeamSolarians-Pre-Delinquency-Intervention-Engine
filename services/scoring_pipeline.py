"""
Scoring Pipeline – Loads models, engineers features, and scores all customers.
Simulates real-time batch scoring pipeline.
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.generate_data import generate_dataset
from services.feature_engineering import engineer_features, get_feature_columns
from models.pd_model import PDModel
from models.recovery_model import RecoveryModel
from models.intervention_model import InterventionModel
from services.loss_engine import score_customers
from utils.logger import get_logger
from utils import config

logger = get_logger("scoring_pipeline")


def run_scoring_pipeline(df: pd.DataFrame = None, save: bool = True) -> pd.DataFrame:
    """
    Full scoring pipeline:
    1. Load/generate data
    2. Engineer features
    3. Load trained models
    4. Score all customers
    5. Return ranked results

    Args:
        df: Optional pre-loaded DataFrame. If None, loads from CSV.
        save: Whether to save scored results to CSV.

    Returns:
        Scored and ranked DataFrame.
    """
    # ── Load/prepare data ──
    if df is None:
        data_path = config.DATA_DIR / "transactions.csv"
        if data_path.exists():
            logger.info(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)
        else:
            logger.info("No data found, generating fresh dataset...")
            df = generate_dataset()

    # ── Feature engineering ──
    logger.info("Running feature engineering pipeline...")
    df = engineer_features(df)

    # ── Use latest snapshot per customer ──
    latest = df.sort_values("week_number").groupby("customer_id").tail(1).copy()
    latest = latest.reset_index(drop=True)
    logger.info(f"Scoring {len(latest)} customers (latest snapshots)")

    # ── Load models ──
    logger.info("Loading trained models...")
    pd_model = PDModel.load()
    recovery_model = RecoveryModel.load()
    intervention_model = InterventionModel.load()

    # ── Generate predictions ──
    feature_cols = get_feature_columns()

    logger.info("Generating PD predictions...")
    pd_probs = pd_model.predict_proba(latest)

    logger.info("Generating recovery time predictions...")
    recovery_preds = recovery_model.predict(latest)

    logger.info("Generating intervention effectiveness predictions...")
    intervention_eff = intervention_model.predict_effectiveness(latest)

    # ── Score and rank ──
    logger.info("Computing expected loss, priority scores, and Monte Carlo simulation...")
    scored = score_customers(latest, pd_probs, recovery_preds, intervention_eff)

    # ── Save results ──
    if save:
        output_path = config.DATA_DIR / "scored_customers.csv"
        # Select key columns for output
        output_cols = [
            "customer_id", "employer_name", "customer_tenure",
            "salary_credit", "savings_balance", "emi_amount", "outstanding_balance",
            "pd_probability", "expected_loss", "loss_velocity", "loss_p95", "loss_p99",
            "recovery_p10", "recovery_p50", "recovery_p90",
            "intervention_effectiveness", "stress_velocity_score",
            "employer_cluster_risk", "priority_score", "priority_rank",
            "default_next_30_days", "days_to_recovery",
            # Feature columns for explainability
            "salary_delay_proxy", "savings_decline_pct", "emi_to_income_ratio",
            "lending_app_spike", "atm_withdrawal_spike", "liquidity_buffer_ratio",
            "failed_autodebit_streak", "payment_regularity_score", "low_liquidity_high_emi",
        ]
        existing_cols = [c for c in output_cols if c in scored.columns]
        scored[existing_cols].to_csv(output_path, index=False)
        logger.info(f"Scored results saved to {output_path}")

    return scored


if __name__ == "__main__":
    scored = run_scoring_pipeline()
    print(f"\nTop 10 Priority Customers:")
    print(scored[["customer_id", "pd_probability", "expected_loss",
                   "priority_score", "priority_rank"]].head(10).to_string())
