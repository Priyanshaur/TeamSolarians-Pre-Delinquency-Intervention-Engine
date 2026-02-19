"""
Training orchestrator – trains all models sequentially.
"""
import sys
import os
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.generate_data import generate_dataset
from services.feature_engineering import engineer_features, get_feature_columns
from models.pd_model import PDModel
from models.recovery_model import RecoveryModel
from models.intervention_model import InterventionModel
from models.evaluation import full_evaluation
from utils.logger import get_logger
from utils import config

logger = get_logger("trainer")


def train_all(df: pd.DataFrame = None):
    """
    Train all three models: PD, Recovery, and Intervention.

    Args:
        df: Pre-processed DataFrame with engineered features.
            If None, generates and engineers data from scratch.
    """
    # ── Data preparation ──
    if df is None:
        logger.info("No data provided, generating fresh dataset...")
        raw_df = generate_dataset()
        df = engineer_features(raw_df)

    # Use latest snapshot per customer for training
    latest = df.sort_values("week_number").groupby("customer_id").tail(1).copy()
    logger.info(f"Training on {len(latest)} customer snapshots")

    feature_cols = get_feature_columns()

    # ── Train PD Model ──
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING MODEL 1: Probability of Default (PD)")
    logger.info("=" * 60)
    pd_model = PDModel()
    pd_metrics = pd_model.train(latest, latest["default_next_30_days"], feature_cols)
    pd_model.save()

    # Run banking-grade evaluation
    if hasattr(pd_model, "test_data"):
        full_evaluation(
            pd_model.test_data["y_test"].values,
            pd_model.test_data["y_prob"],
            model_name="PD",
        )

    # ── Train Recovery Model ──
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING MODEL 2: Recovery Time Prediction")
    logger.info("=" * 60)
    recovery_model = RecoveryModel()
    recovery_metrics = recovery_model.train(latest, latest["days_to_recovery"], feature_cols)
    recovery_model.save()

    # ── Train Intervention Model ──
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING MODEL 3: Intervention Effectiveness")
    logger.info("=" * 60)
    intervention_model = InterventionModel()
    intervention_metrics = intervention_model.train(latest)
    intervention_model.save()

    # ── Summary ──
    logger.info("\n" + "=" * 60)
    logger.info("ALL MODELS TRAINED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"PD Model:          AUC={pd_metrics.get('auc_roc', 0):.4f}, "
                f"Gini={pd_metrics.get('gini', 0):.4f}")
    logger.info(f"Recovery Model:    MAE={recovery_metrics.get('mae', 0):.1f} days, "
                f"R²={recovery_metrics.get('r2', 0):.4f}")
    logger.info(f"Intervention Model: AUC={intervention_metrics.get('auc_roc', 0):.4f}")

    return {
        "pd_metrics": pd_metrics,
        "recovery_metrics": recovery_metrics,
        "intervention_metrics": intervention_metrics,
    }


if __name__ == "__main__":
    train_all()
