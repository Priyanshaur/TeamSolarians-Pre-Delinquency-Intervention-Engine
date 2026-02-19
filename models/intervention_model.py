"""
Intervention Effectiveness Model – Logistic Regression.
Estimates probability that intervention will prevent default.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from utils.logger import get_logger
from utils import config

logger = get_logger("intervention_model")


class InterventionModel:
    """
    Estimates P(intervention prevents default).
    Uses logistic regression with L2 regularization on:
    - customer_tenure, past_response_rate, autodebit_reliability
    - prior_restructuring_success, payment_regularity_score
    - failed_autodebit_streak (inversely correlated)
    """

    FEATURE_COLS = [
        "customer_tenure", "past_response_rate", "autodebit_reliability",
        "prior_restructuring_success", "payment_regularity_score",
        "failed_autodebit_streak", "emi_to_income_ratio",
    ]

    def __init__(self):
        self.model = None
        self.scaler = None
        self.metrics = {}

    def _create_synthetic_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Create synthetic intervention success label.
        Logic: intervention is more likely to succeed when:
        - Higher tenure → more loyalty
        - Higher past response rate → more engaged
        - Higher autodebit reliability → more stable
        - Lower failed_autodebit_streak → not too far gone
        - Lower EMI-to-income ratio → more financial headroom
        """
        max_streak = max(df["failed_autodebit_streak"].max(), 1)
        score = (
            0.25 * (df["customer_tenure"] / df["customer_tenure"].max()).clip(0, 1) +
            0.25 * df["past_response_rate"] +
            0.20 * df["autodebit_reliability"] +
            0.15 * (1 - df["failed_autodebit_streak"] / max_streak) +
            0.10 * (1 - df["emi_to_income_ratio"] / config.EMI_RATIO_CAP) +
            0.05 * df["prior_restructuring_success"]
        )
        # Add noise and convert to binary
        score += np.random.normal(0, 0.1, len(score))
        return (score > score.median()).astype(int)

    def train(self, df: pd.DataFrame):
        """Train intervention effectiveness model."""
        logger.info("Training Intervention Effectiveness model")

        # Create synthetic target
        y = self._create_synthetic_target(df)
        X = df[self.FEATURE_COLS].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.TEST_SIZE,
            stratify=y,
            random_state=config.RANDOM_STATE,
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Logistic regression with L2
        self.model = LogisticRegression(
            C=1.0, penalty="l2", max_iter=1000,
            random_state=config.RANDOM_STATE,
        )
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        self.metrics = {
            "auc_roc": roc_auc_score(y_test, y_prob),
        }

        logger.info("=" * 50)
        logger.info("Intervention Model Evaluation:")
        logger.info(f"  AUC-ROC: {self.metrics['auc_roc']:.4f}")
        logger.info("=" * 50)

        return self.metrics

    def predict_effectiveness(self, df: pd.DataFrame) -> np.ndarray:
        """Return intervention effectiveness probability."""
        X = df[self.FEATURE_COLS].copy()
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def save(self, path=None):
        path = path or config.MODEL_ARTIFACTS_DIR / "intervention_model.pkl"
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "metrics": self.metrics,
        }, path)
        logger.info(f"Intervention model saved to {path}")

    @classmethod
    def load(cls, path=None):
        path = path or config.MODEL_ARTIFACTS_DIR / "intervention_model.pkl"
        data = joblib.load(path)
        instance = cls()
        instance.model = data["model"]
        instance.scaler = data["scaler"]
        instance.metrics = data["metrics"]
        logger.info(f"Intervention model loaded from {path}")
        return instance
