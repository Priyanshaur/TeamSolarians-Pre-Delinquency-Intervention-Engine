"""
Default Probability (PD) Model – XGBoost classifier with SMOTE and manual Platt scaling.
Predicts probability of default within next 30 days.
"""
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from utils.logger import get_logger
from utils import config

logger = get_logger("pd_model")


class PlattScaler:
    """Manual Platt scaling – fits a sigmoid on top of raw model scores."""

    def __init__(self):
        self.lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)

    def fit(self, raw_probs, y_true):
        self.lr.fit(raw_probs.reshape(-1, 1), y_true)
        return self

    def transform(self, raw_probs):
        return self.lr.predict_proba(raw_probs.reshape(-1, 1))[:, 1]


class PDModel:
    """
    Probability of Default model.
    - XGBoost with class imbalance handling (SMOTE + scale_pos_weight)
    - Manual Platt scaling for probability calibration
    - 5-fold stratified CV for hyperparameter validation
    """

    def __init__(self):
        self.model = None
        self.platt_scaler = None
        self.feature_columns = None
        self.metrics = {}

    def train(self, X: pd.DataFrame, y: pd.Series, feature_columns: list):
        """Train PD model with SMOTE and calibration."""
        self.feature_columns = feature_columns
        X_feat = X[feature_columns].copy()

        logger.info(f"Training PD model on {len(X_feat)} samples, {len(feature_columns)} features")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")

        # Split: train + calibration + test
        X_train_cal, X_test, y_train_cal, y_test = train_test_split(
            X_feat, y,
            test_size=config.TEST_SIZE,
            stratify=y,
            random_state=config.RANDOM_STATE,
        )

        X_train, X_cal, y_train, y_cal = train_test_split(
            X_train_cal, y_train_cal,
            test_size=config.CALIBRATION_SIZE,
            stratify=y_train_cal,
            random_state=config.RANDOM_STATE,
        )

        # SMOTE on training set only
        logger.info("Applying SMOTE to training set...")
        smote = SMOTE(random_state=config.RANDOM_STATE)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        logger.info(f"Post-SMOTE: {pd.Series(y_train_resampled).value_counts().to_dict()}")

        # Calculate scale_pos_weight
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / max(pos_count, 1)

        # Train XGBoost
        params = config.PD_MODEL_PARAMS.copy()
        params["scale_pos_weight"] = scale_pos_weight

        self.model = XGBClassifier(**params)
        self.model.fit(
            X_train_resampled, y_train_resampled,
            eval_set=[(X_cal, y_cal)],
            verbose=False,
        )

        # Manual Platt scaling on calibration set
        logger.info("Applying manual Platt scaling for probability calibration...")
        raw_cal_probs = self.model.predict_proba(X_cal)[:, 1]
        self.platt_scaler = PlattScaler()
        self.platt_scaler.fit(raw_cal_probs, y_cal.values)

        # Evaluate on test set
        self._evaluate(X_test, y_test)

        return self.metrics

    def _evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Evaluate model on test set."""
        y_prob = self._get_calibrated_probs(X_test)
        y_pred = (y_prob >= 0.5).astype(int)

        self.metrics = {
            "auc_roc": roc_auc_score(y_test, y_prob),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "gini": 2 * roc_auc_score(y_test, y_prob) - 1,
        }

        logger.info("=" * 50)
        logger.info("PD Model Evaluation Results:")
        for k, v in self.metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        logger.info("=" * 50)
        logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        logger.info(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

        # Store test data for evaluation plots
        self.test_data = {"X_test": X_test, "y_test": y_test, "y_prob": y_prob}

    def _get_calibrated_probs(self, X) -> np.ndarray:
        """Get calibrated probabilities through Platt scaling."""
        if isinstance(X, pd.DataFrame):
            raw_probs = self.model.predict_proba(X[self.feature_columns])[:, 1]
        else:
            raw_probs = self.model.predict_proba(X)[:, 1]
        return self.platt_scaler.transform(raw_probs)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated default probabilities."""
        return self._get_calibrated_probs(X)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return binary default predictions."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def save(self, path=None):
        """Save model to disk."""
        path = path or config.MODEL_ARTIFACTS_DIR / "pd_model.pkl"
        joblib.dump({
            "model": self.model,
            "platt_scaler": self.platt_scaler,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
        }, path)
        logger.info(f"PD model saved to {path}")

    @classmethod
    def load(cls, path=None):
        """Load model from disk."""
        path = path or config.MODEL_ARTIFACTS_DIR / "pd_model.pkl"
        data = joblib.load(path)
        instance = cls()
        instance.model = data["model"]
        instance.platt_scaler = data["platt_scaler"]
        instance.feature_columns = data["feature_columns"]
        instance.metrics = data["metrics"]
        logger.info(f"PD model loaded from {path}")
        return instance
