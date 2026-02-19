"""
Recovery Time Prediction Model – XGBoost Regressor.
Predicts days_to_recovery with prediction intervals (P10, P50, P90).
"""
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.logger import get_logger
from utils import config

logger = get_logger("recovery_model")


class RecoveryModel:
    """
    Recovery time prediction model.
    - XGBoost regressor for point estimate (P50)
    - Quantile regressors for P10 and P90 prediction intervals
    """

    def __init__(self):
        self.model_median = None
        self.model_p10 = None
        self.model_p90 = None
        self.feature_columns = None
        self.metrics = {}

    def train(self, X: pd.DataFrame, y: pd.Series, feature_columns: list):
        """Train recovery time model with prediction intervals."""
        self.feature_columns = feature_columns
        X_feat = X[feature_columns].copy()

        logger.info(f"Training Recovery model on {len(X_feat)} samples")
        logger.info(f"Recovery time stats: mean={y.mean():.1f}, std={y.std():.1f}")

        X_train, X_test, y_train, y_test = train_test_split(
            X_feat, y,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
        )

        # Median model (main point estimate)
        params = config.RECOVERY_MODEL_PARAMS.copy()
        self.model_median = XGBRegressor(
            objective="reg:squarederror", **params
        )
        self.model_median.fit(X_train, y_train, verbose=False)

        # P10 model (optimistic recovery)
        self.model_p10 = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=0.10,
            **params,
        )
        self.model_p10.fit(X_train, y_train, verbose=False)

        # P90 model (pessimistic recovery)
        self.model_p90 = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=0.90,
            **params,
        )
        self.model_p90.fit(X_train, y_train, verbose=False)

        # Evaluate
        self._evaluate(X_test, y_test)
        return self.metrics

    def _evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Evaluate model on test set."""
        y_pred = self.model_median.predict(X_test)
        y_p10 = self.model_p10.predict(X_test)
        y_p90 = self.model_p90.predict(X_test)

        self.metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
            "coverage_90": np.mean((y_test >= y_p10) & (y_test <= y_p90)),
        }

        logger.info("=" * 50)
        logger.info("Recovery Model Evaluation Results:")
        for k, v in self.metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        logger.info("=" * 50)

        self.test_data = {
            "X_test": X_test, "y_test": y_test,
            "y_pred": y_pred, "y_p10": y_p10, "y_p90": y_p90,
        }

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return recovery predictions with intervals."""
        X_feat = X[self.feature_columns]
        return pd.DataFrame({
            "recovery_p10": np.clip(self.model_p10.predict(X_feat), 5, 365),
            "recovery_p50": np.clip(self.model_median.predict(X_feat), 5, 365),
            "recovery_p90": np.clip(self.model_p90.predict(X_feat), 5, 365),
        })

    def save(self, path=None):
        path = path or config.MODEL_ARTIFACTS_DIR / "recovery_model.pkl"
        joblib.dump({
            "model_median": self.model_median,
            "model_p10": self.model_p10,
            "model_p90": self.model_p90,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
        }, path)
        logger.info(f"Recovery model saved to {path}")

    @classmethod
    def load(cls, path=None):
        path = path or config.MODEL_ARTIFACTS_DIR / "recovery_model.pkl"
        data = joblib.load(path)
        instance = cls()
        instance.model_median = data["model_median"]
        instance.model_p10 = data["model_p10"]
        instance.model_p90 = data["model_p90"]
        instance.feature_columns = data["feature_columns"]
        instance.metrics = data["metrics"]
        logger.info(f"Recovery model loaded from {path}")
        return instance
