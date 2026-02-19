"""
SHAP Explainability for Financial Triage AI.
Per-customer feature explanations and global importance.
"""
import numpy as np
import pandas as pd
import shap
import json
from utils.logger import get_logger
from utils import config

logger = get_logger("explainability")


class ShapExplainer:
    """
    SHAP-based model explainability.
    - TreeExplainer for XGBoost PD model
    - Per-customer top-N contributing features
    - Global feature importance
    """

    def __init__(self, model, feature_columns: list):
        """
        Args:
            model: Trained XGBoost model (the underlying model, not calibrated wrapper)
            feature_columns: List of feature column names
        """
        self.feature_columns = feature_columns
        self.explainer = shap.TreeExplainer(model)
        self.global_shap_values = None
        logger.info("SHAP TreeExplainer initialized")

    def explain_batch(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for a batch of customers."""
        X_feat = X[self.feature_columns].copy()
        shap_values = self.explainer.shap_values(X_feat)

        # For binary classification, shap_values might be a list [neg, pos]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class

        self.global_shap_values = shap_values
        self.X_explain = X_feat
        logger.info(f"Computed SHAP values for {len(X_feat)} customers")
        return shap_values

    def get_top_features(self, customer_idx: int, shap_values: np.ndarray,
                         X: pd.DataFrame, top_n: int = 5) -> list:
        """Get top N contributing features for a specific customer."""
        sv = shap_values[customer_idx]
        feature_importance = list(zip(self.feature_columns, sv,
                                       X[self.feature_columns].iloc[customer_idx].values))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

        return [
            {
                "feature": name,
                "shap_value": round(float(val), 4),
                "feature_value": round(float(feat_val), 4),
                "direction": "increases risk" if val > 0 else "decreases risk",
            }
            for name, val, feat_val in feature_importance[:top_n]
        ]

    def get_global_importance(self, shap_values: np.ndarray) -> pd.DataFrame:
        """Get global feature importance (mean absolute SHAP)."""
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": mean_abs_shap,
        }).sort_values("importance", ascending=False)
        return importance

    def explain_customer(self, customer_idx: int, shap_values: np.ndarray,
                         X: pd.DataFrame) -> dict:
        """Full explanation for one customer."""
        top_features = self.get_top_features(customer_idx, shap_values, X)
        base_value = float(self.explainer.expected_value)
        if isinstance(self.explainer.expected_value, np.ndarray):
            base_value = float(self.explainer.expected_value[1])

        return {
            "base_value": base_value,
            "output_value": base_value + float(shap_values[customer_idx].sum()),
            "top_features": top_features,
            "all_shap_values": dict(zip(
                self.feature_columns,
                [round(float(v), 4) for v in shap_values[customer_idx]]
            )),
        }

    def save_explanations(self, scored_df: pd.DataFrame, shap_values: np.ndarray,
                          top_n_customers: int = 50):
        """Save explanations for top-N priority customers to JSON."""
        explanations = {}
        X = scored_df[self.feature_columns]

        for i in range(min(top_n_customers, len(scored_df))):
            cid = scored_df.iloc[i]["customer_id"]
            explanations[cid] = self.explain_customer(i, shap_values, X)

        output_path = config.DATA_DIR / "shap_explanations.json"
        with open(output_path, "w") as f:
            json.dump(explanations, f, indent=2, default=str)
        logger.info(f"Saved SHAP explanations for {len(explanations)} customers to {output_path}")

        # Save global importance
        importance = self.get_global_importance(shap_values)
        importance.to_csv(config.DATA_DIR / "feature_importance.csv", index=False)
        logger.info("Saved global feature importance")

        return explanations
