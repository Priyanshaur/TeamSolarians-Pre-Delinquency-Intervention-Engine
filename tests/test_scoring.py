"""
Integration test for end-to-end scoring pipeline.
"""
import sys
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestScoringPipeline:

    def test_end_to_end_small(self):
        """Full pipeline should run on small dataset without errors."""
        from data.generate_data import generate_dataset
        from services.feature_engineering import engineer_features, get_feature_columns
        from models.pd_model import PDModel
        from models.recovery_model import RecoveryModel
        from models.intervention_model import InterventionModel
        from services.loss_engine import score_customers

        # Generate small dataset
        raw = generate_dataset(num_customers=200, num_weeks=8, save=False)
        df = engineer_features(raw)
        latest = df.sort_values("week_number").groupby("customer_id").tail(1).reset_index(drop=True)

        feature_cols = get_feature_columns()

        # Train models
        pd_model = PDModel()
        pd_model.train(latest, latest["default_next_30_days"], feature_cols)

        recovery_model = RecoveryModel()
        recovery_model.train(latest, latest["days_to_recovery"], feature_cols)

        intervention_model = InterventionModel()
        intervention_model.train(latest)

        # Score
        pd_probs = pd_model.predict_proba(latest)
        recovery_preds = recovery_model.predict(latest)
        intervention_eff = intervention_model.predict_effectiveness(latest)

        scored = score_customers(latest, pd_probs, recovery_preds, intervention_eff)

        # Validate output
        assert len(scored) == 200
        assert "priority_score" in scored.columns
        assert "priority_rank" in scored.columns
        assert "expected_loss" in scored.columns
        assert scored["priority_score"].notna().all()
        assert scored["priority_rank"].min() == 1

    def test_alert_generation(self):
        """Alerts should be generated from scored data."""
        import numpy as np
        from services.alert_service import generate_alerts

        # Create minimal scored DataFrame
        scored = pd.DataFrame({
            "customer_id": [f"CUST_{i:05d}" for i in range(100)],
            "employer_name": ["TechCorp"] * 30 + ["FinServe"] * 70,
            "pd_probability": np.random.uniform(0, 1, 100),
            "expected_loss": np.random.uniform(0, 100000, 100),
            "loss_velocity": np.random.uniform(-0.5, 2, 100),
            "loss_p95": np.random.uniform(0, 150000, 100),
            "loss_p99": np.random.uniform(0, 200000, 100),
            "priority_score": np.random.uniform(0, 500000, 100),
            "emi_to_income_ratio": np.random.uniform(0.2, 0.9, 100),
            "liquidity_buffer_ratio": np.random.uniform(0.5, 5, 100),
            "intervention_effectiveness": np.random.uniform(0.2, 0.9, 100),
        })

        alerts = generate_alerts(scored)
        assert "individual_alerts" in alerts
        assert "cluster_alerts" in alerts
        assert isinstance(alerts["individual_alerts"], list)
