"""
Tests for feature engineering pipeline.
Validates feature values, no NaN/Inf after engineering.
"""
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.generate_data import generate_dataset
from services.feature_engineering import engineer_features, get_feature_columns


@pytest.fixture(scope="module")
def engineered_df():
    """Generate and engineer a small dataset for testing."""
    raw = generate_dataset(num_customers=100, num_weeks=8, save=False)
    return engineer_features(raw)


class TestFeatureEngineering:

    def test_no_nan_in_features(self, engineered_df):
        """Engineered features should have no NaN values."""
        feature_cols = get_feature_columns()
        existing = [c for c in feature_cols if c in engineered_df.columns]
        nan_counts = engineered_df[existing].isna().sum()
        has_nan = nan_counts[nan_counts > 0]
        assert len(has_nan) == 0, f"NaN found in features: {has_nan.to_dict()}"

    def test_no_inf_in_features(self, engineered_df):
        """Engineered features should have no Inf values."""
        feature_cols = get_feature_columns()
        existing = [c for c in feature_cols if c in engineered_df.columns]
        for col in existing:
            vals = engineered_df[col].values
            assert not np.any(np.isinf(vals)), f"Inf found in {col}"

    def test_emi_ratio_capped(self, engineered_df):
        """EMI-to-income ratio should be capped at EMI_RATIO_CAP."""
        from utils.config import EMI_RATIO_CAP
        assert engineered_df["emi_to_income_ratio"].max() <= EMI_RATIO_CAP + 0.01

    def test_liquidity_non_negative(self, engineered_df):
        """Liquidity buffer ratio should be >= 0."""
        assert (engineered_df["liquidity_buffer_ratio"] >= 0).all()

    def test_autodebit_streak_non_negative(self, engineered_df):
        """Failed autodebit streak should be >= 0."""
        assert (engineered_df["failed_autodebit_streak"] >= 0).all()

    def test_feature_columns_exist(self, engineered_df):
        """All expected feature columns should be present."""
        feature_cols = get_feature_columns()
        for col in feature_cols:
            assert col in engineered_df.columns, f"Missing feature: {col}"

    def test_stress_velocity_computed(self, engineered_df):
        """Stress velocity score should be computed."""
        assert "stress_velocity_score" in engineered_df.columns
        assert not engineered_df["stress_velocity_score"].isna().any()


class TestDataGeneration:

    def test_generates_correct_size(self):
        """Dataset should have expected number of rows."""
        df = generate_dataset(num_customers=50, num_weeks=4, save=False)
        assert len(df) == 50 * 4

    def test_has_required_columns(self):
        """Dataset should have all required columns."""
        df = generate_dataset(num_customers=10, num_weeks=4, save=False)
        required = [
            "customer_id", "date", "salary_credit", "savings_balance",
            "emi_amount", "outstanding_balance", "atm_withdrawals",
            "lending_app_upi_count", "failed_autodebit_flag",
            "employer_name", "customer_tenure", "default_next_30_days",
            "days_to_recovery",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_default_rate_reasonable(self):
        """Default rate should be roughly around expected rate."""
        df = generate_dataset(num_customers=500, num_weeks=4, save=False)
        rate = df.groupby("customer_id")["default_next_30_days"].first().mean()
        assert 0.05 <= rate <= 0.50, f"Default rate {rate:.2%} is outside expected range"

    def test_no_negative_balances(self):
        """No negative savings balance."""
        df = generate_dataset(num_customers=50, num_weeks=4, save=False)
        assert (df["savings_balance"] >= 0).all()
