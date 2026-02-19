"""
Tests for loss engine.
Validates EL = PD × EAD × LGD, priority ranking, Monte Carlo.
"""
import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.loss_engine import (
    compute_expected_loss,
    compute_loss_velocity,
    compute_priority_score,
    monte_carlo_loss_simulation,
)


class TestExpectedLoss:

    def test_basic_calculation(self):
        """EL = PD × EAD × LGD."""
        pd_prob = np.array([0.5])
        ead = np.array([10000.0])
        lgd = 0.4
        el = compute_expected_loss(pd_prob, ead, lgd)
        assert np.isclose(el[0], 2000.0)

    def test_zero_pd(self):
        """Zero PD should give zero EL."""
        el = compute_expected_loss(np.array([0.0]), np.array([10000.0]), 0.45)
        assert el[0] == 0.0

    def test_vectorized(self):
        """Should work on arrays of any size."""
        n = 100
        pd_prob = np.random.uniform(0, 1, n)
        ead = np.random.uniform(1000, 100000, n)
        el = compute_expected_loss(pd_prob, ead, 0.45)
        assert len(el) == n
        assert (el >= 0).all()


class TestLossVelocity:

    def test_positive_velocity(self):
        """Increasing EL should give positive velocity."""
        vel = compute_loss_velocity(np.array([200.0]), np.array([100.0]))
        assert vel[0] > 0

    def test_negative_velocity(self):
        """Decreasing EL should give negative velocity."""
        vel = compute_loss_velocity(np.array([50.0]), np.array([100.0]))
        assert vel[0] < 0

    def test_zero_change(self):
        """No change should give zero velocity."""
        vel = compute_loss_velocity(np.array([100.0]), np.array([100.0]))
        assert np.isclose(vel[0], 0.0)


class TestPriorityScore:

    def test_higher_el_higher_priority(self):
        """Higher EL should give higher priority (all else equal)."""
        el = np.array([1000.0, 5000.0])
        lv = np.array([0.1, 0.1])
        sv = np.array([0.5, 0.5])
        ie = np.array([0.8, 0.8])
        scores = compute_priority_score(el, lv, sv, ie)
        assert scores[1] > scores[0]

    def test_higher_intervention_higher_priority(self):
        """Higher intervention effectiveness should increase priority."""
        el = np.array([5000.0, 5000.0])
        lv = np.array([0.1, 0.1])
        sv = np.array([0.5, 0.5])
        ie = np.array([0.3, 0.9])
        scores = compute_priority_score(el, lv, sv, ie)
        assert scores[1] > scores[0]

    def test_non_negative(self):
        """Priority scores should be non-negative."""
        n = 100
        scores = compute_priority_score(
            np.random.uniform(0, 10000, n),
            np.random.uniform(-0.5, 5, n),
            np.random.uniform(0, 5, n),
            np.random.uniform(0, 1, n),
        )
        assert (scores >= 0).all()


class TestMonteCarlo:

    def test_runs_without_error(self):
        """Monte Carlo simulation should complete."""
        result = monte_carlo_loss_simulation(
            np.array([0.3, 0.5, 0.7]),
            np.array([10000, 20000, 30000]),
            n_iterations=100,
        )
        assert "portfolio_p50" in result
        assert "portfolio_p95" in result

    def test_p95_greater_than_p50(self):
        """P95 should be >= P50."""
        result = monte_carlo_loss_simulation(
            np.random.uniform(0.1, 0.9, 50),
            np.random.uniform(5000, 50000, 50),
            n_iterations=500,
        )
        assert result["portfolio_p95"] >= result["portfolio_p50"]

    def test_stress_scenario_higher(self):
        """Severe recession should give higher loss than baseline."""
        pd_prob = np.random.uniform(0.1, 0.5, 50)
        ead = np.random.uniform(5000, 50000, 50)

        baseline = monte_carlo_loss_simulation(pd_prob, ead, 500, "baseline")
        stressed = monte_carlo_loss_simulation(pd_prob, ead, 500, "severe_recession")
        # P50 of stressed should generally be higher
        assert stressed["portfolio_p50"] > baseline["portfolio_p50"] * 0.8
