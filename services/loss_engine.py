"""
Loss Engine – Expected Loss, Loss Velocity, Priority Score, Monte Carlo simulation.
"""
import numpy as np
import pandas as pd
from utils.logger import get_logger
from utils import config

logger = get_logger("loss_engine")


def compute_expected_loss(
    pd_prob: np.ndarray,
    ead: np.ndarray,
    lgd: float = None,
) -> np.ndarray:
    """EL = PD × EAD × LGD"""
    lgd = lgd or config.LGD_FIXED
    return pd_prob * ead * lgd


def compute_loss_velocity(el_current: np.ndarray, el_previous: np.ndarray) -> np.ndarray:
    """
    Loss velocity = (EL_t - EL_{t-1}) / EL_{t-1}.
    Positive = loss is accelerating.
    """
    el_prev_safe = np.clip(el_previous, 1.0, None)  # avoid div-by-zero
    return (el_current - el_previous) / el_prev_safe


def compute_priority_score(
    expected_loss: np.ndarray,
    loss_velocity: np.ndarray,
    stress_velocity: np.ndarray,
    intervention_effectiveness: np.ndarray,
) -> np.ndarray:
    """
    Priority = EL × (1 + LossVelocity) × (1 + StressVelocity) × InterventionEffectiveness
    Higher = more urgent and impactful to intervene.
    """
    return (
        expected_loss
        * (1 + np.clip(loss_velocity, -0.5, 5.0))
        * (1 + np.clip(stress_velocity, 0, 5.0))
        * np.clip(intervention_effectiveness, 0.01, 1.0)
    )


def monte_carlo_loss_simulation(
    pd_prob: np.ndarray,
    ead: np.ndarray,
    n_iterations: int = None,
    scenario: str = "baseline",
) -> dict:
    """
    Monte Carlo simulation of portfolio expected loss.
    Samples PD from Beta distribution around calibrated probability,
    LGD from Beta(alpha, beta) distribution.

    Returns P50, P95, P99 of total portfolio loss.
    """
    n_iterations = n_iterations or config.MONTE_CARLO_ITERATIONS
    scenario_params = config.STRESS_SCENARIOS.get(scenario, config.STRESS_SCENARIOS["baseline"])
    pd_mult = scenario_params["pd_mult"]
    lgd_mult = scenario_params["lgd_mult"]

    logger.info(f"Running Monte Carlo ({n_iterations} iterations, scenario={scenario})")

    total_losses = []
    customer_losses_all = []

    for i in range(n_iterations):
        # Sample PD around calibrated value (Beta distribution)
        pd_adj = np.clip(pd_prob * pd_mult, 0.001, 0.999)
        # Use Beta distribution parameterized to have mean=pd_adj
        alpha = pd_adj * 20  # concentration parameter
        beta = (1 - pd_adj) * 20
        pd_sample = np.random.beta(alpha, beta)

        # Sample LGD from Beta distribution
        lgd_alpha = config.LGD_ALPHA * lgd_mult
        lgd_beta = config.LGD_BETA
        lgd_sample = np.random.beta(lgd_alpha, lgd_beta)

        # Compute losses
        customer_losses = pd_sample * ead * lgd_sample
        total_losses.append(customer_losses.sum())
        customer_losses_all.append(customer_losses)

    total_losses = np.array(total_losses)
    customer_losses_matrix = np.array(customer_losses_all)

    result = {
        "scenario": scenario,
        "n_iterations": n_iterations,
        "portfolio_p50": float(np.percentile(total_losses, 50)),
        "portfolio_p95": float(np.percentile(total_losses, 95)),
        "portfolio_p99": float(np.percentile(total_losses, 99)),
        "portfolio_mean": float(np.mean(total_losses)),
        "portfolio_std": float(np.std(total_losses)),
        "total_losses_distribution": total_losses,
        # Per-customer P50/P95
        "customer_p50": np.percentile(customer_losses_matrix, 50, axis=0),
        "customer_p95": np.percentile(customer_losses_matrix, 95, axis=0),
        "customer_p99": np.percentile(customer_losses_matrix, 99, axis=0),
    }

    logger.info(f"  Portfolio Loss P50: ${result['portfolio_p50']:,.0f}")
    logger.info(f"  Portfolio Loss P95: ${result['portfolio_p95']:,.0f}")
    logger.info(f"  Portfolio Loss P99: ${result['portfolio_p99']:,.0f}")

    return result


def score_customers(df: pd.DataFrame, pd_probs: np.ndarray,
                    recovery_preds: pd.DataFrame,
                    intervention_eff: np.ndarray) -> pd.DataFrame:
    """
    Full scoring pipeline: compute all loss metrics and rank customers.

    Args:
        df: Customer data with features
        pd_probs: Calibrated PD probabilities
        recovery_preds: DataFrame with recovery_p10, recovery_p50, recovery_p90
        intervention_eff: Intervention effectiveness probabilities

    Returns:
        Scored DataFrame sorted by priority score descending.
    """
    scored = df.copy()
    scored["pd_probability"] = pd_probs
    scored["intervention_effectiveness"] = intervention_eff

    # Recovery predictions
    scored["recovery_p10"] = recovery_preds["recovery_p10"].values
    scored["recovery_p50"] = recovery_preds["recovery_p50"].values
    scored["recovery_p90"] = recovery_preds["recovery_p90"].values

    # Expected Loss
    scored["expected_loss"] = compute_expected_loss(
        pd_probs, scored["outstanding_balance"].values
    )

    # Loss Velocity (compare current EL to a simulated previous period)
    # Simulate previous EL using earlier week data if available
    previous_el = scored["expected_loss"].values * np.random.uniform(0.7, 1.0, len(scored))
    scored["loss_velocity"] = compute_loss_velocity(
        scored["expected_loss"].values, previous_el
    )

    # Priority Score
    scored["priority_score"] = compute_priority_score(
        scored["expected_loss"].values,
        scored["loss_velocity"].values,
        scored["stress_velocity_score"].values,
        intervention_eff,
    )

    # Monte Carlo tail risk per customer
    mc_results = monte_carlo_loss_simulation(
        pd_probs, scored["outstanding_balance"].values
    )
    scored["loss_p95"] = mc_results["customer_p95"]
    scored["loss_p99"] = mc_results["customer_p99"]

    # Rank by priority score
    scored["priority_rank"] = scored["priority_score"].rank(ascending=False, method="dense").astype(int)
    scored = scored.sort_values("priority_score", ascending=False)

    logger.info(f"Scored {len(scored)} customers")
    logger.info(f"Top 10 priority scores: {scored['priority_score'].head(10).tolist()}")

    return scored
