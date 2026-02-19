"""
Model monitoring – PSI (Population Stability Index) for drift detection.
"""
import numpy as np
import pandas as pd
from utils.logger import get_logger
from utils import config

logger = get_logger("monitoring")


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Compute Population Stability Index (PSI).
    PSI < 0.10: no significant shift
    PSI 0.10-0.20: moderate shift, monitor
    PSI > 0.20: significant shift, retrain

    Args:
        reference: Reference (training) distribution
        current: Current (scoring) distribution
        bins: Number of bins

    Returns:
        PSI value
    """
    # Create bins from reference distribution
    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    # Avoid duplicate edges
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)

    # Avoid log(0)
    ref_pct = np.clip(ref_pct, 0.001, None)
    cur_pct = np.clip(cur_pct, 0.001, None)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def monitor_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: list,
) -> pd.DataFrame:
    """
    Compute PSI for each feature between reference and current distributions.

    Returns:
        DataFrame with feature, PSI value, and drift status.
    """
    results = []
    for col in feature_columns:
        if col not in reference_df.columns or col not in current_df.columns:
            continue

        ref_vals = reference_df[col].dropna().values.astype(float)
        cur_vals = current_df[col].dropna().values.astype(float)

        if len(ref_vals) < 10 or len(cur_vals) < 10:
            continue

        psi = compute_psi(ref_vals, cur_vals)

        if psi > config.PSI_RETRAIN_THRESHOLD:
            status = "RETRAIN_NEEDED"
        elif psi > config.PSI_WARNING_THRESHOLD:
            status = "WARNING"
        else:
            status = "STABLE"

        results.append({
            "feature": col,
            "psi": round(psi, 4),
            "status": status,
        })

    drift_df = pd.DataFrame(results).sort_values("psi", ascending=False)

    # Log summary
    n_retrain = (drift_df["status"] == "RETRAIN_NEEDED").sum()
    n_warning = (drift_df["status"] == "WARNING").sum()
    logger.info(f"Feature drift: {n_retrain} need retraining, {n_warning} warnings, "
                f"{len(drift_df) - n_retrain - n_warning} stable")

    if n_retrain > 0:
        logger.warning(f"Features needing retraining: "
                       f"{drift_df[drift_df['status'] == 'RETRAIN_NEEDED']['feature'].tolist()}")

    return drift_df


def monitor_score_drift(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
) -> dict:
    """Monitor drift in model output scores (PD)."""
    psi = compute_psi(reference_scores, current_scores)

    result = {
        "score_psi": round(psi, 4),
        "reference_mean": round(float(np.mean(reference_scores)), 4),
        "current_mean": round(float(np.mean(current_scores)), 4),
        "reference_std": round(float(np.std(reference_scores)), 4),
        "current_std": round(float(np.std(current_scores)), 4),
    }

    if psi > config.PSI_RETRAIN_THRESHOLD:
        result["status"] = "RETRAIN_NEEDED"
        logger.warning(f"Score distribution drift detected! PSI={psi:.4f}")
    elif psi > config.PSI_WARNING_THRESHOLD:
        result["status"] = "WARNING"
        logger.info(f"Moderate score drift. PSI={psi:.4f}")
    else:
        result["status"] = "STABLE"
        logger.info(f"Score distribution stable. PSI={psi:.4f}")

    return result
