"""
Banking-grade model evaluation suite.
KS statistic, Gini coefficient, decile analysis, calibration curves, lift charts.
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    brier_score_loss
)
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    from sklearn.metrics import calibration_curve
from utils.logger import get_logger
from utils import config

logger = get_logger("evaluation")


def compute_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute Kolmogorov-Smirnov statistic – standard in credit risk."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks_stat = np.max(tpr - fpr)
    ks_threshold = thresholds[np.argmax(tpr - fpr)]
    return {
        "ks_statistic": float(ks_stat),
        "ks_threshold": float(ks_threshold),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "ks_values": (tpr - fpr).tolist(),
    }


def compute_gini_coefficient(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Gini = 2 * AUC - 1."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(2 * auc(fpr, tpr) - 1)


def compute_decile_analysis(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """Decile-wise default rate analysis – shows model discrimination."""
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df["decile"] = pd.qcut(df["y_prob"], 10, labels=False, duplicates="drop") + 1

    analysis = df.groupby("decile").agg(
        count=("y_true", "count"),
        default_count=("y_true", "sum"),
        mean_predicted_prob=("y_prob", "mean"),
        actual_default_rate=("y_true", "mean"),
    ).reset_index()

    analysis["cumulative_default_pct"] = (
        analysis["default_count"].cumsum() / analysis["default_count"].sum()
    )

    return analysis


def compute_calibration_data(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict:
    """Calibration curve data – predicted vs actual probabilities."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    brier = brier_score_loss(y_true, y_prob)
    return {
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
        "brier_score": float(brier),
    }


def compute_lift_chart(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Lift chart – how much better than random baseline."""
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df = df.sort_values("y_prob", ascending=False)
    df["decile"] = pd.qcut(range(len(df)), n_bins, labels=False, duplicates="drop") + 1

    baseline_rate = y_true.mean()
    lift_data = df.groupby("decile")["y_true"].mean() / baseline_rate

    return {
        "deciles": lift_data.index.tolist(),
        "lift": lift_data.values.tolist(),
        "baseline_rate": float(baseline_rate),
    }


def full_evaluation(y_true: np.ndarray, y_prob: np.ndarray, model_name: str = "PD") -> dict:
    """Run all banking-grade evaluations and save results."""
    logger.info(f"Running full evaluation for {model_name} model")

    results = {}

    # KS Statistic
    ks = compute_ks_statistic(y_true, y_prob)
    results["ks_statistic"] = ks["ks_statistic"]
    results["ks_threshold"] = ks["ks_threshold"]
    logger.info(f"  KS Statistic: {ks['ks_statistic']:.4f} at threshold {ks['ks_threshold']:.4f}")

    # Gini
    gini = compute_gini_coefficient(y_true, y_prob)
    results["gini"] = gini
    logger.info(f"  Gini Coefficient: {gini:.4f}")

    # Calibration
    cal = compute_calibration_data(y_true, y_prob)
    results["brier_score"] = cal["brier_score"]
    logger.info(f"  Brier Score: {cal['brier_score']:.4f}")

    # Decile analysis
    decile_df = compute_decile_analysis(y_true, y_prob)
    results["decile_analysis"] = decile_df.to_dict("records")
    logger.info(f"\nDecile Analysis:\n{decile_df.to_string(index=False)}")

    # Lift chart
    lift = compute_lift_chart(y_true, y_prob)
    results["lift_chart"] = lift
    logger.info(f"  Top decile lift: {lift['lift'][0]:.2f}x")

    # Save full results
    output_path = config.EVAL_ARTIFACTS_DIR / f"{model_name}_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"  Evaluation saved to {output_path}")

    # Save raw data for dashboard plots
    plot_data = {
        "ks_data": ks,
        "calibration_data": cal,
        "lift_data": lift,
        "decile_analysis": decile_df.to_dict("records"),
    }
    plot_path = config.EVAL_ARTIFACTS_DIR / f"{model_name}_plot_data.json"
    with open(plot_path, "w") as f:
        json.dump(plot_data, f, indent=2, default=str)

    return results
