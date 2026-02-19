"""
Financial Triage AI – Main Entry Point
CLI orchestrator for all pipeline stages.
"""
import sys
import argparse
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger
from utils import config

logger = get_logger("main")


def run_generate_data():
    """Generate synthetic financial dataset."""
    logger.info("=" * 60)
    logger.info("STAGE 1: Data Generation")
    logger.info("=" * 60)
    from data.generate_data import generate_dataset
    df = generate_dataset()
    logger.info(f"Generated {len(df)} rows for {df['customer_id'].nunique()} customers")
    return df


def run_train():
    """Train all models."""
    logger.info("=" * 60)
    logger.info("STAGE 2: Model Training")
    logger.info("=" * 60)
    from models.train import train_all
    metrics = train_all()
    return metrics


def run_score():
    """Score all customers and generate rankings."""
    logger.info("=" * 60)
    logger.info("STAGE 3: Scoring Pipeline")
    logger.info("=" * 60)
    from services.scoring_pipeline import run_scoring_pipeline
    scored = run_scoring_pipeline()
    return scored


def run_explain(scored_df=None):
    """Generate SHAP explanations."""
    logger.info("=" * 60)
    logger.info("STAGE 4: Explainability (SHAP)")
    logger.info("=" * 60)
    import pandas as pd
    from models.pd_model import PDModel
    from services.explainability import ShapExplainer
    from services.feature_engineering import get_feature_columns

    if scored_df is None:
        path = config.DATA_DIR / "scored_customers.csv"
        scored_df = pd.read_csv(path)

    pd_model = PDModel.load()
    feature_cols = get_feature_columns()

    explainer = ShapExplainer(pd_model.model, feature_cols)
    shap_values = explainer.explain_batch(scored_df)
    explanations = explainer.save_explanations(scored_df, shap_values)

    logger.info(f"Generated SHAP explanations for {len(explanations)} customers")
    return explanations


def run_alerts(scored_df=None):
    """Generate alerts for high-priority customers."""
    logger.info("=" * 60)
    logger.info("STAGE 5: Alert Generation")
    logger.info("=" * 60)
    import pandas as pd
    from services.alert_service import generate_alerts

    if scored_df is None:
        path = config.DATA_DIR / "scored_customers.csv"
        scored_df = pd.read_csv(path)

    alerts = generate_alerts(scored_df)

    # Save alerts
    output_path = config.DATA_DIR / "alerts.json"
    serializable_alerts = {
        k: v for k, v in alerts.items()
        if k != "total_losses_distribution"
    }
    with open(output_path, "w") as f:
        json.dump(serializable_alerts, f, indent=2, default=str)

    logger.info(f"Individual alerts: {len(alerts['individual_alerts'])}")
    logger.info(f"Cluster alerts: {len(alerts['cluster_alerts'])}")

    # Print top alerts
    for alert in alerts["individual_alerts"][:5]:
        logger.info(
            f"  [{alert['severity']}] {alert['customer_id']} | "
            f"EL=${alert['expected_loss']:,.0f} | "
            f"Action: {alert['suggested_action']}"
        )

    return alerts


def run_recommendations(scored_df=None, shap_explanations=None):
    """Generate intervention recommendations."""
    logger.info("=" * 60)
    logger.info("STAGE 6: Intervention Recommendations")
    logger.info("=" * 60)
    import pandas as pd
    from services.agentic_layer import generate_batch_recommendations

    if scored_df is None:
        path = config.DATA_DIR / "scored_customers.csv"
        scored_df = pd.read_csv(path)

    if shap_explanations is None:
        shap_path = config.DATA_DIR / "shap_explanations.json"
        if shap_path.exists():
            with open(shap_path) as f:
                shap_explanations = json.load(f)
        else:
            shap_explanations = {}

    recommendations = generate_batch_recommendations(scored_df, shap_explanations, top_n=20)

    # Save recommendations
    output_path = config.DATA_DIR / "recommendations.json"
    with open(output_path, "w") as f:
        json.dump(recommendations, f, indent=2, default=str)

    # Print sample
    for rec in recommendations[:3]:
        logger.info(f"\n--- {rec['customer_id']} ---")
        logger.info(f"Summary: {rec['stress_summary'][:200]}...")
        logger.info(f"Intervention: {rec['recommended_intervention']['name']}")
        logger.info(f"Channel: {rec['recommended_channel']}")

    return recommendations


def run_dashboard():
    """Launch Streamlit dashboard."""
    import subprocess
    dashboard_path = PROJECT_ROOT / "dashboard" / "app.py"
    logger.info(f"Launching dashboard at {dashboard_path}")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path),
                    "--server.headless", "true"])


def run_all():
    """Run complete pipeline end-to-end."""
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║   Financial Triage AI – Full Pipeline Execution          ║")
    logger.info("╚" + "═" * 58 + "╝")

    # Stage 1: Generate Data
    df = run_generate_data()

    # Stage 2: Train Models
    metrics = run_train()

    # Stage 3: Score Customers
    scored = run_score()

    # Stage 4: SHAP Explanations
    explanations = run_explain(scored)

    # Stage 5: Generate Alerts
    alerts = run_alerts(scored)

    # Stage 6: Recommendations
    recommendations = run_recommendations(scored, explanations)

    logger.info("\n" + "=" * 60)
    logger.info("✅ FULL PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Customers scored:       {len(scored)}")
    logger.info(f"  Individual alerts:      {len(alerts['individual_alerts'])}")
    logger.info(f"  Cluster alerts:         {len(alerts['cluster_alerts'])}")
    logger.info(f"  Recommendations:        {len(recommendations)}")
    logger.info(f"\n  Launch dashboard:  python main.py --dashboard")

    return {
        "metrics": metrics,
        "n_scored": len(scored),
        "n_alerts": len(alerts["individual_alerts"]),
        "n_cluster_alerts": len(alerts["cluster_alerts"]),
        "n_recommendations": len(recommendations),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Financial Triage AI – Loss-Optimized Early Intervention Engine"
    )
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic dataset")
    parser.add_argument("--train", action="store_true", help="Train all models")
    parser.add_argument("--score", action="store_true", help="Score all customers")
    parser.add_argument("--explain", action="store_true", help="Generate SHAP explanations")
    parser.add_argument("--alerts", action="store_true", help="Generate alerts")
    parser.add_argument("--recommend", action="store_true", help="Generate recommendations")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")
    parser.add_argument("--run-all", action="store_true", help="Run complete pipeline")

    args = parser.parse_args()

    if args.run_all:
        run_all()
    elif args.generate_data:
        run_generate_data()
    elif args.train:
        run_train()
    elif args.score:
        run_score()
    elif args.explain:
        run_explain()
    elif args.alerts:
        run_alerts()
    elif args.recommend:
        run_recommendations()
    elif args.dashboard:
        run_dashboard()
    else:
        parser.print_help()
        print("\n💡 Quick start: python main.py --run-all")


if __name__ == "__main__":
    main()
