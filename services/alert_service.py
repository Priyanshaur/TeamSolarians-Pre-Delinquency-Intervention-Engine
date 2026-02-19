"""
Alert Service – Smart alerting with dynamic thresholds, cooldown, and employer clustering.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from utils.logger import get_logger
from utils import config

logger = get_logger("alert_service")


class AlertService:
    """
    Generates actionable alerts for high-priority customers.
    - Dynamic threshold based on rolling percentile
    - Alert deduplication with cooldown window
    - Employer cluster grouping for systemic alerts
    - Severity tiers: CRITICAL / HIGH / MEDIUM
    """

    def __init__(self):
        self.alert_history = {}  # customer_id -> last_alert_datetime

    def _determine_severity(self, row: pd.Series) -> str:
        """Classify alert severity based on loss exposure."""
        loss_p99 = row.get("loss_p99", row.get("expected_loss", 0))
        if loss_p99 >= config.SEVERITY_THRESHOLDS["CRITICAL"]:
            return "CRITICAL"
        elif loss_p99 >= config.SEVERITY_THRESHOLDS["HIGH"]:
            return "HIGH"
        elif loss_p99 >= config.SEVERITY_THRESHOLDS["MEDIUM"]:
            return "MEDIUM"
        return "LOW"

    def _suggest_action(self, row: pd.Series) -> str:
        """Suggest intervention action based on customer profile."""
        pd_prob = row.get("pd_probability", 0)
        emi_ratio = row.get("emi_to_income_ratio", 0)
        intervention_eff = row.get("intervention_effectiveness", 0)
        liquidity = row.get("liquidity_buffer_ratio", 0)

        if pd_prob > 0.7 and emi_ratio > 0.6:
            return "EMI Restructuring – reduce monthly burden"
        elif pd_prob > 0.5 and liquidity < 1.0:
            return "EMI Holiday – 1-2 month payment pause"
        elif intervention_eff > 0.6 and pd_prob < 0.5:
            return "Proactive Reminder – automated payment nudge"
        elif pd_prob > 0.6:
            return "Personal Call – relationship manager outreach"
        else:
            return "Monitoring – continue observation"

    def _check_cooldown(self, customer_id: str, current_time: datetime) -> bool:
        """Check if customer is within cooldown window."""
        if customer_id in self.alert_history:
            last_alert = self.alert_history[customer_id]
            if (current_time - last_alert).days < config.ALERT_COOLDOWN_DAYS:
                return True  # Still in cooldown
        return False

    def generate_alerts(
        self,
        scored_df: pd.DataFrame,
        current_time: datetime = None,
    ) -> dict:
        """
        Generate alerts for customers exceeding dynamic threshold.

        Returns:
            dict with 'individual_alerts' and 'cluster_alerts'
        """
        current_time = current_time or datetime.now()
        logger.info(f"Generating alerts for {len(scored_df)} customers")

        # Dynamic threshold
        threshold = np.percentile(
            scored_df["priority_score"].values,
            config.ALERT_THRESHOLD_PERCENTILE,
        )
        logger.info(f"Dynamic threshold (P{config.ALERT_THRESHOLD_PERCENTILE}): {threshold:.2f}")

        # Filter above threshold
        alert_candidates = scored_df[scored_df["priority_score"] >= threshold].copy()
        logger.info(f"Candidates above threshold: {len(alert_candidates)}")

        # Apply cooldown
        individual_alerts = []
        for _, row in alert_candidates.iterrows():
            cid = row["customer_id"]
            if self._check_cooldown(cid, current_time):
                continue

            severity = self._determine_severity(row)
            action = self._suggest_action(row)

            alert = {
                "customer_id": cid,
                "employer_name": row.get("employer_name", "Unknown"),
                "severity": severity,
                "priority_score": round(row["priority_score"], 2),
                "pd_probability": round(row.get("pd_probability", 0), 4),
                "expected_loss": round(row.get("expected_loss", 0), 2),
                "loss_p95": round(row.get("loss_p95", 0), 2),
                "suggested_action": action,
                "timestamp": current_time.isoformat(),
            }
            individual_alerts.append(alert)
            self.alert_history[cid] = current_time

        # ── Employer cluster alerts ──
        cluster_alerts = []
        if individual_alerts:
            alert_df = pd.DataFrame(individual_alerts)
            employer_counts = alert_df["employer_name"].value_counts()
            systemic = employer_counts[employer_counts >= config.EMPLOYER_CLUSTER_MIN_SIZE]

            for employer, count in systemic.items():
                emp_alerts = alert_df[alert_df["employer_name"] == employer]
                cluster_alerts.append({
                    "employer_name": employer,
                    "affected_customers": count,
                    "total_exposure": round(emp_alerts["expected_loss"].sum(), 2),
                    "avg_pd": round(emp_alerts["pd_probability"].mean(), 4),
                    "severity": "SYSTEMIC",
                    "suggested_action": "Escalate to Risk Committee – cluster stress detected",
                    "customer_ids": emp_alerts["customer_id"].tolist(),
                    "timestamp": current_time.isoformat(),
                })

        logger.info(f"Generated {len(individual_alerts)} individual alerts, "
                    f"{len(cluster_alerts)} cluster alerts")

        return {
            "individual_alerts": individual_alerts,
            "cluster_alerts": cluster_alerts,
            "threshold": threshold,
            "total_candidates": len(alert_candidates),
            "after_cooldown": len(individual_alerts),
        }


def generate_alerts(scored_df: pd.DataFrame) -> dict:
    """Convenience function."""
    service = AlertService()
    return service.generate_alerts(scored_df)
