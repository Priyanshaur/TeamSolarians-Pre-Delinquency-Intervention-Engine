"""
Agentic Layer – Offline template-based explanation generator.
Summarizes stress profiles, suggests interventions, generates outreach messages.
No external API calls – fully local inference using rule-based templates.
"""
import numpy as np
import pandas as pd
from utils.logger import get_logger

logger = get_logger("agentic_layer")


# ── Intervention Decision Matrix ──────────────────────────────
INTERVENTION_TYPES = {
    "emi_holiday": {
        "name": "EMI Holiday",
        "description": "Temporary suspension of EMI payments for 1-3 months",
        "conditions": lambda r: r.get("pd_probability", 0) > 0.4 and r.get("liquidity_buffer_ratio", 0) < 1.5,
        "priority": 1,
    },
    "restructuring": {
        "name": "Loan Restructuring",
        "description": "Extend tenure or reduce EMI to lower monthly burden",
        "conditions": lambda r: r.get("emi_to_income_ratio", 0) > 0.5 and r.get("pd_probability", 0) > 0.3,
        "priority": 2,
    },
    "payment_reminder": {
        "name": "Proactive Payment Reminder",
        "description": "Automated reminder via preferred channel before due date",
        "conditions": lambda r: r.get("pd_probability", 0) < 0.4 and r.get("intervention_effectiveness", 0) > 0.5,
        "priority": 3,
    },
    "relationship_call": {
        "name": "Relationship Manager Call",
        "description": "Personal call from relationship manager to understand situation",
        "conditions": lambda r: r.get("pd_probability", 0) > 0.5 and r.get("customer_tenure", 0) > 36,
        "priority": 2,
    },
    "monitoring": {
        "name": "Enhanced Monitoring",
        "description": "Increase monitoring frequency, no active intervention yet",
        "conditions": lambda r: True,  # Fallback
        "priority": 4,
    },
}

# ── Communication Channels ────────────────────────────────────
CHANNEL_LOGIC = {
    "sms": lambda r: r.get("pd_probability", 0) < 0.3,
    "app_notification": lambda r: r.get("pd_probability", 0) < 0.4 and r.get("customer_tenure", 0) < 24,
    "email": lambda r: r.get("pd_probability", 0) < 0.5,
    "phone_call": lambda r: r.get("pd_probability", 0) >= 0.5,
}


def suggest_intervention(customer_data: dict) -> dict:
    """Suggest best intervention based on customer profile."""
    matched = []
    for key, intervention in INTERVENTION_TYPES.items():
        if intervention["conditions"](customer_data):
            matched.append({
                "type": key,
                "name": intervention["name"],
                "description": intervention["description"],
                "priority": intervention["priority"],
            })

    matched.sort(key=lambda x: x["priority"])
    return matched[0] if matched else {
        "type": "monitoring",
        "name": "Enhanced Monitoring",
        "description": "Continue observation",
        "priority": 4,
    }


def suggest_channel(customer_data: dict) -> str:
    """Suggest best communication channel."""
    for channel, condition in CHANNEL_LOGIC.items():
        if condition(customer_data):
            return channel
    return "phone_call"


def generate_stress_summary(customer_data: dict, shap_explanation: dict = None) -> str:
    """Generate a human-readable stress narrative from customer data."""
    cid = customer_data.get("customer_id", "Unknown")
    pd_prob = customer_data.get("pd_probability", 0)
    el = customer_data.get("expected_loss", 0)
    emi_ratio = customer_data.get("emi_to_income_ratio", 0)
    savings_decline = customer_data.get("savings_decline_pct", 0)
    liquidity = customer_data.get("liquidity_buffer_ratio", 0)
    atm_spike = customer_data.get("atm_withdrawal_spike", 0)
    lending_spike = customer_data.get("lending_app_spike", 0)
    autodebit_streak = customer_data.get("failed_autodebit_streak", 0)
    stress_vel = customer_data.get("stress_velocity_score", 0)
    recovery_p50 = customer_data.get("recovery_p50", 0)
    employer = customer_data.get("employer_name", "Unknown")
    tenure = customer_data.get("customer_tenure", 0)

    # Risk level classification
    if pd_prob > 0.7:
        risk_level = "critically high"
    elif pd_prob > 0.5:
        risk_level = "high"
    elif pd_prob > 0.3:
        risk_level = "elevated"
    else:
        risk_level = "moderate"

    # Build narrative
    parts = [
        f"Customer {cid} (employed at {employer}, tenure: {tenure} months) "
        f"shows {risk_level} financial stress with a {pd_prob:.0%} probability of default "
        f"within 30 days. Potential exposure: ${el:,.0f}."
    ]

    # Add specific stress signals
    signals = []
    if emi_ratio > 0.6:
        signals.append(f"EMI-to-income ratio is strained at {emi_ratio:.0%}")
    if savings_decline < -0.05:
        signals.append(f"savings are declining at {savings_decline:.1%} per period")
    if liquidity < 2.0:
        signals.append(f"liquidity buffer is thin at {liquidity:.1f}× monthly EMI")
    if atm_spike > 1.5:
        signals.append("unusual ATM withdrawal pattern detected")
    if lending_spike > 1.5:
        signals.append("spike in lending app activity (possible debt cycling)")
    if autodebit_streak > 0:
        signals.append(f"{autodebit_streak} consecutive failed auto-debits")
    if stress_vel > 0.5:
        signals.append("financial stress is accelerating rapidly")

    if signals:
        parts.append("Key stress signals: " + "; ".join(signals) + ".")

    if recovery_p50 > 0:
        parts.append(
            f"Without intervention, estimated recovery timeline is approximately "
            f"{recovery_p50:.0f} days."
        )

    # Add SHAP-driven insights
    if shap_explanation and "top_features" in shap_explanation:
        top = shap_explanation["top_features"][:3]
        shap_insights = []
        for feat in top:
            name = feat["feature"].replace("_", " ").title()
            direction = feat["direction"]
            shap_insights.append(f"{name} ({direction})")
        if shap_insights:
            parts.append(
                "Primary risk drivers (from model): " + ", ".join(shap_insights) + "."
            )

    return " ".join(parts)


def generate_outreach_message(customer_data: dict, intervention: dict, channel: str) -> str:
    """Generate tone-adjusted outreach message."""
    pd_prob = customer_data.get("pd_probability", 0)
    cid = customer_data.get("customer_id", "Unknown")

    # Tone adjustment based on severity
    if pd_prob > 0.6:
        tone = "empathetic and urgent"
        greeting = "We understand you may be going through a challenging time"
    elif pd_prob > 0.3:
        tone = "supportive and proactive"
        greeting = "We're reaching out because we care about your financial wellness"
    else:
        tone = "friendly and informative"
        greeting = "As part of our commitment to your financial well-being"

    intervention_name = intervention.get("name", "assistance")
    intervention_desc = intervention.get("description", "")

    if channel == "sms":
        return (
            f"Hi, {greeting.lower()}. "
            f"We have a {intervention_name.lower()} option that may help. "
            f"Reply YES to learn more or call 1800-XXX-XXXX."
        )
    elif channel == "app_notification":
        return (
            f"{greeting}. 💡 You may be eligible for {intervention_name}: "
            f"{intervention_desc.lower()}. Tap to explore options."
        )
    elif channel == "email":
        return (
            f"Dear Valued Customer,\n\n"
            f"{greeting}, and we'd like to offer personalized support.\n\n"
            f"Based on our analysis, you may benefit from our "
            f"{intervention_name} program: {intervention_desc}.\n\n"
            f"This is a proactive measure to help you stay on track with your "
            f"financial obligations. There is no impact on your credit score "
            f"for exploring this option.\n\n"
            f"Please reach out to your relationship manager or call "
            f"1800-XXX-XXXX to discuss.\n\n"
            f"Warm regards,\n"
            f"Financial Wellness Team"
        )
    else:  # phone_call
        return (
            f"CALL SCRIPT ({tone}):\n"
            f"1. Greet customer warmly, reference their relationship history.\n"
            f"2. '{greeting}.'\n"
            f"3. Mention: 'We've noticed some changes in your account activity "
            f"and wanted to reach out proactively.'\n"
            f"4. Offer: {intervention_name} – {intervention_desc.lower()}.\n"
            f"5. Reassure: No credit score impact for discussion.\n"
            f"6. Ask about current situation, listen actively.\n"
            f"7. Schedule follow-up if needed."
        )


def generate_full_recommendation(customer_data: dict,
                                  shap_explanation: dict = None) -> dict:
    """Generate complete recommendation for a customer."""
    intervention = suggest_intervention(customer_data)
    channel = suggest_channel(customer_data)
    summary = generate_stress_summary(customer_data, shap_explanation)
    message = generate_outreach_message(customer_data, intervention, channel)

    return {
        "customer_id": customer_data.get("customer_id"),
        "stress_summary": summary,
        "recommended_intervention": intervention,
        "recommended_channel": channel,
        "outreach_message": message,
    }


def generate_batch_recommendations(scored_df: pd.DataFrame,
                                    shap_explanations: dict = None,
                                    top_n: int = 50) -> list:
    """Generate recommendations for top-N priority customers."""
    recommendations = []
    for i in range(min(top_n, len(scored_df))):
        row = scored_df.iloc[i]
        customer_data = row.to_dict()
        cid = customer_data.get("customer_id")
        shap_exp = shap_explanations.get(cid) if shap_explanations else None

        rec = generate_full_recommendation(customer_data, shap_exp)
        recommendations.append(rec)

    logger.info(f"Generated {len(recommendations)} recommendations")
    return recommendations
