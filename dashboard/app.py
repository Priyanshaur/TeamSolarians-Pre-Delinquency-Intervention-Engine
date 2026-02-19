"""
Streamlit Dashboard for Financial Triage AI.
6 pages: Executive Summary, Priority Customers, Risk Trends,
Loss Distribution, Cluster Heatmap, Customer Drill-Down.
"""
import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import config

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Triage AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(90deg, #1a73e8, #00bcd4);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f, #2a2a4a);
        padding: 1.2rem; border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 2rem; font-weight: 700; color: #00e5ff;
    }
    .metric-label {
        font-size: 0.85rem; color: #aaa; text-transform: uppercase; letter-spacing: 1px;
    }
    .severity-critical { color: #ff1744; font-weight: 700; }
    .severity-high { color: #ff9100; font-weight: 700; }
    .severity-medium { color: #ffd740; font-weight: 600; }
    .severity-low { color: #69f0ae; }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117, #161b22);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e2f;
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_scored_data():
    """Load scored customer data."""
    path = config.DATA_DIR / "scored_customers.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_shap_explanations():
    """Load SHAP explanations."""
    path = config.DATA_DIR / "shap_explanations.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_feature_importance():
    """Load feature importance."""
    path = config.DATA_DIR / "feature_importance.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_eval_data():
    """Load evaluation plot data."""
    path = config.EVAL_ARTIFACTS_DIR / "PD_plot_data.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_transaction_data():
    """Load raw transaction time-series data."""
    path = config.DATA_DIR / "transactions.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def severity_badge(severity: str) -> str:
    """Generate HTML severity badge."""
    colors = {
        "CRITICAL": "#ff1744", "HIGH": "#ff9100",
        "MEDIUM": "#ffd740", "LOW": "#69f0ae",
    }
    color = colors.get(severity, "#aaa")
    return f'<span style="background:{color};color:#000;padding:2px 10px;border-radius:12px;font-size:0.8rem;font-weight:600;">{severity}</span>'


def classify_severity(row):
    """Classify severity for display."""
    loss = row.get("loss_p99", row.get("expected_loss", 0))
    if loss >= config.SEVERITY_THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    elif loss >= config.SEVERITY_THRESHOLDS["HIGH"]:
        return "HIGH"
    elif loss >= config.SEVERITY_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
st.sidebar.markdown("## 🏦 Financial Triage AI")
st.sidebar.markdown("*Loss-Optimized Early Intervention Engine*")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Executive Summary",
        "🎯 Priority Customers",
        "📈 Risk Trends",
        "💰 Loss Distribution",
        "🔥 Cluster Heatmap",
        "🔍 Customer Drill-Down",
    ],
)

# Load data
scored_df = load_scored_data()

if scored_df is None:
    st.error("⚠️ No scored data found. Please run the pipeline first:\n"
             "```\npython main.py --run-all\n```")
    st.stop()

# Add severity column
scored_df["severity"] = scored_df.apply(classify_severity, axis=1)

# Sidebar stats
st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Stats")
st.sidebar.metric("Total Customers", f"{len(scored_df):,}")
st.sidebar.metric("Avg PD", f"{scored_df['pd_probability'].mean():.1%}")
st.sidebar.metric("Total Exposure", f"${scored_df['expected_loss'].sum():,.0f}")
critical_count = (scored_df["severity"] == "CRITICAL").sum()
st.sidebar.metric("Critical Alerts", f"{critical_count}")


# ══════════════════════════════════════════════════════════════
# PAGE 1: EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════
if page == "📊 Executive Summary":
    st.markdown('<p class="main-header">Executive Summary</p>', unsafe_allow_html=True)
    st.markdown("Real-time portfolio risk overview and intervention ROI estimates.")

    # ── Top-level KPIs ──
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_el = scored_df["expected_loss"].sum()
        st.metric("💰 Total Expected Loss", f"${total_el:,.0f}")
    with col2:
        high_risk = (scored_df["pd_probability"] > 0.5).sum()
        st.metric("⚠️ High-Risk Customers", f"{high_risk}")
    with col3:
        avg_recovery = scored_df["recovery_p50"].mean()
        st.metric("⏱️ Avg Recovery Time", f"{avg_recovery:.0f} days")
    with col4:
        # Estimated savings if top 50 are intervened
        top_50 = scored_df.nlargest(50, "priority_score")
        potential_savings = (top_50["expected_loss"] * top_50["intervention_effectiveness"]).sum()
        st.metric("💚 Est. Savings (Top 50)", f"${potential_savings:,.0f}")

    st.markdown("---")

    # ── Severity Distribution ──
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Risk Distribution")
        sev_counts = scored_df["severity"].value_counts().reset_index()
        sev_counts.columns = ["Severity", "Count"]
        color_map = {"CRITICAL": "#ff1744", "HIGH": "#ff9100", "MEDIUM": "#ffd740", "LOW": "#69f0ae"}
        fig = px.pie(sev_counts, values="Count", names="Severity",
                     color="Severity", color_discrete_map=color_map,
                     hole=0.45)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"), height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### Expected Loss by Employer")
        emp_loss = scored_df.groupby("employer_name")["expected_loss"].sum().sort_values(ascending=False).head(10)
        fig = px.bar(x=emp_loss.values, y=emp_loss.index, orientation="h",
                     labels={"x": "Expected Loss ($)", "y": "Employer"},
                     color=emp_loss.values, color_continuous_scale="Reds")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"), height=350, showlegend=False,
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── ROI Analysis ──
    st.markdown("---")
    st.markdown("#### 📈 Intervention ROI Analysis")

    # Simulate ROI by intervening on top-K customers
    roi_data = []
    for k in [10, 25, 50, 100, 200, 500]:
        if k > len(scored_df):
            break
        top_k = scored_df.nlargest(k, "priority_score")
        cost_per_intervention = 500  # assumed cost
        total_cost = k * cost_per_intervention
        savings = (top_k["expected_loss"] * top_k["intervention_effectiveness"]).sum()
        roi = (savings - total_cost) / total_cost * 100 if total_cost > 0 else 0
        roi_data.append({
            "Customers Intervened": k,
            "Intervention Cost ($)": total_cost,
            "Expected Savings ($)": round(savings),
            "Net Benefit ($)": round(savings - total_cost),
            "ROI (%)": round(roi, 1),
        })

    roi_df = pd.DataFrame(roi_data)
    st.dataframe(roi_df, use_container_width=True, hide_index=True)

    fig = px.line(roi_df, x="Customers Intervened", y="ROI (%)",
                  markers=True, title="ROI vs Number of Interventions")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2: PRIORITY CUSTOMERS
# ══════════════════════════════════════════════════════════════
elif page == "🎯 Priority Customers":
    st.markdown('<p class="main-header">Priority Customers</p>', unsafe_allow_html=True)
    st.markdown("Customers ranked by intervention priority score (descending).")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        severity_filter = st.multiselect(
            "Filter by Severity", ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            default=["CRITICAL", "HIGH"]
        )
    with col2:
        employer_filter = st.multiselect(
            "Filter by Employer", sorted(scored_df["employer_name"].unique())
        )
    with col3:
        top_n = st.slider("Show Top N", 10, min(500, len(scored_df)), 50)

    filtered = scored_df[scored_df["severity"].isin(severity_filter)]
    if employer_filter:
        filtered = filtered[filtered["employer_name"].isin(employer_filter)]
    filtered = filtered.nlargest(top_n, "priority_score")

    # Display table
    display_cols = [
        "priority_rank", "customer_id", "employer_name", "severity",
        "pd_probability", "expected_loss", "loss_velocity",
        "recovery_p50", "intervention_effectiveness", "priority_score",
    ]
    existing_cols = [c for c in display_cols if c in filtered.columns]

    styled = filtered[existing_cols].copy()
    styled["pd_probability"] = styled["pd_probability"].apply(lambda x: f"{x:.1%}")
    styled["expected_loss"] = styled["expected_loss"].apply(lambda x: f"${x:,.0f}")
    styled["priority_score"] = styled["priority_score"].apply(lambda x: f"{x:,.0f}")
    styled["loss_velocity"] = styled["loss_velocity"].apply(lambda x: f"{x:+.2%}")
    styled["recovery_p50"] = styled["recovery_p50"].apply(lambda x: f"{x:.0f} days")
    styled["intervention_effectiveness"] = styled["intervention_effectiveness"].apply(lambda x: f"{x:.0%}")

    st.dataframe(styled, use_container_width=True, hide_index=True, height=600)


# ══════════════════════════════════════════════════════════════
# PAGE 3: RISK TRENDS
# ══════════════════════════════════════════════════════════════
elif page == "📈 Risk Trends":
    st.markdown('<p class="main-header">Risk Trends</p>', unsafe_allow_html=True)

    txn_df = load_transaction_data()
    if txn_df is not None:
        # Aggregate weekly metrics
        weekly = txn_df.groupby("week_number").agg(
            avg_savings=("savings_balance", "mean"),
            avg_salary_delay=("salary_delay_days", "mean"),
            total_failed_autodebit=("failed_autodebit_flag", "sum"),
            avg_lending_app=("lending_app_upi_count", "mean"),
            avg_atm=("atm_withdrawals", "mean"),
        ).reset_index()

        # Each metric gets its own chart
        st.markdown("#### Average Savings Balance Over Time")
        fig1 = px.line(weekly, x="week_number", y="avg_savings",
                       labels={"week_number": "Week", "avg_savings": "Avg Savings ($)"})
        fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font=dict(color="white"))
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Average Salary Delay Trend")
        fig2 = px.line(weekly, x="week_number", y="avg_salary_delay",
                       labels={"week_number": "Week", "avg_salary_delay": "Avg Delay (days)"})
        fig2.update_traces(line_color="#ff9100")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font=dict(color="white"))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Failed Autodebit Count Over Time")
        fig3 = px.bar(weekly, x="week_number", y="total_failed_autodebit",
                      labels={"week_number": "Week", "total_failed_autodebit": "Failed Autodebits"})
        fig3.update_traces(marker_color="#ff1744")
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font=dict(color="white"))
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Lending App Usage Trend")
        fig4 = px.line(weekly, x="week_number", y="avg_lending_app",
                       labels={"week_number": "Week", "avg_lending_app": "Avg Lending App UPI Count"})
        fig4.update_traces(line_color="#ffd740")
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font=dict(color="white"))
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("Transaction data not found. Run data generation first.")


# ══════════════════════════════════════════════════════════════
# PAGE 4: LOSS DISTRIBUTION
# ══════════════════════════════════════════════════════════════
elif page == "💰 Loss Distribution":
    st.markdown('<p class="main-header">Loss Distribution</p>', unsafe_allow_html=True)
    st.markdown("Monte Carlo simulated loss distribution with tail risk metrics.")

    # Expected Loss histogram
    st.markdown("#### Expected Loss Distribution Across Portfolio")
    fig = px.histogram(scored_df, x="expected_loss", nbins=50,
                       labels={"expected_loss": "Expected Loss ($)", "count": "Customers"},
                       color_discrete_sequence=["#1a73e8"])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # P95 vs P50 loss scatter
    st.markdown("#### Tail Risk: P95 vs Expected Loss")
    if "loss_p95" in scored_df.columns:
        fig2 = px.scatter(
            scored_df, x="expected_loss", y="loss_p95",
            color="severity",
            color_discrete_map={"CRITICAL": "#ff1744", "HIGH": "#ff9100",
                                "MEDIUM": "#ffd740", "LOW": "#69f0ae"},
            labels={"expected_loss": "Expected Loss ($)", "loss_p95": "P95 Loss ($)"},
            hover_data=["customer_id", "pd_probability"],
        )
        fig2.add_traces(
            go.Scatter(
                x=[0, scored_df["expected_loss"].max()],
                y=[0, scored_df["expected_loss"].max()],
                mode="lines", line=dict(dash="dash", color="gray"),
                name="EL=P95 line",
            )
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Portfolio loss summary
    st.markdown("---")
    st.markdown("#### Portfolio Loss Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("P50 (Median) Loss", f"${scored_df['expected_loss'].median():,.0f}")
    with col2:
        if "loss_p95" in scored_df.columns:
            st.metric("P95 Tail Risk", f"${scored_df['loss_p95'].sum():,.0f}")
    with col3:
        if "loss_p99" in scored_df.columns:
            st.metric("P99 Extreme Risk", f"${scored_df['loss_p99'].sum():,.0f}")


# ══════════════════════════════════════════════════════════════
# PAGE 5: CLUSTER HEATMAP
# ══════════════════════════════════════════════════════════════
elif page == "🔥 Cluster Heatmap":
    st.markdown('<p class="main-header">Employer Cluster Risk Heatmap</p>', unsafe_allow_html=True)
    st.markdown("Identify systemic risk concentrations by employer.")

    # Aggregate by employer
    emp_stats = scored_df.groupby("employer_name").agg(
        customer_count=("customer_id", "count"),
        avg_pd=("pd_probability", "mean"),
        total_exposure=("expected_loss", "sum"),
        avg_stress_velocity=("stress_velocity_score", "mean"),
        critical_count=("severity", lambda x: (x == "CRITICAL").sum()),
        high_count=("severity", lambda x: (x == "HIGH").sum()),
    ).reset_index()

    emp_stats["risk_score"] = (
        emp_stats["avg_pd"] * 0.4 +
        (emp_stats["critical_count"] / emp_stats["customer_count"]) * 0.3 +
        emp_stats["avg_stress_velocity"].clip(0, 1) * 0.3
    )
    emp_stats = emp_stats.sort_values("risk_score", ascending=False)

    # Heatmap
    fig = px.treemap(
        emp_stats,
        path=["employer_name"],
        values="customer_count",
        color="avg_pd",
        color_continuous_scale="RdYlGn_r",
        hover_data=["total_exposure", "critical_count", "customer_count"],
        labels={"avg_pd": "Avg PD"},
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"), height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Employer Risk Details")
    display = emp_stats.copy()
    display["avg_pd"] = display["avg_pd"].apply(lambda x: f"{x:.1%}")
    display["total_exposure"] = display["total_exposure"].apply(lambda x: f"${x:,.0f}")
    display["risk_score"] = display["risk_score"].apply(lambda x: f"{x:.3f}")
    st.dataframe(display, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# PAGE 6: CUSTOMER DRILL-DOWN
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Customer Drill-Down":
    st.markdown('<p class="main-header">Customer Drill-Down</p>', unsafe_allow_html=True)

    # Customer selector
    customer_list = scored_df.sort_values("priority_rank")["customer_id"].tolist()
    selected_customer = st.selectbox(
        "Select Customer (ranked by priority)",
        customer_list,
        index=0,
    )

    customer = scored_df[scored_df["customer_id"] == selected_customer].iloc[0]

    # ── Key Metrics Row ──
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("PD", f"{customer['pd_probability']:.1%}")
    with col2:
        st.metric("Expected Loss", f"${customer['expected_loss']:,.0f}")
    with col3:
        st.metric("Loss Velocity", f"{customer['loss_velocity']:+.2%}")
    with col4:
        st.metric("Recovery (P50)", f"{customer['recovery_p50']:.0f} days")
    with col5:
        st.metric("Intervention Eff.", f"{customer['intervention_effectiveness']:.0%}")

    st.markdown("---")

    # ── SHAP Explanation ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 🧠 SHAP Risk Explanation")
        shap_data = load_shap_explanations()

        if selected_customer in shap_data:
            explanation = shap_data[selected_customer]
            top_features = explanation.get("top_features", [])

            if top_features:
                feat_names = [f["feature"].replace("_", " ").title() for f in top_features]
                feat_values = [f["shap_value"] for f in top_features]
                feat_colors = ["#ff1744" if v > 0 else "#69f0ae" for v in feat_values]

                fig = go.Figure(go.Bar(
                    x=feat_values, y=feat_names,
                    orientation="h",
                    marker_color=feat_colors,
                ))
                fig.update_layout(
                    title="Top Risk Drivers (SHAP)",
                    xaxis_title="SHAP Value (impact on risk)",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"), height=350,
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No SHAP features available for this customer.")
        else:
            st.info("SHAP explanation not available. Run: `python main.py --run-all`")

    with col_right:
        st.markdown("#### 📋 Customer Profile")
        profile_data = {
            "Employer": customer.get("employer_name", "N/A"),
            "Tenure (months)": customer.get("customer_tenure", "N/A"),
            "Salary": f"${customer.get('salary_credit', 0):,.0f}",
            "Savings": f"${customer.get('savings_balance', 0):,.0f}",
            "EMI": f"${customer.get('emi_amount', 0):,.0f}",
            "Outstanding": f"${customer.get('outstanding_balance', 0):,.0f}",
            "EMI/Income Ratio": f"{customer.get('emi_to_income_ratio', 0):.0%}",
            "Liquidity Buffer": f"{customer.get('liquidity_buffer_ratio', 0):.1f}×",
            "Stress Velocity": f"{customer.get('stress_velocity_score', 0):.3f}",
            "Priority Rank": f"#{int(customer.get('priority_rank', 0))}",
        }
        for k, v in profile_data.items():
            st.markdown(f"**{k}:** {v}")

    # ── Intervention Recommendation ──
    st.markdown("---")
    st.markdown("#### 🎯 AI-Generated Intervention Recommendation")

    # Generate recommendation inline
    try:
        from services.agentic_layer import generate_full_recommendation
        shap_exp = shap_data.get(selected_customer) if shap_data else None
        rec = generate_full_recommendation(customer.to_dict(), shap_exp)

        st.markdown(f"**Stress Summary:**\n> {rec['stress_summary']}")
        st.markdown(f"**Recommended Intervention:** {rec['recommended_intervention']['name']} — "
                    f"{rec['recommended_intervention']['description']}")
        st.markdown(f"**Channel:** {rec['recommended_channel'].replace('_', ' ').title()}")

        with st.expander("📞 View Outreach Message"):
            st.text(rec["outreach_message"])
    except Exception as e:
        st.warning(f"Recommendation generation unavailable: {e}")

    # ── Recovery Timeline ──
    st.markdown("---")
    st.markdown("#### ⏱️ Recovery Timeline")
    recovery_data = {
        "Scenario": ["Optimistic (P10)", "Expected (P50)", "Pessimistic (P90)"],
        "Days": [
            customer.get("recovery_p10", 0),
            customer.get("recovery_p50", 0),
            customer.get("recovery_p90", 0),
        ],
    }
    fig = px.bar(
        pd.DataFrame(recovery_data), x="Days", y="Scenario", orientation="h",
        color="Scenario",
        color_discrete_map={
            "Optimistic (P10)": "#69f0ae",
            "Expected (P50)": "#ffd740",
            "Pessimistic (P90)": "#ff1744",
        },
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"), height=250, showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
