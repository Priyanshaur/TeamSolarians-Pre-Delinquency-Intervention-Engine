"""
Financial Stress Triage Dashboard
Pre-Delinquency Intervention Intelligence Platform
Institutional-grade risk monitoring interface.
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import config

# ─── Page Configuration ──────────────────────────────────────
st.set_page_config(
    page_title="Financial Stress Triage",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Design System ───────────────────────────────────────────
COLORS = {
    "bg": "#0B1120",
    "surface": "#111827",
    "surface_alt": "#1E293B",
    "primary": "#0F172A",
    "accent": "#14B8A6",
    "accent_dim": "#0D9488",
    "text": "#E2E8F0",
    "text_muted": "#94A3B8",
    "text_dim": "#64748B",
    "border": "#1E293B",
    "red": "#DC2626",
    "amber": "#F59E0B",
    "green": "#16A34A",
    "red_bg": "rgba(220,38,38,0.1)",
    "amber_bg": "rgba(245,158,11,0.1)",
    "green_bg": "rgba(22,163,74,0.1)",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, -apple-system, sans-serif", color=COLORS["text_muted"], size=12),
    margin=dict(l=0, r=0, t=30, b=0),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.04)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.04)"),
    hoverlabel=dict(bgcolor=COLORS["surface_alt"], font_size=12, font_family="Inter, sans-serif"),
)

# ─── Global Styles ────────────────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Root */
    .stApp {{
        background-color: {COLORS["bg"]};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    header[data-testid="stHeader"] {{
        background-color: {COLORS["bg"]};
        border-bottom: 1px solid {COLORS["border"]};
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {COLORS["primary"]};
        border-right: 1px solid {COLORS["border"]};
    }}

    /* Typography */
    h1, h2, h3, h4, h5, h6, p, span, div, label {{
        font-family: 'Inter', -apple-system, sans-serif !important;
    }}

    /* Remove default padding */
    .block-container {{
        padding: 1.5rem 2.5rem 2rem 2.5rem;
    }}

    /* KPI Card */
    .kpi-card {{
        background: {COLORS["surface"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        height: 100%;
    }}
    .kpi-label {{
        font-size: 0.7rem;
        font-weight: 500;
        color: {COLORS["text_dim"]};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }}
    .kpi-value {{
        font-size: 1.75rem;
        font-weight: 700;
        color: {COLORS["text"]};
        line-height: 1.2;
    }}
    .kpi-sub {{
        font-size: 0.75rem;
        color: {COLORS["text_dim"]};
        margin-top: 0.3rem;
    }}
    .kpi-accent {{ color: {COLORS["accent"]}; }}
    .kpi-red {{ color: {COLORS["red"]}; }}
    .kpi-amber {{ color: {COLORS["amber"]}; }}
    .kpi-green {{ color: {COLORS["green"]}; }}

    /* Section Headers */
    .section-header {{
        font-size: 0.8rem;
        font-weight: 600;
        color: {COLORS["text_dim"]};
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid {COLORS["border"]};
    }}

    /* Panel */
    .panel {{
        background: {COLORS["surface"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
    }}
    .panel-title {{
        font-size: 0.7rem;
        font-weight: 600;
        color: {COLORS["text_dim"]};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 1rem;
    }}

    /* Risk Badges */
    .badge {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 4px;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }}
    .badge-critical {{
        background: {COLORS["red_bg"]};
        color: {COLORS["red"]};
        border: 1px solid rgba(220,38,38,0.25);
    }}
    .badge-high {{
        background: {COLORS["amber_bg"]};
        color: {COLORS["amber"]};
        border: 1px solid rgba(245,158,11,0.25);
    }}
    .badge-medium {{
        background: rgba(100,116,139,0.15);
        color: {COLORS["text_dim"]};
        border: 1px solid rgba(100,116,139,0.25);
    }}
    .badge-stable {{
        background: {COLORS["green_bg"]};
        color: {COLORS["green"]};
        border: 1px solid rgba(22,163,74,0.25);
    }}

    /* Platform Header */
    .platform-header {{
        padding: 0.5rem 0 1.5rem 0;
    }}
    .platform-title {{
        font-size: 1.35rem;
        font-weight: 700;
        color: {COLORS["text"]};
        margin: 0;
    }}
    .platform-subtitle {{
        font-size: 0.75rem;
        font-weight: 400;
        color: {COLORS["text_dim"]};
        margin-top: 0.15rem;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        border-bottom: 1px solid {COLORS["border"]};
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        border-radius: 0;
        padding: 0.6rem 1.25rem;
        font-size: 0.8rem;
        font-weight: 500;
        color: {COLORS["text_dim"]};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: transparent;
        border-bottom: 2px solid {COLORS["accent"]};
        color: {COLORS["text"]};
    }}

    /* Dataframe styling */
    .stDataFrame {{
        border: 1px solid {COLORS["border"]};
        border-radius: 8px;
    }}

    /* Hide streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stDeployButton {{display: none;}}

    /* Selectbox */
    .stSelectbox label {{
        font-size: 0.75rem;
        color: {COLORS["text_dim"]};
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}

    /* Metric override */
    [data-testid="stMetric"] {{
        background: {COLORS["surface"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 8px;
        padding: 1rem;
    }}
    [data-testid="stMetricLabel"] {{
        font-size: 0.7rem !important;
        color: {COLORS["text_dim"]} !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    [data-testid="stMetricValue"] {{
        font-size: 1.5rem !important;
        color: {COLORS["text"]} !important;
    }}

    /* Drill-down panel */
    .drill-panel {{
        background: {COLORS["surface"]};
        border: 1px solid {COLORS["border"]};
        border-left: 3px solid {COLORS["accent"]};
        border-radius: 0 8px 8px 0;
        padding: 1.5rem;
        margin-top: 1rem;
    }}
    .drill-label {{
        font-size: 0.65rem;
        font-weight: 500;
        color: {COLORS["text_dim"]};
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    .drill-value {{
        font-size: 1rem;
        font-weight: 600;
        color: {COLORS["text"]};
    }}

    /* Explanation text */
    .explain-text {{
        font-size: 0.85rem;
        color: {COLORS["text_muted"]};
        line-height: 1.6;
        padding: 1rem;
        background: {COLORS["surface"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 8px;
    }}
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ─────────────────────────────────────────────
@st.cache_data
def load_scored():
    path = config.DATA_DIR / "scored_customers.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data
def load_shap():
    path = config.DATA_DIR / "shap_explanations.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

@st.cache_data
def load_feature_importance():
    path = config.DATA_DIR / "feature_importance.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data
def load_transactions():
    path = config.DATA_DIR / "transactions.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


# ─── Helpers ──────────────────────────────────────────────────
def classify_risk(row):
    loss = row.get("loss_p99", row.get("expected_loss", 0))
    if loss >= config.SEVERITY_THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    elif loss >= config.SEVERITY_THRESHOLDS["HIGH"]:
        return "HIGH"
    elif loss >= config.SEVERITY_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "STABLE"

def risk_badge(level):
    cls_map = {"CRITICAL": "badge-critical", "HIGH": "badge-high",
               "MEDIUM": "badge-medium", "STABLE": "badge-stable"}
    return f'<span class="badge {cls_map.get(level, "badge-medium")}">{level}</span>'

def fmt_currency(v):
    if abs(v) >= 1_000_000:
        return f"Rs. {v/1_000_000:,.2f}M"
    elif abs(v) >= 1_000:
        return f"Rs. {v/1_000:,.1f}K"
    return f"Rs. {v:,.0f}"

def fmt_pct(v):
    return f"{v*100:.2f}%"

def apply_plotly_layout(fig, **kwargs):
    layout = {**PLOTLY_LAYOUT, **kwargs}
    fig.update_layout(**layout)
    return fig


# ─── Load Data ────────────────────────────────────────────────
scored_df = load_scored()
if scored_df is None:
    st.markdown("""
    <div style="text-align:center; padding:4rem;">
        <p style="color:#94A3B8; font-size:1rem;">No scored data available.</p>
        <p style="color:#64748B; font-size:0.85rem;">Run the pipeline first: <code>python main.py --run-all</code></p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

scored_df["risk_tier"] = scored_df.apply(classify_risk, axis=1)
shap_data = load_shap()
txn_df = load_transactions()


# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="platform-header">
    <p class="platform-title">Financial Stress Triage Dashboard</p>
    <p class="platform-subtitle">Pre-Delinquency Intervention Intelligence Platform &mdash; Loss-Optimized Early Warning System</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tab_overview, tab_customers, tab_explain, tab_monitor = st.tabs([
    "Executive Overview", "Customer Intelligence", "Explainability", "Risk Monitor"
])


# ═══════════════════════════════════════════════════════════════
# TAB 1: EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════
with tab_overview:

    # ── KPI Row ──
    total_el = scored_df["expected_loss"].sum()
    portfolio_risk_idx = scored_df["pd_probability"].mean()
    high_priority = ((scored_df["risk_tier"] == "CRITICAL") | (scored_df["risk_tier"] == "HIGH")).sum()
    avg_intervention = scored_df["intervention_effectiveness"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Portfolio Risk Index</div>
            <div class="kpi-value kpi-{'red' if portfolio_risk_idx > 0.2 else 'amber' if portfolio_risk_idx > 0.1 else 'accent'}">{fmt_pct(portfolio_risk_idx)}</div>
            <div class="kpi-sub">Weighted avg default probability</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Expected Loss</div>
            <div class="kpi-value">{fmt_currency(total_el)}</div>
            <div class="kpi-sub">PD x Exposure x LGD across portfolio</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">High Priority Customers</div>
            <div class="kpi-value kpi-red">{high_priority:,}</div>
            <div class="kpi-sub">of {len(scored_df):,} total &mdash; {high_priority/len(scored_df)*100:.1f}% flagged</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Intervention Success Rate</div>
            <div class="kpi-value kpi-green">{fmt_pct(avg_intervention)}</div>
            <div class="kpi-sub">Estimated across eligible population</div>
        </div>""", unsafe_allow_html=True)

    # ── Risk Intelligence Section ──
    st.markdown('<div class="section-header">Risk Intelligence</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 0.9], gap="medium")

    with col_left:
        st.markdown('<div class="panel"><div class="panel-title">Expected Loss Trend by Week</div>', unsafe_allow_html=True)

        if txn_df is not None:
            # Build weekly EL proxy trend
            weekly_agg = txn_df.groupby("week_number").agg(
                avg_pd=("default_next_30_days", "mean"),
                avg_outstanding=("outstanding_balance", "mean"),
                n_customers=("customer_id", "nunique"),
            ).reset_index()
            weekly_agg["weekly_el"] = weekly_agg["avg_pd"] * weekly_agg["avg_outstanding"] * config.LGD_FIXED * weekly_agg["n_customers"]

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=weekly_agg["week_number"], y=weekly_agg["weekly_el"],
                mode="lines+markers",
                line=dict(color=COLORS["accent"], width=2),
                marker=dict(size=5, color=COLORS["accent"]),
                fill="tozeroy",
                fillcolor="rgba(20,184,166,0.06)",
                hovertemplate="Week %{x}<br>Expected Loss: Rs. %{y:,.0f}<extra></extra>",
            ))
            apply_plotly_layout(fig_trend, height=280,
                xaxis_title="Week", yaxis_title="Expected Loss (Rs.)",
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)"))
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.markdown(f'<p style="color:{COLORS["text_dim"]}; font-size:0.85rem;">Transaction data not available.</p>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="panel"><div class="panel-title">Loss Velocity by Employer Cluster</div>', unsafe_allow_html=True)

        emp_stats = scored_df.groupby("employer_name").agg(
            avg_pd=("pd_probability", "mean"),
            total_el=("expected_loss", "sum"),
            avg_velocity=("loss_velocity", "mean"),
            count=("customer_id", "count"),
            critical=("risk_tier", lambda x: (x == "CRITICAL").sum()),
        ).reset_index()
        emp_stats = emp_stats.sort_values("total_el", ascending=False).head(10)

        fig_heat = go.Figure(data=go.Heatmap(
            z=[emp_stats["avg_pd"].values, emp_stats["avg_velocity"].values, (emp_stats["critical"]/emp_stats["count"]).values],
            x=emp_stats["employer_name"],
            y=["Avg PD", "Loss Velocity", "Critical Ratio"],
            colorscale=[[0, COLORS["surface_alt"]], [0.5, COLORS["amber"]], [1, COLORS["red"]]],
            showscale=False,
            hovertemplate="Employer: %{x}<br>Metric: %{y}<br>Value: %{z:.3f}<extra></extra>",
        ))
        apply_plotly_layout(fig_heat, height=280,
            xaxis=dict(tickangle=-35, tickfont=dict(size=10)))
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Intervention ROI ──
    st.markdown('<div class="section-header">Intervention ROI Analysis</div>', unsafe_allow_html=True)

    col_roi_left, col_roi_right = st.columns([0.6, 0.4], gap="medium")

    with col_roi_left:
        st.markdown('<div class="panel"><div class="panel-title">Estimated Savings vs. Intervention Count</div>', unsafe_allow_html=True)

        roi_data = []
        for k in [10, 25, 50, 100, 200, 500]:
            if k > len(scored_df): break
            top_k = scored_df.nlargest(k, "priority_score")
            cost = k * 500
            savings = (top_k["expected_loss"] * top_k["intervention_effectiveness"]).sum()
            roi_data.append({"n": k, "savings": savings, "cost": cost, "net": savings - cost, "roi": (savings - cost) / cost * 100})

        roi_df = pd.DataFrame(roi_data)
        fig_roi = go.Figure()
        fig_roi.add_trace(go.Bar(
            x=roi_df["n"], y=roi_df["savings"],
            name="Est. Savings", marker_color=COLORS["accent"],
            hovertemplate="Intervene on %{x} customers<br>Savings: Rs. %{y:,.0f}<extra></extra>",
        ))
        fig_roi.add_trace(go.Bar(
            x=roi_df["n"], y=roi_df["cost"],
            name="Cost", marker_color=COLORS["text_dim"],
            hovertemplate="Cost: Rs. %{y:,.0f}<extra></extra>",
        ))
        apply_plotly_layout(fig_roi, height=250, barmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
            xaxis_title="Customers Intervened", yaxis_title="Amount (Rs.)")
        st.plotly_chart(fig_roi, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_roi_right:
        st.markdown('<div class="panel"><div class="panel-title">Risk Tier Distribution</div>', unsafe_allow_html=True)

        tier_counts = scored_df["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        color_map = {"CRITICAL": COLORS["red"], "HIGH": COLORS["amber"],
                     "MEDIUM": COLORS["text_dim"], "STABLE": COLORS["green"]}

        fig_dist = go.Figure(data=[go.Pie(
            labels=tier_counts["Tier"], values=tier_counts["Count"],
            hole=0.55,
            marker=dict(colors=[color_map.get(t, COLORS["text_dim"]) for t in tier_counts["Tier"]]),
            textinfo="label+percent",
            textfont=dict(size=11),
            hovertemplate="%{label}: %{value:,} customers (%{percent})<extra></extra>",
        )])
        apply_plotly_layout(fig_dist, height=250, showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2: CUSTOMER INTELLIGENCE
# ═══════════════════════════════════════════════════════════════
with tab_customers:

    st.markdown('<div class="section-header">Customer Priority Ranking</div>', unsafe_allow_html=True)

    # ── Filters ──
    fc1, fc2, fc3 = st.columns([1, 1, 1])
    with fc1:
        tier_filter = st.multiselect("Risk Tier", ["CRITICAL", "HIGH", "MEDIUM", "STABLE"], default=["CRITICAL", "HIGH"])
    with fc2:
        emp_filter = st.multiselect("Employer", sorted(scored_df["employer_name"].unique()))
    with fc3:
        top_n = st.slider("Display Limit", 10, min(500, len(scored_df)), 50)

    filtered = scored_df[scored_df["risk_tier"].isin(tier_filter)]
    if emp_filter:
        filtered = filtered[filtered["employer_name"].isin(emp_filter)]
    filtered = filtered.nlargest(top_n, "priority_score")

    # ── Table ──
    display = filtered[["priority_rank", "customer_id", "employer_name", "risk_tier",
                         "pd_probability", "expected_loss", "loss_velocity",
                         "recovery_p50", "intervention_effectiveness", "priority_score"]].copy()
    display.columns = ["Rank", "Customer ID", "Employer", "Risk Tier",
                        "PD (%)", "Expected Loss", "Loss Velocity",
                        "Recovery (days)", "Intervention Prob (%)", "Priority Score"]

    display["PD (%)"] = display["PD (%)"].apply(lambda x: f"{x*100:.2f}%")
    display["Expected Loss"] = display["Expected Loss"].apply(lambda x: f"Rs. {x:,.0f}")
    display["Loss Velocity"] = display["Loss Velocity"].apply(lambda x: f"{x:+.2%}")
    display["Recovery (days)"] = display["Recovery (days)"].apply(lambda x: f"{x:.0f}")
    display["Intervention Prob (%)"] = display["Intervention Prob (%)"].apply(lambda x: f"{x*100:.2f}%")
    display["Priority Score"] = display["Priority Score"].apply(lambda x: f"{x:,.0f}")

    st.dataframe(display, use_container_width=True, hide_index=True, height=400)

    # ── Drill-Down ──
    st.markdown('<div class="section-header">Customer Drill-Down</div>', unsafe_allow_html=True)

    customer_list = filtered["Customer ID"].tolist() if len(filtered) > 0 else scored_df.sort_values("priority_rank")["customer_id"].head(50).tolist()
    selected_id = st.selectbox("Select Customer", customer_list, label_visibility="collapsed")

    if selected_id:
        cust = scored_df[scored_df["customer_id"] == selected_id].iloc[0]

        # Metrics row
        dc1, dc2, dc3, dc4, dc5, dc6 = st.columns(6)
        with dc1:
            st.markdown(f"""<div class="drill-panel"><div class="drill-label">Probability of Default</div>
            <div class="drill-value kpi-{'red' if cust['pd_probability']>0.5 else 'amber' if cust['pd_probability']>0.2 else 'accent'}">{fmt_pct(cust['pd_probability'])}</div></div>""", unsafe_allow_html=True)
        with dc2:
            st.markdown(f"""<div class="drill-panel"><div class="drill-label">Expected Loss</div>
            <div class="drill-value">{fmt_currency(cust['expected_loss'])}</div></div>""", unsafe_allow_html=True)
        with dc3:
            st.markdown(f"""<div class="drill-panel"><div class="drill-label">Loss Velocity</div>
            <div class="drill-value kpi-{'red' if cust['loss_velocity']>0.5 else 'amber' if cust['loss_velocity']>0 else 'green'}">{cust['loss_velocity']:+.2%}</div></div>""", unsafe_allow_html=True)
        with dc4:
            st.markdown(f"""<div class="drill-panel"><div class="drill-label">Recovery Time</div>
            <div class="drill-value">{cust['recovery_p50']:.0f} days</div></div>""", unsafe_allow_html=True)
        with dc5:
            st.markdown(f"""<div class="drill-panel"><div class="drill-label">Intervention Probability</div>
            <div class="drill-value kpi-green">{fmt_pct(cust['intervention_effectiveness'])}</div></div>""", unsafe_allow_html=True)
        with dc6:
            st.markdown(f"""<div class="drill-panel"><div class="drill-label">Priority Score</div>
            <div class="drill-value kpi-accent">{cust['priority_score']:,.0f}</div></div>""", unsafe_allow_html=True)

        # SHAP + Profile
        drill_left, drill_right = st.columns([1.2, 0.8], gap="medium")

        with drill_left:
            st.markdown('<div class="panel"><div class="panel-title">Risk Factor Breakdown (SHAP)</div>', unsafe_allow_html=True)

            if selected_id in shap_data:
                top_feats = shap_data[selected_id].get("top_features", [])
                if top_feats:
                    f_names = [f["feature"].replace("_", " ").title() for f in top_feats]
                    f_vals = [f["shap_value"] for f in top_feats]
                    f_colors = [COLORS["red"] if v > 0 else COLORS["green"] for v in f_vals]

                    fig_shap = go.Figure(go.Bar(
                        x=f_vals, y=f_names, orientation="h",
                        marker_color=f_colors,
                        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
                    ))
                    apply_plotly_layout(fig_shap, height=280,
                        xaxis_title="SHAP Value (impact on default risk)",
                        yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_shap, use_container_width=True)
            else:
                st.markdown(f'<p style="color:{COLORS["text_dim"]}; font-size:0.85rem;">SHAP explanation not available for this customer.</p>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        with drill_right:
            st.markdown('<div class="panel"><div class="panel-title">Customer Profile</div>', unsafe_allow_html=True)

            profile = {
                "Employer": cust.get("employer_name", "N/A"),
                "Tenure": f"{cust.get('customer_tenure', 0):.0f} months",
                "Monthly Salary": fmt_currency(cust.get("salary_credit", 0)),
                "Savings Balance": fmt_currency(cust.get("savings_balance", 0)),
                "EMI Amount": fmt_currency(cust.get("emi_amount", 0)),
                "Outstanding Balance": fmt_currency(cust.get("outstanding_balance", 0)),
                "EMI / Income Ratio": fmt_pct(cust.get("emi_to_income_ratio", 0)),
                "Liquidity Buffer": f"{cust.get('liquidity_buffer_ratio', 0):.1f}x",
                "Stress Velocity": f"{cust.get('stress_velocity_score', 0):.3f}",
            }
            for k, v in profile.items():
                st.markdown(f"""<div style="display:flex; justify-content:space-between; padding:0.35rem 0; border-bottom:1px solid {COLORS['border']};">
                    <span style="font-size:0.8rem; color:{COLORS['text_dim']};">{k}</span>
                    <span style="font-size:0.8rem; font-weight:500; color:{COLORS['text']};">{v}</span>
                </div>""", unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # Recovery + Recommendation
        rec_left, rec_right = st.columns(2, gap="medium")

        with rec_left:
            st.markdown('<div class="panel"><div class="panel-title">Recovery Timeline</div>', unsafe_allow_html=True)

            recovery_data = pd.DataFrame({
                "Scenario": ["Optimistic (P10)", "Expected (P50)", "Pessimistic (P90)"],
                "Days": [cust.get("recovery_p10", 0), cust.get("recovery_p50", 0), cust.get("recovery_p90", 0)],
            })
            fig_rec = go.Figure(go.Bar(
                x=recovery_data["Days"], y=recovery_data["Scenario"], orientation="h",
                marker_color=[COLORS["green"], COLORS["amber"], COLORS["red"]],
                hovertemplate="%{y}: %{x:.0f} days<extra></extra>",
            ))
            apply_plotly_layout(fig_rec, height=180, showlegend=False,
                xaxis_title="Days to Recovery",
                yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_rec, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with rec_right:
            st.markdown('<div class="panel"><div class="panel-title">Recommended Intervention</div>', unsafe_allow_html=True)

            try:
                from services.agentic_layer import generate_full_recommendation
                shap_exp = shap_data.get(selected_id)
                rec = generate_full_recommendation(cust.to_dict(), shap_exp)

                st.markdown(f"""
                <div style="margin-bottom:1rem;">
                    <div class="drill-label">Strategy</div>
                    <div style="font-size:0.95rem; font-weight:600; color:{COLORS['text']}; margin-top:0.25rem;">{rec['recommended_intervention']['name']}</div>
                    <div style="font-size:0.8rem; color:{COLORS['text_muted']}; margin-top:0.2rem;">{rec['recommended_intervention']['description']}</div>
                </div>
                <div style="margin-bottom:1rem;">
                    <div class="drill-label">Channel</div>
                    <div style="font-size:0.85rem; color:{COLORS['text']}; margin-top:0.25rem;">{rec['recommended_channel'].replace('_', ' ').title()}</div>
                </div>
                <div>
                    <div class="drill-label">Assessment</div>
                    <div style="font-size:0.8rem; color:{COLORS['text_muted']}; line-height:1.5; margin-top:0.25rem;">{rec['stress_summary'][:300]}</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception:
                st.markdown(f'<p style="color:{COLORS["text_dim"]}; font-size:0.85rem;">Recommendation engine unavailable.</p>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 3: EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════
with tab_explain:

    st.markdown('<div class="section-header">Model Explainability</div>', unsafe_allow_html=True)

    ex_left, ex_right = st.columns([1, 1], gap="medium")

    with ex_left:
        st.markdown('<div class="panel"><div class="panel-title">Global Feature Importance (SHAP)</div>', unsafe_allow_html=True)

        feat_imp = load_feature_importance()
        if feat_imp is not None:
            feat_imp = feat_imp.sort_values("importance", ascending=True).tail(12)
            fig_global = go.Figure(go.Bar(
                x=feat_imp["importance"].values,
                y=[f.replace("_", " ").title() for f in feat_imp["feature"].values],
                orientation="h",
                marker_color=COLORS["accent"],
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            ))
            apply_plotly_layout(fig_global, height=400,
                xaxis_title="Mean |SHAP Value|")
            st.plotly_chart(fig_global, use_container_width=True)
        else:
            st.markdown(f'<p style="color:{COLORS["text_dim"]};">Feature importance data not available.</p>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with ex_right:
        st.markdown('<div class="panel"><div class="panel-title">Individual Customer Explanation</div>', unsafe_allow_html=True)

        explain_customer = st.selectbox(
            "Select customer for local explanation",
            list(shap_data.keys())[:50] if shap_data else ["No data"],
            key="explain_selectbox"
        )

        if explain_customer and explain_customer in shap_data:
            ex = shap_data[explain_customer]
            top_f = ex.get("top_features", [])

            if top_f:
                # SHAP waterfall
                f_names = [f["feature"].replace("_", " ").title() for f in top_f]
                f_vals = [f["shap_value"] for f in top_f]
                f_actual = [f.get("feature_value", "") for f in top_f]
                f_colors = [COLORS["red"] if v > 0 else COLORS["green"] for v in f_vals]

                fig_local = go.Figure(go.Bar(
                    x=f_vals, y=f_names, orientation="h",
                    marker_color=f_colors,
                    text=[f"val={a}" if a else "" for a in f_actual],
                    textposition="outside", textfont=dict(size=10, color=COLORS["text_dim"]),
                    hovertemplate="%{y}<br>SHAP: %{x:.4f}<extra></extra>",
                ))
                apply_plotly_layout(fig_local, height=350,
                    xaxis_title="SHAP Value",
                    yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_local, use_container_width=True)

                # Human-readable explanation
                increasing = [f for f in top_f if f["shap_value"] > 0]
                decreasing = [f for f in top_f if f["shap_value"] < 0]

                explanation_parts = []
                if increasing:
                    drivers = ", ".join([f["feature"].replace("_", " ") for f in increasing[:3]])
                    explanation_parts.append(f"Primary risk drivers include {drivers}.")
                if decreasing:
                    mitigants = ", ".join([f["feature"].replace("_", " ") for f in decreasing[:2]])
                    explanation_parts.append(f"Mitigating factors include {mitigants}.")

                cust_row = scored_df[scored_df["customer_id"] == explain_customer]
                if len(cust_row) > 0:
                    c = cust_row.iloc[0]
                    explanation_parts.append(
                        f"This customer has a {c['pd_probability']*100:.2f}% probability of default "
                        f"with an expected loss of {fmt_currency(c['expected_loss'])}. "
                        f"Estimated recovery time is {c['recovery_p50']:.0f} days under the median scenario."
                    )

                st.markdown(f'<div class="explain-text">{" ".join(explanation_parts)}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 4: RISK MONITOR
# ═══════════════════════════════════════════════════════════════
with tab_monitor:

    st.markdown('<div class="section-header">Portfolio Risk Monitoring</div>', unsafe_allow_html=True)

    if txn_df is not None:
        weekly = txn_df.groupby("week_number").agg(
            avg_savings=("savings_balance", "mean"),
            avg_salary_delay=("salary_delay_days", "mean"),
            total_failed=("failed_autodebit_flag", "sum"),
            avg_lending=("lending_app_upi_count", "mean"),
        ).reset_index()

        mon_c1, mon_c2 = st.columns(2, gap="medium")

        with mon_c1:
            st.markdown('<div class="panel"><div class="panel-title">Average Savings Balance</div>', unsafe_allow_html=True)
            fig_s = go.Figure(go.Scatter(
                x=weekly["week_number"], y=weekly["avg_savings"],
                mode="lines+markers",
                line=dict(color=COLORS["accent"], width=2),
                marker=dict(size=4),
                fill="tozeroy", fillcolor="rgba(20,184,166,0.05)",
                hovertemplate="Week %{x}<br>Rs. %{y:,.0f}<extra></extra>",
            ))
            apply_plotly_layout(fig_s, height=240, xaxis_title="Week", yaxis_title="Rs.")
            st.plotly_chart(fig_s, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="panel" style="margin-top:1rem;"><div class="panel-title">Failed Autodebit Count</div>', unsafe_allow_html=True)
            fig_f = go.Figure(go.Bar(
                x=weekly["week_number"], y=weekly["total_failed"],
                marker_color=COLORS["red"],
                hovertemplate="Week %{x}<br>Failed: %{y:,}<extra></extra>",
            ))
            apply_plotly_layout(fig_f, height=200, xaxis_title="Week", yaxis_title="Count")
            st.plotly_chart(fig_f, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with mon_c2:
            st.markdown('<div class="panel"><div class="panel-title">Average Salary Delay</div>', unsafe_allow_html=True)
            fig_d = go.Figure(go.Scatter(
                x=weekly["week_number"], y=weekly["avg_salary_delay"],
                mode="lines+markers",
                line=dict(color=COLORS["amber"], width=2),
                marker=dict(size=4),
                hovertemplate="Week %{x}<br>%{y:.1f} days<extra></extra>",
            ))
            apply_plotly_layout(fig_d, height=240, xaxis_title="Week", yaxis_title="Days")
            st.plotly_chart(fig_d, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="panel" style="margin-top:1rem;"><div class="panel-title">Lending App Usage Trend</div>', unsafe_allow_html=True)
            fig_l = go.Figure(go.Scatter(
                x=weekly["week_number"], y=weekly["avg_lending"],
                mode="lines+markers",
                line=dict(color=COLORS["text_dim"], width=2),
                marker=dict(size=4),
                hovertemplate="Week %{x}<br>%{y:.1f} avg UPI<extra></extra>",
            ))
            apply_plotly_layout(fig_l, height=200, xaxis_title="Week", yaxis_title="Avg Count")
            st.plotly_chart(fig_l, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Tail Risk Section
    st.markdown('<div class="section-header">Tail Risk Analysis</div>', unsafe_allow_html=True)

    tail_c1, tail_c2 = st.columns(2, gap="medium")

    with tail_c1:
        st.markdown('<div class="panel"><div class="panel-title">Expected Loss Distribution</div>', unsafe_allow_html=True)
        fig_hist = go.Figure(go.Histogram(
            x=scored_df["expected_loss"], nbinsx=40,
            marker_color=COLORS["accent"],
            hovertemplate="Rs. %{x:,.0f}<br>Count: %{y}<extra></extra>",
        ))
        apply_plotly_layout(fig_hist, height=280,
            xaxis_title="Expected Loss (Rs.)", yaxis_title="Frequency")
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tail_c2:
        st.markdown('<div class="panel"><div class="panel-title">P95 vs Expected Loss</div>', unsafe_allow_html=True)
        if "loss_p95" in scored_df.columns:
            color_map_scatter = {"CRITICAL": COLORS["red"], "HIGH": COLORS["amber"],
                                 "MEDIUM": COLORS["text_dim"], "STABLE": COLORS["green"]}
            fig_sc = px.scatter(
                scored_df, x="expected_loss", y="loss_p95",
                color="risk_tier", color_discrete_map=color_map_scatter,
                labels={"expected_loss": "Expected Loss (Rs.)", "loss_p95": "P95 Loss (Rs.)"},
                hover_data=["customer_id"],
            )
            fig_sc.add_traces(go.Scatter(
                x=[0, scored_df["expected_loss"].max()],
                y=[0, scored_df["expected_loss"].max()],
                mode="lines", line=dict(dash="dot", color="rgba(255,255,255,0.15)", width=1),
                showlegend=False,
            ))
            apply_plotly_layout(fig_sc, height=280,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)))
            st.plotly_chart(fig_sc, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Portfolio summary
    ps1, ps2, ps3 = st.columns(3)
    with ps1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Median Portfolio Loss</div>
        <div class="kpi-value">{fmt_currency(scored_df['expected_loss'].median())}</div></div>""", unsafe_allow_html=True)
    with ps2:
        if "loss_p95" in scored_df.columns:
            st.markdown(f"""<div class="kpi-card"><div class="kpi-label">P95 Tail Risk (Portfolio)</div>
            <div class="kpi-value kpi-amber">{fmt_currency(scored_df['loss_p95'].sum())}</div></div>""", unsafe_allow_html=True)
    with ps3:
        if "loss_p99" in scored_df.columns:
            st.markdown(f"""<div class="kpi-card"><div class="kpi-label">P99 Extreme Risk</div>
            <div class="kpi-value kpi-red">{fmt_currency(scored_df['loss_p99'].sum())}</div></div>""", unsafe_allow_html=True)
