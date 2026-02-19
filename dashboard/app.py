"""
Financial Stress Triage Dashboard
Pre-Delinquency Intervention Intelligence Platform
"""
import sys, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils import config

# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Stress Triage",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Minimal CSS (only what Streamlit can't do natively) ──────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif !important; }
    .block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 100%; }
    #MainMenu, footer, .stDeployButton { display: none; }
    div[data-testid="stMetric"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
    }
    div[data-testid="stMetricLabel"] > div {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        opacity: 0.7;
    }
    .risk-critical { color: #ef4444; font-weight: 700; }
    .risk-high { color: #f59e0b; font-weight: 700; }
    .risk-medium { color: #94a3b8; font-weight: 600; }
    .risk-stable { color: #22c55e; font-weight: 600; }
    .stTabs [data-baseweb="tab"] { font-size: 0.85rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ─── Color Palette ────────────────────────────────────────────
C = {
    "sky": "#0ea5e9",
    "teal": "#14b8a6",
    "red": "#ef4444",
    "amber": "#f59e0b",
    "green": "#22c55e",
    "slate": "#94a3b8",
    "dark": "#1e293b",
    "darker": "#0f172a",
}

PLOTLY_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94a3b8", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)"),
    hoverlabel=dict(bgcolor="#1e293b", font_size=12),
)


def styled_plotly(fig, **kw):
    fig.update_layout(**{**PLOTLY_DEFAULTS, **kw})
    return fig


# ─── Data Loading ─────────────────────────────────────────────
@st.cache_data
def load_scored():
    p = config.DATA_DIR / "scored_customers.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_shap():
    p = config.DATA_DIR / "shap_explanations.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}

@st.cache_data
def load_feature_importance():
    p = config.DATA_DIR / "feature_importance.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_transactions():
    p = config.DATA_DIR / "transactions.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_recommendations():
    p = config.DATA_DIR / "recommendations.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return []

@st.cache_data
def load_alerts():
    p = config.DATA_DIR / "alerts.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def classify_risk(row):
    v = row.get("loss_p99", row.get("expected_loss", 0))
    if v >= config.SEVERITY_THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    elif v >= config.SEVERITY_THRESHOLDS["HIGH"]:
        return "HIGH"
    elif v >= config.SEVERITY_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "STABLE"


def fmt_inr(v):
    if abs(v) >= 1e7:
        return f"Rs {v/1e7:,.2f} Cr"
    if abs(v) >= 1e5:
        return f"Rs {v/1e5:,.2f} L"
    if abs(v) >= 1e3:
        return f"Rs {v/1e3:,.1f} K"
    return f"Rs {v:,.0f}"


# ─── Load Data ────────────────────────────────────────────────
scored = load_scored()
if scored is None:
    st.error("No scored data. Run `python main.py --run-all` first.")
    st.stop()

scored["risk_tier"] = scored.apply(classify_risk, axis=1)
shap_data = load_shap()
txn_df = load_transactions()
feat_imp = load_feature_importance()
recs_data = load_recommendations()
alerts_data = load_alerts()

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
st.title("Financial Stress Triage Dashboard")
st.caption("Pre-Delinquency Intervention Intelligence Platform — Loss-Optimized Early Warning System")
st.divider()

# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Overview",
    "Customer Intelligence",
    "Explainability",
    "Risk Monitor",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════
with tab1:
    total_el = scored["expected_loss"].sum()
    avg_pd = scored["pd_probability"].mean()
    n_high = ((scored["risk_tier"] == "CRITICAL") | (scored["risk_tier"] == "HIGH")).sum()
    avg_interv = scored["intervention_effectiveness"].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Portfolio Risk Index", f"{avg_pd*100:.2f}%", help="Weighted average PD across all customers")
    k2.metric("Total Expected Loss", fmt_inr(total_el), help="Sum of EL = PD × EAD × LGD")
    k3.metric("High Priority Customers", f"{n_high:,} / {len(scored):,}", help="CRITICAL + HIGH risk tiers")
    k4.metric("Avg Intervention Success", f"{avg_interv*100:.1f}%", help="Mean estimated intervention effectiveness")

    st.markdown("---")

    # ── Row 2: EL Trend + Employer Heatmap ──
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.subheader("Expected Loss Trend by Week")
        if txn_df is not None:
            wagg = txn_df.groupby("week_number").agg(
                avg_pd=("default_next_30_days", "mean"),
                avg_out=("outstanding_balance", "mean"),
                n=("customer_id", "nunique"),
            ).reset_index()
            wagg["weekly_el"] = wagg["avg_pd"] * wagg["avg_out"] * config.LGD_FIXED * wagg["n"]

            fig = go.Figure(go.Scatter(
                x=wagg["week_number"], y=wagg["weekly_el"],
                mode="lines+markers",
                line=dict(color=C["sky"], width=2.5),
                marker=dict(size=6, color=C["sky"]),
                fill="tozeroy", fillcolor="rgba(14,165,233,0.08)",
                hovertemplate="Week %{x}<br>EL: Rs %{y:,.0f}<extra></extra>",
            ))
            styled_plotly(fig, height=320, xaxis_title="Week", yaxis_title="Expected Loss (Rs)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Transaction data not available.")

    with r2c2:
        st.subheader("Loss Velocity by Employer")
        emp = scored.groupby("employer_name").agg(
            avg_pd=("pd_probability", "mean"),
            total_el=("expected_loss", "sum"),
            avg_vel=("loss_velocity", "mean"),
            count=("customer_id", "count"),
        ).reset_index().sort_values("total_el", ascending=False).head(10)

        fig2 = go.Figure(go.Bar(
            x=emp["employer_name"], y=emp["total_el"],
            marker_color=[C["red"] if v > emp["total_el"].quantile(0.75) else C["amber"] if v > emp["total_el"].median() else C["sky"] for v in emp["total_el"]],
            hovertemplate="%{x}<br>Total EL: Rs %{y:,.0f}<extra></extra>",
        ))
        styled_plotly(fig2, height=320, xaxis_title="Employer", yaxis_title="Total Expected Loss (Rs)",
                      xaxis=dict(tickangle=-30, tickfont=dict(size=10)))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ── Row 3: Risk Distribution + ROI ──
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        st.subheader("Risk Tier Distribution")
        tier_ct = scored["risk_tier"].value_counts().reset_index()
        tier_ct.columns = ["Tier", "Count"]
        color_map = {"CRITICAL": C["red"], "HIGH": C["amber"], "MEDIUM": C["slate"], "STABLE": C["green"]}

        fig3 = go.Figure(go.Pie(
            labels=tier_ct["Tier"], values=tier_ct["Count"],
            hole=0.5,
            marker=dict(colors=[color_map.get(t, C["slate"]) for t in tier_ct["Tier"]]),
            textinfo="label+percent", textfont=dict(size=12),
            hovertemplate="%{label}: %{value:,} customers (%{percent})<extra></extra>",
        ))
        styled_plotly(fig3, height=300, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with r3c2:
        st.subheader("Intervention ROI Analysis")
        roi = []
        for k in [10, 25, 50, 100, 200, 500]:
            if k > len(scored):
                break
            top = scored.nlargest(k, "priority_score")
            cost = k * 500
            sav = (top["expected_loss"] * top["intervention_effectiveness"]).sum()
            roi.append({"Customers": k, "Savings": sav, "Cost": cost, "Net ROI": sav - cost})

        roi_df = pd.DataFrame(roi)
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(x=roi_df["Customers"], y=roi_df["Savings"], name="Est. Savings", marker_color=C["teal"]))
        fig4.add_trace(go.Bar(x=roi_df["Customers"], y=roi_df["Cost"], name="Intervention Cost", marker_color=C["slate"]))
        styled_plotly(fig4, height=300, barmode="group",
                      xaxis_title="Customers Intervened", yaxis_title="Amount (Rs)",
                      legend=dict(orientation="h", y=1.1, font=dict(size=11)))
        st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2 — CUSTOMER INTELLIGENCE
# ═══════════════════════════════════════════════════════════════
with tab2:

    # ── Filters ──
    st.subheader("Customer Priority Ranking")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        sel_tiers = st.multiselect("Risk Tier", ["CRITICAL", "HIGH", "MEDIUM", "STABLE"],
                                   default=["CRITICAL", "HIGH"], key="tier_f")
    with fc2:
        sel_emps = st.multiselect("Employer", sorted(scored["employer_name"].unique()), key="emp_f")
    with fc3:
        show_n = st.slider("Show top N", 10, min(500, len(scored)), 50, key="topn_f")

    filt = scored[scored["risk_tier"].isin(sel_tiers)]
    if sel_emps:
        filt = filt[filt["employer_name"].isin(sel_emps)]
    filt = filt.nlargest(show_n, "priority_score")

    # ── Display Table ──
    disp = filt[["priority_rank", "customer_id", "employer_name", "risk_tier",
                 "pd_probability", "expected_loss", "loss_velocity",
                 "recovery_p50", "intervention_effectiveness", "priority_score"]].copy()

    disp["pd_probability"] = (disp["pd_probability"] * 100).round(2).astype(str) + "%"
    disp["expected_loss"] = disp["expected_loss"].apply(lambda x: f"Rs {x:,.0f}")
    disp["loss_velocity"] = disp["loss_velocity"].apply(lambda x: f"{x:+.2%}")
    disp["recovery_p50"] = disp["recovery_p50"].round(0).astype(int).astype(str) + " days"
    disp["intervention_effectiveness"] = (disp["intervention_effectiveness"] * 100).round(2).astype(str) + "%"
    disp["priority_score"] = disp["priority_score"].apply(lambda x: f"{x:,.0f}")

    disp.columns = ["Rank", "Customer ID", "Employer", "Risk Tier",
                     "PD", "Expected Loss", "Loss Velocity",
                     "Recovery", "Intervention %", "Priority Score"]

    st.dataframe(disp, use_container_width=True, hide_index=True, height=400)

    st.markdown("---")

    # ── Drill-Down ──
    st.subheader("Customer Drill-Down")
    cid_list = filt["customer_id"].tolist()
    if not cid_list:
        cid_list = scored.nlargest(50, "priority_score")["customer_id"].tolist()

    sel_cid = st.selectbox("Select a customer to inspect", cid_list, key="drill_cid")

    if sel_cid:
        c = scored[scored["customer_id"] == sel_cid].iloc[0]

        # ── Metrics Row ──
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("PD", f"{c['pd_probability']*100:.2f}%")
        m2.metric("Expected Loss", fmt_inr(c['expected_loss']))
        m3.metric("Loss Velocity", f"{c['loss_velocity']:+.2%}")
        m4.metric("Recovery (P50)", f"{c['recovery_p50']:.0f} days")
        m5.metric("Intervention Prob", f"{c['intervention_effectiveness']*100:.1f}%")
        m6.metric("Priority Score", f"{c['priority_score']:,.0f}")

        st.markdown("---")

        # ── SHAP + Profile ──
        dl, dr = st.columns([3, 2])

        with dl:
            st.subheader("Risk Factor Breakdown (SHAP)")
            if sel_cid in shap_data:
                tf = shap_data[sel_cid].get("top_features", [])
                if tf:
                    names = [f["feature"].replace("_", " ").title() for f in tf]
                    vals = [f["shap_value"] for f in tf]
                    colors = [C["red"] if v > 0 else C["green"] for v in vals]

                    fig_shap = go.Figure(go.Bar(
                        x=vals, y=names, orientation="h",
                        marker_color=colors,
                        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
                    ))
                    styled_plotly(fig_shap, height=300,
                                  xaxis_title="SHAP Value (impact on default risk)",
                                  yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_shap, use_container_width=True)
                else:
                    st.info("No feature breakdown available.")
            else:
                st.info("SHAP explanation not available for this customer. SHAP is computed for top 50 priority customers only.")

        with dr:
            st.subheader("Customer Profile")
            profile_data = {
                "Employer": c.get("employer_name", "N/A"),
                "Customer Tenure": f"{c.get('customer_tenure', 0):.0f} months",
                "Monthly Salary": fmt_inr(c.get("salary_credit", 0)),
                "Savings Balance": fmt_inr(c.get("savings_balance", 0)),
                "EMI Amount": fmt_inr(c.get("emi_amount", 0)),
                "Outstanding Balance": fmt_inr(c.get("outstanding_balance", 0)),
                "EMI / Income Ratio": f"{c.get('emi_to_income_ratio', 0)*100:.1f}%",
                "Liquidity Buffer": f"{c.get('liquidity_buffer_ratio', 0):.2f}x",
                "Stress Velocity": f"{c.get('stress_velocity_score', 0):.4f}",
                "Risk Tier": c.get("risk_tier", "N/A"),
            }
            st.dataframe(
                pd.DataFrame(list(profile_data.items()), columns=["Attribute", "Value"]),
                use_container_width=True, hide_index=True,
            )

        st.markdown("---")

        # ── Recovery Timeline + Recommendation ──
        rl, rr = st.columns(2)

        with rl:
            st.subheader("Recovery Timeline")
            rec_df = pd.DataFrame({
                "Scenario": ["Optimistic (P10)", "Expected (P50)", "Pessimistic (P90)"],
                "Days": [c.get("recovery_p10", 0), c.get("recovery_p50", 0), c.get("recovery_p90", 0)],
            })
            fig_rec = go.Figure(go.Bar(
                x=rec_df["Days"], y=rec_df["Scenario"], orientation="h",
                marker_color=[C["green"], C["amber"], C["red"]],
                text=rec_df["Days"].apply(lambda d: f"{d:.0f} days"),
                textposition="outside",
                hovertemplate="%{y}: %{x:.0f} days<extra></extra>",
            ))
            styled_plotly(fig_rec, height=200, showlegend=False,
                          xaxis_title="Days to Recovery", yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_rec, use_container_width=True)

        with rr:
            st.subheader("Recommended Intervention")
            try:
                from services.agentic_layer import generate_full_recommendation
                shap_exp = shap_data.get(sel_cid)
                rec = generate_full_recommendation(c.to_dict(), shap_exp)

                st.markdown(f"**Strategy:** {rec['recommended_intervention']['name']}")
                st.markdown(f"_{rec['recommended_intervention']['description']}_")
                st.markdown(f"**Channel:** {rec['recommended_channel'].replace('_', ' ').title()}")
                st.markdown(f"**Assessment:** {rec['stress_summary'][:400]}")
            except Exception as e:
                # Fallback: use pre-generated recommendations
                cust_rec = [r for r in recs_data if r.get("customer_id") == sel_cid]
                if cust_rec:
                    r = cust_rec[0]
                    st.markdown(f"**Strategy:** {r['recommended_intervention']['name']}")
                    st.markdown(f"_{r['recommended_intervention']['description']}_")
                    st.markdown(f"**Channel:** {r['recommended_channel'].replace('_', ' ').title()}")
                    st.markdown(f"**Assessment:** {r['stress_summary'][:400]}")
                else:
                    st.info("No recommendation available for this customer.")


# ═══════════════════════════════════════════════════════════════
# TAB 3 — EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════
with tab3:

    el, er = st.columns(2)

    with el:
        st.subheader("Global Feature Importance (SHAP)")
        if feat_imp is not None:
            fi = feat_imp.sort_values("importance", ascending=True).tail(15)
            fig_gi = go.Figure(go.Bar(
                x=fi["importance"].values,
                y=[n.replace("_", " ").title() for n in fi["feature"].values],
                orientation="h",
                marker_color=C["sky"],
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            ))
            styled_plotly(fig_gi, height=450, xaxis_title="Mean |SHAP Value|")
            st.plotly_chart(fig_gi, use_container_width=True)
        else:
            st.warning("Feature importance data not available. Run `python main.py --explain` first.")

    with er:
        st.subheader("Individual Customer Explanation")
        shap_ids = list(shap_data.keys())[:50] if shap_data else []
        if not shap_ids:
            st.warning("No SHAP explanations available.")
        else:
            ex_cid = st.selectbox("Select customer", shap_ids, key="explain_sel")
            if ex_cid and ex_cid in shap_data:
                ex = shap_data[ex_cid]
                tf = ex.get("top_features", [])

                if tf:
                    names = [f["feature"].replace("_", " ").title() for f in tf]
                    vals = [f["shap_value"] for f in tf]
                    feat_vals = [f.get("feature_value", "") for f in tf]
                    colors = [C["red"] if v > 0 else C["green"] for v in vals]

                    fig_loc = go.Figure(go.Bar(
                        x=vals, y=names, orientation="h",
                        marker_color=colors,
                        text=[f"= {fv}" if fv != "" else "" for fv in feat_vals],
                        textposition="outside", textfont=dict(size=10, color=C["slate"]),
                        hovertemplate="%{y}<br>SHAP: %{x:.4f}<extra></extra>",
                    ))
                    styled_plotly(fig_loc, height=350, xaxis_title="SHAP Value",
                                  yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_loc, use_container_width=True)

                    # Human-readable explanation
                    incr = [f for f in tf if f["shap_value"] > 0]
                    decr = [f for f in tf if f["shap_value"] < 0]
                    parts = []
                    if incr:
                        drivers = ", ".join(f["feature"].replace("_", " ") for f in incr[:3])
                        parts.append(f"**Primary risk drivers:** {drivers}.")
                    if decr:
                        mits = ", ".join(f["feature"].replace("_", " ") for f in decr[:2])
                        parts.append(f"**Mitigating factors:** {mits}.")

                    crow = scored[scored["customer_id"] == ex_cid]
                    if len(crow) > 0:
                        cr = crow.iloc[0]
                        parts.append(
                            f"This customer has a **{cr['pd_probability']*100:.2f}%** probability of default "
                            f"with an expected loss of **{fmt_inr(cr['expected_loss'])}**. "
                            f"Estimated recovery: **{cr['recovery_p50']:.0f} days** (median)."
                        )

                    st.markdown(" ".join(parts))


# ═══════════════════════════════════════════════════════════════
# TAB 4 — RISK MONITOR
# ═══════════════════════════════════════════════════════════════
with tab4:

    # ── Weekly Trends ──
    st.subheader("Portfolio Trend Monitoring")

    if txn_df is not None:
        wk = txn_df.groupby("week_number").agg(
            avg_savings=("savings_balance", "mean"),
            avg_delay=("salary_delay_days", "mean"),
            total_fail=("failed_autodebit_flag", "sum"),
            avg_lend=("lending_app_upi_count", "mean"),
        ).reset_index()

        tc1, tc2 = st.columns(2)

        with tc1:
            st.markdown("##### Average Savings Balance")
            f1 = go.Figure(go.Scatter(
                x=wk["week_number"], y=wk["avg_savings"],
                mode="lines+markers", line=dict(color=C["teal"], width=2.5),
                marker=dict(size=5), fill="tozeroy", fillcolor="rgba(20,184,166,0.08)",
                hovertemplate="Week %{x}<br>Rs %{y:,.0f}<extra></extra>",
            ))
            styled_plotly(f1, height=250, xaxis_title="Week", yaxis_title="Rs")
            st.plotly_chart(f1, use_container_width=True)

        with tc2:
            st.markdown("##### Average Salary Delay")
            f2 = go.Figure(go.Scatter(
                x=wk["week_number"], y=wk["avg_delay"],
                mode="lines+markers", line=dict(color=C["amber"], width=2.5),
                marker=dict(size=5),
                hovertemplate="Week %{x}<br>%{y:.1f} days<extra></extra>",
            ))
            styled_plotly(f2, height=250, xaxis_title="Week", yaxis_title="Days")
            st.plotly_chart(f2, use_container_width=True)

        tc3, tc4 = st.columns(2)

        with tc3:
            st.markdown("##### Failed Autodebit Count")
            f3 = go.Figure(go.Bar(
                x=wk["week_number"], y=wk["total_fail"],
                marker_color=C["red"],
                hovertemplate="Week %{x}<br>Failed: %{y:,}<extra></extra>",
            ))
            styled_plotly(f3, height=250, xaxis_title="Week", yaxis_title="Count")
            st.plotly_chart(f3, use_container_width=True)

        with tc4:
            st.markdown("##### Lending App UPI Usage")
            f4 = go.Figure(go.Scatter(
                x=wk["week_number"], y=wk["avg_lend"],
                mode="lines+markers", line=dict(color=C["slate"], width=2.5),
                marker=dict(size=5),
                hovertemplate="Week %{x}<br>%{y:.1f} avg<extra></extra>",
            ))
            styled_plotly(f4, height=250, xaxis_title="Week", yaxis_title="Count")
            st.plotly_chart(f4, use_container_width=True)
    else:
        st.warning("Transaction data not available.")

    st.markdown("---")

    # ── Tail Risk ──
    st.subheader("Tail Risk Analysis")

    tl1, tl2 = st.columns(2)

    with tl1:
        st.markdown("##### Expected Loss Distribution")
        fig_h = go.Figure(go.Histogram(
            x=scored["expected_loss"], nbinsx=50,
            marker_color=C["sky"],
            hovertemplate="Rs %{x:,.0f}<br>Count: %{y}<extra></extra>",
        ))
        styled_plotly(fig_h, height=300, xaxis_title="Expected Loss (Rs)", yaxis_title="Frequency")
        st.plotly_chart(fig_h, use_container_width=True)

    with tl2:
        st.markdown("##### P95 vs Expected Loss (Tail Risk Scatter)")
        if "loss_p95" in scored.columns:
            scatter_colors = {"CRITICAL": C["red"], "HIGH": C["amber"],
                              "MEDIUM": C["slate"], "STABLE": C["green"]}
            fig_sc = px.scatter(
                scored, x="expected_loss", y="loss_p95",
                color="risk_tier", color_discrete_map=scatter_colors,
                hover_data=["customer_id"],
                labels={"expected_loss": "Expected Loss (Rs)", "loss_p95": "P95 Loss (Rs)"},
            )
            # 45-degree line
            mx = max(scored["expected_loss"].max(), scored["loss_p95"].max())
            fig_sc.add_trace(go.Scatter(
                x=[0, mx], y=[0, mx],
                mode="lines", line=dict(dash="dot", color="rgba(255,255,255,0.2)", width=1),
                showlegend=False,
            ))
            styled_plotly(fig_sc, height=300,
                          legend=dict(orientation="h", y=1.12, font=dict(size=10)))
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.info("P95 tail risk data not available.")

    # ── Portfolio Summary KPIs ──
    st.markdown("---")
    st.subheader("Portfolio Summary")

    ps1, ps2, ps3, ps4 = st.columns(4)
    ps1.metric("Median EL", fmt_inr(scored["expected_loss"].median()))
    if "loss_p95" in scored.columns:
        ps2.metric("P95 Portfolio Risk", fmt_inr(scored["loss_p95"].sum()))
    if "loss_p99" in scored.columns:
        ps3.metric("P99 Extreme Risk", fmt_inr(scored["loss_p99"].sum()))
    ps4.metric("Total Customers Scored", f"{len(scored):,}")

    # ── Alerts Summary ──
    if alerts_data:
        st.markdown("---")
        st.subheader("Active Alerts")
        ac1, ac2 = st.columns(2)

        with ac1:
            st.metric("Individual Alerts (after cooldown)",
                      alerts_data.get("after_cooldown", "N/A"),
                      delta=f"{alerts_data.get('before_cooldown', 0) - alerts_data.get('after_cooldown', 0)} deduplicated" if "before_cooldown" in alerts_data else None)

        with ac2:
            cluster = alerts_data.get("cluster_alerts", [])
            if cluster:
                st.metric("Employer Cluster Alerts", len(cluster))
                for ca in cluster:
                    st.markdown(f"- **{ca['employer_name']}**: {ca['affected_customers']} customers flagged")
            else:
                st.metric("Employer Cluster Alerts", 0)
