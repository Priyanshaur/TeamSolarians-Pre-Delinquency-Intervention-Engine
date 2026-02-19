# 🏦 Financial Triage AI – Loss-Optimized Early Intervention Engine

A production-style prototype that detects early financial stress in customers **2–4 weeks before a missed payment**, quantifies potential loss exposure, and prioritizes customers for proactive intervention based on financial impact and intervention effectiveness.

## 🎯 Core Capabilities

| Capability | Description |
|---|---|
| **Early Stress Detection** | Identifies financial stress signals from multi-source transaction data |
| **Default Probability (PD)** | XGBoost classifier with SMOTE + Platt scaling calibration |
| **Expected Loss** | EL = PD × EAD × LGD with Monte Carlo tail risk simulation |
| **Loss Velocity** | Rate at which potential loss is increasing |
| **Recovery Time Prediction** | Quantile regression with P10/P50/P90 prediction intervals |
| **Intervention Effectiveness** | Estimates probability intervention will prevent default |
| **Priority Scoring** | Ranks customers by Expected Loss × Stress Velocity × Intervention Effectiveness |
| **SHAP Explainability** | Per-customer risk factor explanations |
| **Smart Alerting** | Dynamic thresholds, cooldown dedup, employer cluster detection |
| **Agentic Layer** | Offline stress summaries, intervention suggestions, outreach messages |

## 🏗️ Architecture

```
financial_triage/
├── data/                     # Data generation & validation
│   ├── generate_data.py      # Synthetic data with temporal patterns
│   └── schemas.py            # Pandera validation schemas
├── models/                   # ML models
│   ├── pd_model.py           # PD classifier (XGBoost + SMOTE + Platt)
│   ├── recovery_model.py     # Recovery regressor (quantile regression)
│   ├── intervention_model.py # Intervention effectiveness (logistic reg)
│   ├── evaluation.py         # Banking-grade evaluation (KS, Gini, decile)
│   ├── train.py              # Training orchestrator
│   └── artifacts/            # Saved models & metrics
├── services/                 # Business logic
│   ├── feature_engineering.py # 11 engineered stress features
│   ├── loss_engine.py         # EL + Monte Carlo + Priority Score
│   ├── scoring_pipeline.py    # End-to-end scoring
│   ├── alert_service.py       # Smart alerting
│   ├── explainability.py      # SHAP integration
│   ├── agentic_layer.py       # Offline explanation generator
│   └── monitoring.py          # PSI drift detection
├── tests/                    # pytest suite
├── dashboard/                # Streamlit UI
│   └── app.py                # 6-page dashboard
├── main.py                   # CLI entry point
└── requirements.txt
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (generate → train → score → explain → alert)
python main.py --run-all

# 3. Launch the dashboard
python main.py --dashboard
```

### Individual Pipeline Stages

```bash
python main.py --generate-data   # Generate synthetic dataset
python main.py --train            # Train all 3 models
python main.py --score            # Score & rank customers
python main.py --explain          # Generate SHAP explanations
python main.py --alerts           # Generate alerts
python main.py --recommend        # Generate intervention recommendations
python main.py --dashboard        # Launch Streamlit dashboard
```

## 📊 Model Approach

### Model 1: Probability of Default (PD)
- **Algorithm**: XGBoost Classifier
- **Class Imbalance**: SMOTE oversampling on training set + cost-sensitive `scale_pos_weight`
- **Calibration**: Platt scaling (sigmoid) on held-out calibration set
- **Evaluation**: AUC-ROC, KS Statistic, Gini Coefficient, Decile Analysis, Calibration Curve, Lift Chart

### Model 2: Recovery Time
- **Algorithm**: XGBoost Regressor with quantile loss
- **Output**: P10 (optimistic), P50 (expected), P90 (pessimistic) prediction intervals
- **Evaluation**: MAE, RMSE, R², 90% prediction interval coverage

### Model 3: Intervention Effectiveness
- **Algorithm**: Logistic Regression with L2 regularization
- **Features**: Tenure, past response rate, autodebit reliability, prior restructuring success
- **Evaluation**: AUC-ROC

## 🧮 Loss Engine

```
Expected Loss   = PD × Outstanding Balance × LGD
Loss Velocity   = ΔEL / EL_{previous}
Priority Score  = EL × (1 + Loss Velocity) × (1 + Stress Velocity) × Intervention Effectiveness
```

**Monte Carlo Simulation (1,000 iterations):**
- PD sampled from Beta distribution around calibrated probability
- LGD sampled from Beta(α, β) distribution
- Outputs: P50, P95, P99 portfolio loss
- Supports stress scenarios: baseline, mild recession, severe recession, sector shock

## 📈 Feature Engineering (11 Features)

| Feature | Description |
|---|---|
| `salary_delay_proxy` | Rolling z-score of salary credit timing deviation |
| `savings_decline_pct` | EWMA week-over-week percentage drop in savings |
| `emi_to_income_ratio` | EMI / Salary, capped at 1.5 |
| `lending_app_spike` | Z-score vs rolling mean of lending app UPI count |
| `atm_withdrawal_spike` | Z-score vs rolling mean of ATM withdrawals |
| `stress_velocity_score` | Weighted acceleration indicator from all signals |
| `liquidity_buffer_ratio` | Savings / Monthly EMI |
| `employer_cluster_risk` | Percentile rank of stress within employer group |
| `failed_autodebit_streak` | Consecutive failed autodebit count |
| `payment_regularity_score` | Entropy-based payment timing regularity |
| `low_liquidity_high_emi` | Interaction: low liquidity × high EMI ratio × declining savings |

## 🖥️ Dashboard Pages

1. **Executive Summary** – Portfolio KPIs, severity distribution, employer exposure, ROI analysis
2. **Priority Customers** – Sortable/filterable table with severity badges
3. **Risk Trends** – Weekly trends for savings, salary delay, autodebits, lending app usage
4. **Loss Distribution** – Expected loss histogram, P95 vs EL scatter, portfolio tail risk
5. **Cluster Heatmap** – Employer risk treemap with drill-down details
6. **Customer Drill-Down** – Full profile: PD, EL, SHAP waterfall, recovery timeline, AI recommendation

## ⚠️ Limitations

- Uses **synthetic data** – real-world performance will vary
- Intervention effectiveness model uses **heuristic synthetic labels** (no real A/B test data)
- LGD distribution parameters are **assumed**, not calibrated to real portfolio
- Agentic layer uses **template-based** generation (no LLM API) – recommendations are rule-driven
- No real-time streaming support (batch scoring simulation only)

## 🔮 Future Improvements

- **Real data integration** – connect to banking data warehouse
- **Online learning** – update models incrementally as new data arrives
- **Basel III/IV alignment** – regulatory-compliant PD/LGD/EAD calculations
- **A/B testing framework** – measure actual intervention effectiveness
- **LLM integration** – connect to local LLM (e.g., Llama) for richer explanations
- **Real-time streaming** – Kafka/Spark integration for live scoring
- **Feature store** – centralized feature management (Feast/Tecton)
- **Model registry** – MLflow for model versioning and deployment tracking
