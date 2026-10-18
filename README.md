# Pre-Delinquency Financial Stress & Intervention Intelligence Platform

A loss-optimized early financial stress detection system that identifies behavioral distress signals **2–4 weeks before a missed payment**, quantifies loss exposure, predicts recovery timelines, and prioritizes customers for proactive intervention based on financial impact and intervention effectiveness.

## Results

Evaluated on a hold-out set of 1,000 customers, from a synthetic portfolio of 5,000 customers with 12 weeks of behaviour each (about 13% default rate).

| Model | Metric | Value |
|---|---|---|
| Probability of default | AUC-ROC | 0.876 |
| | Gini | 0.752 |
| | KS statistic | 0.709 |
| | Brier score | 0.066 |
| | Top-decile lift | 5.12x (66% default rate vs 12.9% base) |
| Recovery time | MAE | 6.2 days |
| | R² | 0.906 |
| | 90% interval coverage | 74.6% (intervals are narrower than nominal) |
| Intervention effectiveness | AUC-ROC | 0.829 |

The data is synthetic and regenerated on each run, so figures vary slightly between runs. These results show that the pipeline works end to end, not how it would perform on real customers. See Limitations below.

## Core Capabilities

| Capability                  | Method                                                                           |
| --------------------------- | -------------------------------------------------------------------------------- |
| Early Stress Detection      | Multi-signal behavioral analysis across 11 engineered features                   |
| Probability of Default (PD) | XGBoost classifier with SMOTE resampling + Platt scaling calibration             |
| Expected Loss (EL)          | PD × Exposure × LGD with Monte Carlo tail risk simulation                        |
| Loss Velocity               | Rate-of-change measurement on expected loss trajectory                           |
| Recovery Time Prediction    | Quantile regression producing P10/P50/P90 prediction intervals                   |
| Intervention Effectiveness  | Logistic regression estimating P(intervention prevents default)                  |
| Priority Scoring            | EL × (1 + Loss Velocity) × (1 + Stress Velocity) × Intervention Effectiveness    |
| Explainability              | SHAP TreeExplainer with per-customer risk factor attribution                     |
| Smart Alerting              | Dynamic thresholds, 7-day cooldown deduplication, employer cluster detection     |
| Agentic Layer               | Rule-based intervention strategy, channel selection, outreach message generation |

## System Architecture

```
financial_triage/
├── data/                       # Data generation & validation
│   ├── generate_data.py        # Synthetic data: temporal patterns, employer cohorts, macro shocks
│   └── schemas.py              # Pandera validation schemas
├── models/                     # ML model layer
│   ├── pd_model.py             # PD classifier (XGBoost + SMOTE + manual Platt scaling)
│   ├── recovery_model.py       # Recovery regressor (XGBoost quantile regression)
│   ├── intervention_model.py   # Intervention effectiveness (Logistic Regression L2)
│   ├── evaluation.py           # Banking-grade metrics: KS, Gini, decile analysis, lift chart
│   └── train.py                # Training orchestrator
├── services/                   # Business logic & scoring
│   ├── feature_engineering.py  # 11 engineered behavioral stress features
│   ├── loss_engine.py          # EL calculation + Monte Carlo + priority scoring
│   ├── scoring_pipeline.py     # End-to-end scoring pipeline
│   ├── alert_service.py        # Smart alerting with employer cluster detection
│   ├── explainability.py       # SHAP integration
│   ├── agentic_layer.py        # Intervention recommendation engine
│   └── monitoring.py           # PSI drift detection for model monitoring
├── dashboard/
│   └── app.py                  # Institutional-grade Streamlit interface
├── tests/                      # pytest suite (features, loss engine, integration)
├── main.py                     # CLI orchestrator (7 pipeline stages)
└── requirements.txt
```

## Quick Start

```
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (generate → train → score → explain → alert → recommend)
python main.py --run-all

# Launch the dashboard
python main.py --dashboard
```

### Individual Pipeline Stages

```
python main.py --generate-data   # Generate synthetic dataset (5000 customers × 12 weeks)
python main.py --train            # Train PD, Recovery, and Intervention models
python main.py --score            # Score & rank all customers
python main.py --explain          # Generate SHAP explanations
python main.py --alerts           # Generate alerts with employer clustering
python main.py --recommend        # Generate intervention recommendations
python main.py --dashboard        # Launch Streamlit dashboard
```

## Model Architecture

### Model 1: Probability of Default

- **Algorithm**: XGBoost Classifier
- **Class Imbalance**: SMOTE oversampling on training set + cost-sensitive `scale_pos_weight`
- **Calibration**: Manual Platt scaling (sigmoid fit on calibration holdout)
- **Evaluation**: AUC-ROC, KS Statistic, Gini Coefficient, Decile Analysis, Calibration Curve, Lift Chart

### Model 2: Recovery Time

- **Algorithm**: XGBoost Regressor with quantile loss (`reg:quantileerror`)
- **Output**: P10 (optimistic), P50 (expected), P90 (pessimistic) prediction intervals
- **Evaluation**: MAE, RMSE, R², 90% prediction interval coverage

### Model 3: Intervention Effectiveness

- **Algorithm**: Logistic Regression with L2 regularization
- **Features**: Tenure, past response rate, autodebit reliability, prior restructuring success
- **Evaluation**: AUC-ROC

## Loss Engine

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

## Feature Engineering (11 Features)

| Feature                    | Description                                                     |
| -------------------------- | --------------------------------------------------------------- |
| `salary_delay_proxy`       | Rolling z-score of salary credit timing deviation               |
| `savings_decline_pct`      | EWMA week-over-week percentage drop in savings                  |
| `emi_to_income_ratio`      | EMI / Salary, capped at 1.5                                     |
| `lending_app_spike`        | Z-score vs rolling mean of lending app UPI count                |
| `atm_withdrawal_spike`     | Z-score vs rolling mean of ATM withdrawals                      |
| `stress_velocity_score`    | Weighted acceleration indicator from all signals                |
| `liquidity_buffer_ratio`   | Savings / Monthly EMI                                           |
| `employer_cluster_risk`    | Percentile rank of stress within employer group                 |
| `failed_autodebit_streak`  | Consecutive failed autodebit count                              |
| `payment_regularity_score` | Entropy-based payment timing regularity                         |
| `low_liquidity_high_emi`   | Interaction: low liquidity × high EMI ratio × declining savings |

## Dashboard

The dashboard is a custom-styled Streamlit application designed to resemble an institutional banking risk tool, not a generic data app.

**Tabs:**

1. **Executive Overview** — Portfolio Risk Index, Total Expected Loss, High Priority count, Intervention Success Rate, EL trend, Loss Velocity heatmap, ROI analysis
2. **Customer Intelligence** — Filterable priority ranking table with risk badges, customer drill-down with SHAP waterfall, profile, recovery timeline, and rule-based intervention recommendation
3. **Explainability** — Global SHAP feature importance, local per-customer explanation with human-readable narrative
4. **Risk Monitor** — Weekly savings, salary delay, autodebit failure, and lending app trends. Tail risk analysis with EL distribution and P95 scatter

## Production Architecture (Scalability Path)

While this prototype operates in batch mode with synthetic data, the system is designed to be production-extensible:

| Component            | Prototype            | Production Path                                                          |
| -------------------- | -------------------- | ------------------------------------------------------------------------ |
| **Data Ingestion**   | CSV batch generation | Apache Kafka for real-time transaction streaming                         |
| **Feature Store**    | In-memory pandas     | Feast or Tecton for versioned, low-latency feature serving               |
| **Orchestration**    | CLI (`main.py`)      | Apache Airflow DAGs for scheduled training/scoring pipelines             |
| **Model Serving**    | joblib serialization | MLflow Model Registry + BentoML for containerized model serving          |
| **Model Monitoring** | PSI in-memory        | Evidently AI or WhyLabs for automated drift detection and alerting       |
| **Dashboard**        | Streamlit            | Amazon QuickSight, Grafana, or internal React dashboard                  |
| **Alerting**         | JSON output          | Integration with ServiceNow, Jira, or internal CRM for ticket generation |
| **Storage**          | Local filesystem     | S3/GCS for artifacts, PostgreSQL/Snowflake for scored results            |

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────────┐
│   Kafka /    │───>│  Feast       │───>│  Airflow     │───>│   MLflow /    │
│   CDC Stream │    │  Feature     │    │  Orchestrator│    │   BentoML     │
│              │    │  Store       │    │              │    │   Model Server │
└─────────────┘    └──────────────┘    └──────────────┘    └───────────────┘
                                              │
                                              v
                                       ┌──────────────┐    ┌───────────────┐
                                       │  Scoring     │───>│  Dashboard /  │
                                       │  Pipeline    │    │  Alert System │
                                       └──────────────┘    └───────────────┘
```

## Limitations

- Uses synthetic data — real-world performance will vary
- The data generator is not seeded, so results change slightly from run to run
- Intervention effectiveness model uses heuristic synthetic labels (no real A/B test data)
- LGD distribution parameters are assumed, not calibrated to real portfolio
- The recovery model's 90% prediction intervals are too narrow (74.6% empirical coverage)
- Agentic layer uses template-based generation (no LLM API) — recommendations are rule-driven
- No real-time streaming support (batch scoring simulation only)

## Testing

```
pytest tests/ -v
```

Tests cover:

- Feature engineering: NaN/Inf checks, feature ranges, column presence
- Loss engine: EL = PD × EAD × LGD, loss velocity direction, priority score monotonicity, Monte Carlo properties
- Integration: End-to-end small-scale pipeline, alert generation
