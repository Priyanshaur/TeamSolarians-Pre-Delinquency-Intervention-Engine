# Pre-Delinquency Financial Stress & Intervention Intelligence Platform
## Complete Project Documentation
### Team Solarians — Barclays Hackathon

---

## 1. Problem Statement

**Challenge:** Retail banking portfolios face significant losses from customer defaults that could have been prevented through early intervention. Traditional systems detect financial distress only AFTER a payment is missed — by then, recovery costs are high and customer relationships are damaged.

**Our Solution:** A loss-optimized early financial stress detection system that identifies behavioral distress signals **2–4 weeks before a missed payment**, quantifies potential loss exposure using Expected Loss modeling, predicts recovery timelines, estimates intervention effectiveness, and generates a composite **Priority Score** to rank customers for proactive outreach.

**Key Differentiator:** Unlike standard credit scoring (which gives a static risk score), this system measures the **velocity of financial deterioration**, uses Monte Carlo simulation for **tail risk analysis**, and combines multiple models into a single actionable decision layer.

---

## 2. Business Impact & Value Proposition

| Metric | Value |
|--------|-------|
| Portfolio size | 5,000 customers scored |
| Mean Default Probability | 12.52% |
| Total Expected Loss identified | Rs 20.9 Crore |
| High Priority customers flagged | CRITICAL + HIGH tiers |
| Intervention success rate | Estimated across eligible population |
| Top Decile Model Lift | 5.12x (the top 10% captures 5x more defaults than random) |

### ROI Analysis
- Intervening on top 100 priority customers: **savings far exceed the Rs 50,000 cost**
- The system pays for itself by preventing even a single high-value default
- Proactive outreach reduces collection costs by 60–80% vs reactive collections

---

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                             │
│   Synthetic Generator → 5,000 customers × 12 weeks of transactions │
│   (Production: Kafka CDC streams from core banking)                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING LAYER                          │
│   11 behavioral stress features extracted from raw transactions     │
│   Rolling z-scores, EWMA trends, entropy measures, interactions    │
│   (Production: Feast/Tecton feature store)                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     MODEL LAYER (3 Models)                          │
│                                                                     │
│   ┌─────────────┐   ┌──────────────┐   ┌────────────────────┐      │
│   │  PD Model   │   │  Recovery    │   │  Intervention      │      │
│   │  XGBoost    │   │  Model       │   │  Effectiveness     │      │
│   │  +SMOTE     │   │  Quantile    │   │  Model             │      │
│   │  +Platt     │   │  Regression  │   │  Logistic Reg      │      │
│   │  Scaling    │   │  P10/P50/P90 │   │  L2 Regularized    │      │
│   └─────────────┘   └──────────────┘   └────────────────────┘      │
│   (Production: MLflow Registry + BentoML serving)                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LOSS ENGINE LAYER                                │
│   Expected Loss = PD × Exposure × LGD                              │
│   Loss Velocity = Rate of EL change                                │
│   Monte Carlo: 1,000 iterations → P50, P95, P99 tail risk         │
│   Priority Score = EL × Velocity × Stress × Intervention          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DECISION LAYER                                   │
│                                                                     │
│   ┌──────────────┐ ┌────────────────┐ ┌────────────────────┐       │
│   │   Smart      │ │    SHAP        │ │    Agentic         │       │
│   │   Alerting   │ │    Explainer   │ │    Recommendation  │       │
│   │   + Cooldown │ │    Per-customer│ │    Engine           │       │
│   │   + Clusters │ │    attribution │ │    Strategy+Channel│       │
│   └──────────────┘ └────────────────┘ └────────────────────┘       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                               │
│   Streamlit Dashboard — 4 tabs                                     │
│   Executive Overview | Customer Intelligence | Explainability |    │
│   Risk Monitor                                                     │
│   (Production: Grafana / QuickSight / React)                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Directory Structure

```
financial_triage/
├── data/                       # Data layer
│   ├── generate_data.py        # Synthetic transaction generator
│   └── schemas.py              # Pandera validation schemas
├── models/                     # ML model layer
│   ├── pd_model.py             # PD classifier
│   ├── recovery_model.py       # Recovery time predictor
│   ├── intervention_model.py   # Intervention effectiveness
│   ├── evaluation.py           # Banking-grade metrics
│   ├── train.py                # Training orchestrator
│   └── artifacts/              # Saved models (joblib)
├── services/                   # Business logic
│   ├── feature_engineering.py  # 11 engineered features
│   ├── loss_engine.py          # EL + Monte Carlo + Priority
│   ├── scoring_pipeline.py     # End-to-end scoring
│   ├── alert_service.py        # Smart alerting
│   ├── explainability.py       # SHAP integration
│   ├── agentic_layer.py        # Intervention recommendations
│   └── monitoring.py           # PSI drift detection
├── dashboard/
│   └── app.py                  # Streamlit dashboard (4 tabs)
├── tests/                      # pytest suite
│   ├── test_features.py
│   ├── test_loss_engine.py
│   └── test_scoring.py
├── .streamlit/config.toml      # Dashboard theme
├── main.py                     # CLI orchestrator
├── requirements.txt
└── README.md
```

### 3.3 Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | Python 3.10+ | Core development |
| ML Framework | XGBoost | Gradient boosting for PD, Recovery |
| Imbalance Handling | imbalanced-learn (SMOTE) | Synthetic minority oversampling |
| Calibration | Manual Platt Scaling | Probability calibration |
| Explainability | SHAP (TreeExplainer) | Feature attribution |
| Data Validation | Pandera | DataFrame schema enforcement |
| Visualization | Plotly | Interactive charts |
| Dashboard | Streamlit | Web interface |
| Data Processing | pandas, numpy | Feature engineering |
| ML Utilities | scikit-learn | Preprocessing, metrics, splitting |
| Testing | pytest | Unit and integration tests |
| Serialization | joblib | Model persistence |

---

## 4. Data Pipeline

### 4.1 Synthetic Data Generation

Since real banking data is confidential, we built a **realistic synthetic data generator** that mimics real-world financial behavior patterns:

| Parameter | Value |
|-----------|-------|
| Customers | 5,000 unique |
| Time Span | 12 weeks of weekly transactions |
| Total Records | 60,000 (5,000 × 12) |
| Default Rate | ~15% (realistic retail banking) |
| Employers | 15 companies with varying risk profiles |

**Realistic patterns built into the generator:**
1. **Temporal correlation**: Financial stress builds gradually over weeks, not randomly
2. **Employer cohort effects**: Employees of distressed companies show correlated stress (e.g., mass layoffs)
3. **Macro shocks**: Random economic events that affect all customers simultaneously
4. **Seasonality**: Week-of-month patterns in salary credits and spending
5. **Behavioral cascading**: Late salary → savings drawdown → lending app usage → failed autodebit

### 4.2 Raw Features in Transaction Data

| Feature | Description | Range |
|---------|-------------|-------|
| customer_id | Unique identifier | CUST_00001 to CUST_05000 |
| week_number | Transaction week | 1–12 |
| salary_credit | Monthly salary amount | Rs 20K–200K |
| savings_balance | Current savings | Rs 0–500K |
| emi_amount | Monthly EMI payment | Rs 5K–50K |
| outstanding_balance | Total outstanding loan | Rs 50K–5M |
| salary_delay_days | Days salary was late | 0–15 days |
| lending_app_upi_count | UPI transactions to lending apps | 0–20 |
| atm_withdrawal_count | ATM withdrawal frequency | 0–15 |
| failed_autodebit_flag | Whether autodebit bounced | 0 or 1 |
| employer_name | Employer company | 15 categories |
| customer_tenure | Months as customer | 6–120 |
| default_next_30_days | TARGET: Will default within 30 days | 0 or 1 |

---

## 5. Feature Engineering (11 Engineered Features)

This is a critical differentiator — we go beyond raw data to extract **behavioral stress signals** that traditional models miss.

### 5.1 Feature Descriptions

| # | Feature | Formula / Method | Why It Matters |
|---|---------|-----------------|----------------|
| 1 | salary_delay_proxy | Rolling 4-week z-score of salary delay days | Increasing salary delays indicate employer-level financial stress |
| 2 | savings_decline_pct | EWMA (span=4) of week-over-week % savings drop | Sustained savings erosion is a leading indicator of default |
| 3 | emi_to_income_ratio | EMI ÷ Salary (capped at 1.5) | High EMI burden leaves no buffer for financial shocks |
| 4 | lending_app_spike | Z-score of lending app UPI vs rolling mean | Sudden borrowing from fintech apps signals liquidity crisis |
| 5 | atm_withdrawal_spike | Z-score of ATM withdrawals vs rolling mean | Cash hoarding behavior indicates loss of trust in digital banking |
| 6 | stress_velocity_score | Weighted sum of acceleration across all signals | Measures how FAST financial stress is accelerating |
| 7 | liquidity_buffer_ratio | Savings Balance ÷ Monthly EMI | Low buffer = one paycheck away from default |
| 8 | employer_cluster_risk | Percentile rank of stress within employer group | Captures systemic employer-level risk (mass layoffs, salary delays) |
| 9 | failed_autodebit_streak | Consecutive weeks with failed autodebit | Persistent payment failures signal inability, not forgetfulness |
| 10 | payment_regularity_score | Entropy of payment timing patterns | Irregular payments suggest financial unpredictability |
| 11 | low_liquidity_high_emi | Interaction: low buffer × high EMI × declining savings | Compound stress indicator — multiple simultaneous risks |

### 5.2 Stress Velocity Weights

The stress velocity score uses weighted acceleration across 6 signals:

```
Stress Velocity = 0.20 × salary_delay_proxy
                + 0.25 × savings_decline_pct
                + 0.15 × emi_to_income_ratio
                + 0.15 × lending_app_spike
                + 0.10 × atm_withdrawal_spike
                + 0.15 × failed_autodebit_streak
```

---

## 6. Machine Learning Models

### 6.1 Model 1: Probability of Default (PD)

**Objective:** Predict whether a customer will default within the next 30 days.

| Attribute | Detail |
|-----------|--------|
| Algorithm | XGBoost Classifier |
| Class Imbalance | SMOTE oversampling + cost-sensitive scale_pos_weight |
| Calibration | Manual Platt Scaling (sigmoid fit on held-out calibration set) |
| Train/Cal/Test Split | 65% / 15% / 20% |
| n_estimators | 300 |
| max_depth | 6 |
| learning_rate | 0.05 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |

**Performance Metrics:**

| Metric | Value | Assessment |
|--------|-------|------------|
| KS Statistic | 0.6653 | Excellent (>0.4 is strong) |
| Gini Coefficient | 0.6862 | Very Good (industry standard: 0.3–0.7) |
| Brier Score | 0.0705 | Well calibrated (lower is better) |
| Top Decile Lift | 5.12x | Top 10% captures 5x more defaults than random |

**Why XGBoost?**
- Handles non-linear feature interactions naturally
- Built-in regularization prevents overfitting
- Fast training and inference (important for production)
- Industry standard for credit risk modeling

**Why SMOTE?**
- Default rate is ~15%, creating class imbalance
- SMOTE generates synthetic minority samples in feature space
- Applied ONLY on the training set (not test or calibration)
- Combined with `scale_pos_weight` for double protection

**Why Platt Scaling?**
- Raw XGBoost outputs are not well-calibrated probabilities
- Platt scaling fits a sigmoid curve to convert raw scores into true probabilities
- Critical for Expected Loss calculation (EL depends on calibrated PD)
- We used manual implementation (LogisticRegression on raw scores) for sklearn 1.8+ compatibility

### 6.2 Model 2: Recovery Time Prediction

**Objective:** Predict how many days a customer will take to recover from financial stress.

| Attribute | Detail |
|-----------|--------|
| Algorithm | XGBoost Regressor with Quantile Loss |
| Quantiles | P10 (optimistic), P50 (expected), P90 (pessimistic) |
| Loss Function | `reg:quantileerror` |
| Uses | Three separate models, one per quantile |

**Output:** For each customer, three recovery time estimates:
- **P10 (Optimistic):** 90% of similar customers take longer
- **P50 (Expected):** Median recovery time
- **P90 (Pessimistic):** Only 10% of similar customers take longer

**Why Quantile Regression?**
- Point estimates (mean prediction) hide uncertainty
- Quantile regression gives prediction intervals
- Risk managers can plan for the pessimistic scenario (P90)
- More informative for resource allocation decisions

### 6.3 Model 3: Intervention Effectiveness

**Objective:** Predict the probability that proactive intervention will prevent default.

| Attribute | Detail |
|-----------|--------|
| Algorithm | Logistic Regression with L2 Regularization |
| Key Features | Tenure, past response rate, autodebit reliability, restructuring history |
| Output | P(intervention prevents default) ∈ [0, 1] |

**Why Logistic Regression?**
- Interpretable coefficients (important for regulatory compliance)
- Stable with limited training data
- Sufficient for this task (intervention effectiveness is more linear)
- Coefficient signs can be validated against business intuition

---

## 7. Loss Engine

### 7.1 Expected Loss (EL)

The foundation of credit risk quantification:

```
Expected Loss = PD × EAD × LGD

Where:
  PD  = Probability of Default (from Model 1)
  EAD = Exposure at Default (Outstanding Balance)
  LGD = Loss Given Default (0.45 fixed, or Beta-distributed)
```

### 7.2 Loss Velocity

Measures how fast potential loss is increasing:

```
Loss Velocity = (EL_current - EL_previous) / EL_previous

Positive velocity → worsening trajectory
Negative velocity → improving trajectory
```

### 7.3 Priority Score

The composite score that ranks customers for intervention:

```
Priority Score = EL × (1 + |Loss Velocity|) × (1 + Stress Velocity) × Intervention Effectiveness
```

This ensures the highest-priority customers are those with:
- High expected loss (large financial impact)
- Fast-worsening trajectory (time-sensitive)
- High stress velocity (accelerating deterioration)
- High intervention effectiveness (worth intervening on)

### 7.4 Monte Carlo Simulation

For tail risk analysis, we run 1,000 iterations:

```
For each iteration:
  1. Sample PD from Beta(α, β) centered on calibrated PD
  2. Sample LGD from Beta(4.0, 5.0) → mean ≈ 0.44
  3. Calculate EL = PD_sampled × EAD × LGD_sampled
  4. Aggregate across portfolio
```

**Outputs:**
- P50: Median portfolio loss
- P95: Loss exceeded only 5% of the time (regulatory VaR)
- P99: Extreme tail risk

**Stress Scenarios:**
| Scenario | PD Multiplier | LGD Multiplier |
|----------|---------------|-----------------|
| Baseline | 1.0x | 1.0x |
| Mild Recession | 1.3x | 1.1x |
| Severe Recession | 1.5x | 1.2x |
| Sector Shock | 1.8x | 1.3x |

---

## 8. Explainability (SHAP)

### 8.1 Why SHAP?
- Regulatory requirement: Banks must explain why a customer is flagged
- SHAP (SHapley Additive exPlanations) provides mathematically consistent feature attributions
- Based on cooperative game theory (Shapley values)
- TreeExplainer is exact for tree-based models (no approximation error)

### 8.2 Two Levels of Explanation

**Global Importance:** Which features matter most across all customers?
- Computed as mean |SHAP value| for each feature
- Saves as `feature_importance.csv`

**Local Explanation:** Why is THIS specific customer flagged?
- Per-customer top-N feature attributions
- Positive SHAP = increases default risk
- Negative SHAP = decreases default risk
- Human-readable narrative generated from SHAP values

### 8.3 Example Explanation (Human-Readable)

> "Primary risk drivers include savings decline, EMI to income ratio, and lending app spike. Mitigating factors include payment regularity score. This customer has a 67.32% probability of default with an expected loss of Rs 2.41 L. Estimated recovery time is 84 days under the median scenario."

---

## 9. Smart Alerting System

### 9.1 Alert Generation Logic

1. **Dynamic Thresholds:** Alert threshold set at P75 of priority scores (not a fixed cutoff)
2. **Severity Tiers:**
   - CRITICAL: P99 loss > Rs 50,000
   - HIGH: P99 loss > Rs 20,000
   - MEDIUM: P99 loss > Rs 5,000
3. **Cooldown Deduplication:** 7-day cooldown to prevent alert fatigue
4. **Employer Cluster Detection:** If 3+ employees from the same employer are flagged, trigger a systemic alert

### 9.2 Alert Output
- Individual customer alerts with severity and suggested action
- Employer cluster alerts (e.g., "MediaFlow: 141 customers flagged — potential mass layoff")  
- Deduplication stats (before vs after cooldown)

---

## 10. Agentic Recommendation Layer

### 10.1 What It Does
An offline rule-based system that generates:
1. **Intervention Strategy** — EMI restructuring, payment holiday, etc.
2. **Preferred Channel** — App notification, SMS, phone call, or RM visit
3. **Stress Narrative** — Human-readable summary of the customer's situation
4. **Outreach Message Draft** — Tone-adjusted communication

### 10.2 Intervention Strategies

| Strategy | Trigger | Description |
|----------|---------|-------------|
| EMI Restructuring | High EMI/income ratio | Extend tenure, reduce monthly payment |
| Payment Holiday | Sudden income disruption | Temporary payment pause |
| Financial Counselling | Behavioral stress signals | Budget planning assistance |
| Balance Transfer | High outstanding balance | Consolidate at lower rate |
| Emergency Credit Line | Low liquidity buffer | Short-term bridge financing |

### 10.3 Channel Selection Logic
- **App Notification:** Low severity, digitally active customer
- **SMS:** Medium severity, quick response needed
- **Phone Call:** High severity, needs personal conversation
- **RM Visit:** Critical severity, requires in-person intervention

---

## 11. Model Monitoring (PSI)

### Population Stability Index (PSI)
Detects when the model's input distribution shifts over time:

```
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)

PSI < 0.10 → No significant shift
PSI 0.10–0.20 → Warning (monitor closely)
PSI > 0.20 → Retrain recommended
```

- Computed per feature and for the overall PD score
- Prevents silent model degradation

---

## 12. Dashboard (4 Tabs)

### Tab 1: Executive Overview
- 4 KPI cards: Portfolio Risk Index, Total EL, High Priority Count, Intervention Rate
- Expected Loss trend chart (weekly)
- Employer loss bar chart (color-coded by severity)
- Risk tier donut chart (CRITICAL / HIGH / MEDIUM / STABLE)
- Intervention ROI analysis (savings vs cost at various intervention levels)

### Tab 2: Customer Intelligence
- Filterable priority ranking table (filter by risk tier, employer, top N)
- Customer drill-down panel:
  - 6 metric cards (PD, EL, Loss Velocity, Recovery, Intervention, Priority)
  - SHAP waterfall chart (risk factors)
  - Customer profile table
  - Recovery timeline (P10/P50/P90)
  - AI-generated intervention recommendation

### Tab 3: Explainability
- Global SHAP feature importance (top 15 features)
- Per-customer local SHAP explanation
- Human-readable narrative summary

### Tab 4: Risk Monitor
- 4 weekly trend charts (savings, salary delay, failed autodebits, lending app usage)
- Expected Loss distribution histogram
- P95 vs Expected Loss tail risk scatter plot
- Portfolio summary KPIs (Median EL, P95, P99)
- Active alerts with employer cluster breakdown

---

## 13. Production Scalability Path

| Component | Prototype | Production |
|-----------|-----------|------------|
| Data Ingestion | CSV batch | Apache Kafka (real-time CDC from core banking) |
| Feature Store | In-memory pandas | Feast / Tecton (versioned, low-latency) |
| Orchestration | CLI (main.py) | Apache Airflow DAGs (scheduled pipelines) |
| Model Serving | joblib files | MLflow Registry + BentoML (containerized APIs) |
| Model Monitoring | PSI in-memory | Evidently AI / WhyLabs (automated drift alerts) |
| Dashboard | Streamlit | Amazon QuickSight / Grafana / React |
| Alerting | JSON files | ServiceNow / Jira / CRM integration |
| Storage | Local filesystem | S3/GCS + PostgreSQL/Snowflake |

---

## 14. Testing Strategy

### Unit Tests (test_features.py)
- Validates no NaN/Inf values after feature engineering
- Checks feature ranges are within expected bounds
- Verifies all 11 engineered columns are present

### Loss Engine Tests (test_loss_engine.py)
- EL = PD × EAD × LGD calculation correctness
- Loss velocity sign and direction
- Priority score monotonicity (higher EL → higher priority)
- Monte Carlo simulation: P50 ≤ P95 ≤ P99

### Integration Tests (test_scoring.py)
- End-to-end pipeline on small data (100 customers × 6 weeks)
- Train → Score → Validate output schema
- Alert generation integration

### Commands
```bash
pytest tests/ -v                    # Run all tests
pytest tests/test_features.py -v    # Run feature tests only
pytest tests/test_loss_engine.py -v # Run loss engine tests only
```

---

## 15. How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Full Pipeline (recommended)
```bash
python main.py --run-all
```
This runs all 6 stages sequentially:
1. Generate synthetic data (5,000 customers × 12 weeks)
2. Train all 3 models (PD, Recovery, Intervention)
3. Score and rank all customers
4. Generate SHAP explanations (top 50)
5. Generate alerts
6. Generate intervention recommendations

### Launch Dashboard
```bash
python main.py --dashboard
```
Opens at http://localhost:8501

### Individual Stages
```bash
python main.py --generate-data
python main.py --train
python main.py --score
python main.py --explain
python main.py --alerts
python main.py --recommend
```

---

## 16. Key Results Summary (for Presentations)

| What We Built | Detail |
|---------------|--------|
| A pre-delinquency early warning system | Detects financial stress 2–4 weeks before default |
| 3 ML models | PD (XGBoost), Recovery (Quantile), Intervention (Logistic) |
| 11 engineered behavioral features | Stress velocity, employer clustering, liquidity analysis |
| Loss-optimized priority scoring | EL × Velocity × Stress × Intervention Effectiveness |
| Monte Carlo tail risk simulation | 1,000 iterations → P50/P95/P99 portfolio loss |
| Explainable AI | Per-customer SHAP + human-readable narratives |
| Smart alerting | Dynamic thresholds, cooldown dedup, employer clusters |
| Intervention recommendations | Strategy + Channel + Outreach message |
| 4-tab interactive dashboard | Streamlit with Plotly charts |
| Full test suite | Unit + integration tests |

| Model Performance | Value |
|-------------------|-------|
| KS Statistic | 0.6653 (Excellent) |
| Gini Coefficient | 0.6862 (Very Good) |
| Top Decile Lift | 5.12x |
| Brier Score | 0.0705 (Well Calibrated) |
| Customers Scored | 5,000 |
| Total EL Identified | Rs 20.9 Crore |

---

## 17. Limitations & Future Work

### Current Limitations
1. Uses synthetic data — real-world performance will differ
2. Intervention effectiveness uses heuristic labels (no real A/B test)
3. LGD parameters are assumed, not calibrated to actual portfolio
4. Agentic layer uses templates, not an actual LLM
5. Batch scoring only (no real-time streaming)

### Future Improvements
1. Real data integration with core banking systems
2. Online learning — incremental model updates with new data
3. Basel III/IV regulatory alignment for PD/LGD/EAD
4. A/B testing framework for measuring actual intervention ROI
5. LLM integration (e.g., Llama 3) for richer explanations
6. Real-time streaming pipeline (Kafka + Spark Streaming)
7. Feature store (Feast) for production feature serving
8. Model registry (MLflow) for version management and automated retraining

---

## 18. Presentation Talking Points

### Elevator Pitch (30 seconds)
"We built a pre-delinquency intervention engine that detects financial stress 2–4 weeks before a customer defaults. It doesn't just predict who will default — it quantifies potential loss, measures how fast the situation is worsening, predicts recovery time, and recommends the best intervention strategy. Our PD model achieves a KS of 0.67 and 5x lift in the top decile."

### Key Technical Differentiators
1. **Stress Velocity** — Most systems use static scores. We measure acceleration.
2. **Loss-Optimized Ranking** — Priority isn't just about PD. It's about EL × Velocity × Intervention.
3. **Monte Carlo Tail Risk** — We don't just give expected loss. We give P95 and P99 worst-case scenarios.
4. **Employer Clustering** — We detect systemic risk at the employer level (e.g., mass layoffs).
5. **Explainable by Design** — Every prediction comes with SHAP feature attributions and human-readable narratives.

### Questions You Might Get (and Answers)

**Q: Why not use a simple logistic regression for PD?**
A: Complex non-linear feature interactions (e.g., low liquidity × high EMI × declining savings) are captured better by gradient boosting. We add interpretability through SHAP.

**Q: How do you handle class imbalance?**
A: Dual strategy — SMOTE oversampling on training set + cost-sensitive `scale_pos_weight` in XGBoost. This prevents the model from ignoring the minority default class.

**Q: What is Platt scaling and why do you need it?**
A: Raw XGBoost outputs are not well-calibrated probabilities. Platt scaling fits a logistic function on a held-out calibration set, converting raw scores into proper probabilities. This is critical because Expected Loss = PD × EAD × LGD — if PD isn't calibrated, EL is wrong.

**Q: What is KS statistic and why 0.67 is good?**
A: KS (Kolmogorov-Smirnov) measures the maximum separation between the cumulative distribution of defaults and non-defaults. 0.67 means our model separates the two populations very well. Industry benchmarks: <0.2 = poor, 0.2-0.4 = fair, 0.4-0.6 = good, >0.6 = excellent.

**Q: What is Monte Carlo simulation used for here?**
A: We sample PD from a Beta distribution and LGD from another Beta distribution, then compute EL for each sample. Over 1,000 iterations, this gives us a full loss distribution. P95 means "in 95% of scenarios, portfolio loss won't exceed this amount" — this is the Value at Risk concept.

**Q: How is this different from a credit score?**
A: Credit scores are static. Our system is dynamic — it tracks weekly behavioral changes, measures velocity of deterioration, predicts recovery time, and recommends specific interventions. It's an end-to-end decision system, not just a number.
