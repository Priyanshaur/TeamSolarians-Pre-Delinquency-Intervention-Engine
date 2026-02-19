"""
Central configuration for Financial Triage AI system.
All tunable parameters, paths, and thresholds in one place.
"""
import os
from pathlib import Path

# ── Project Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_ARTIFACTS_DIR = BASE_DIR / "models" / "artifacts"
EVAL_ARTIFACTS_DIR = BASE_DIR / "models" / "evaluation_reports"

# Ensure directories exist
for d in [DATA_DIR, MODEL_ARTIFACTS_DIR, EVAL_ARTIFACTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Data Generation ────────────────────────────────────────────
NUM_CUSTOMERS = 5000
NUM_WEEKS = 12
DEFAULT_RATE = 0.15
EMPLOYER_LIST = [
    "TechCorp", "FinServe Ltd", "RetailMax", "AutoMotion",
    "GreenEnergy", "HealthPlus", "EduWorld", "BuildRight",
    "MediaFlow", "LogiTrans", "FoodChain", "CloudNine",
    "SteelWorks", "PharmaCo", "AgroVest"
]
DISTRESSED_EMPLOYER_FRACTION = 0.20  # 20% of employers are "distressed"
MACRO_SHOCK_PROBABILITY = 0.10       # 10% of weeks have macro shocks

# ── Feature Engineering ────────────────────────────────────────
SALARY_DELAY_WINDOW = 4         # weeks lookback for salary delay z-score
SAVINGS_EWMA_SPAN = 4           # exponential weighted MA span
SPIKE_LOOKBACK = 4              # minimum weeks for z-score calculation
EMI_RATIO_DANGER_ZONE = 0.70    # EMI-to-income ratio danger threshold
EMI_RATIO_CAP = 1.50            # cap extreme EMI ratios

STRESS_VELOCITY_WEIGHTS = {
    "salary_delay_proxy": 0.20,
    "savings_decline_pct": 0.25,
    "emi_to_income_ratio": 0.15,
    "lending_app_spike": 0.15,
    "atm_withdrawal_spike": 0.10,
    "failed_autodebit_streak": 0.15,
}

# ── Model Training ─────────────────────────────────────────────
TEST_SIZE = 0.20
CALIBRATION_SIZE = 0.15  # held-out set for Platt scaling
CV_FOLDS = 5
RANDOM_STATE = 42

PD_MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "use_label_encoder": False,
}

RECOVERY_MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
}

# ── Loss Engine ────────────────────────────────────────────────
LGD_ALPHA = 4.0          # Beta distribution parameter for LGD
LGD_BETA = 5.0           # Mean LGD ≈ alpha/(alpha+beta) ≈ 0.44
LGD_FIXED = 0.45         # Fallback fixed LGD
MONTE_CARLO_ITERATIONS = 1000

# Stress scenario multipliers
STRESS_SCENARIOS = {
    "baseline": {"pd_mult": 1.0, "lgd_mult": 1.0},
    "mild_recession": {"pd_mult": 1.3, "lgd_mult": 1.1},
    "severe_recession": {"pd_mult": 1.5, "lgd_mult": 1.2},
    "sector_shock": {"pd_mult": 1.8, "lgd_mult": 1.3},
}

# ── Alerting ───────────────────────────────────────────────────
ALERT_THRESHOLD_PERCENTILE = 75    # dynamic threshold at P75
ALERT_COOLDOWN_DAYS = 7
EMPLOYER_CLUSTER_MIN_SIZE = 3      # min employees for systemic alert
SEVERITY_THRESHOLDS = {
    "CRITICAL": 50000,    # P99 loss > $50K
    "HIGH": 20000,
    "MEDIUM": 5000,
}

# ── Monitoring ─────────────────────────────────────────────────
PSI_WARNING_THRESHOLD = 0.10
PSI_RETRAIN_THRESHOLD = 0.20

# ── Dashboard ──────────────────────────────────────────────────
TOP_N_CUSTOMERS = 50
DASHBOARD_REFRESH_SECONDS = 300
