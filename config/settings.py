"""
Forex AI Models - Configuration Settings
"""
from pathlib import Path

# Paths (relative to repo root)
REPO_ROOT = Path(__file__).parent.parent
DATA_RAW_AV = REPO_ROOT / "data" / "raw" / "alphavantage"
DATA_RAW_YF = REPO_ROOT / "data" / "raw" / "yfinance"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DATA_MODELS = REPO_ROOT / "data" / "models"
OUTPUTS_DIR = REPO_ROOT / "outputs"
LOGS_DIR = REPO_ROOT / "logs"
DATABASE_DIR = REPO_ROOT / "database"

# Trading Configuration
FX_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]

TIMEFRAMES = {
    "1m_7d": ("1min", "7d"),
    "5m_1mo": ("5min", "1mo"),
    "15m_60d": ("15min", "60d"),
    "1h_2y": ("1h", "2y"),
    "1d_5y": ("1d", "5y")
}

TIMEFRAME_WEIGHTS = {
    "1m_7d": 0.5,
    "5m_1mo": 1.0,
    "15m_60d": 1.5,
    "1h_2y": 2.0,
    "1d_5y": 3.0
}

# Data Quality Thresholds
MIN_QUALITY_SCORE = 40.0
MIN_ROWS_REQUIRED = 10
MAX_MISSING_RATIO = 0.95

# Model Parameters
SGD_MAX_ITER = 1000
SGD_TOLERANCE = 1e-3
RF_N_ESTIMATORS = 50
MIN_TRAINING_SAMPLES = 50

# Risk Management
ATR_MULTIPLIER_LOW_VOLATILITY = 2.0
ATR_MULTIPLIER_NORMAL = 1.0
VOLATILITY_THRESHOLD = 0.05

# Database
DB_NAME = "memory_v85.db"
MIN_TRADE_AGE_HOURS = 1
MAX_RETRIES = 3
