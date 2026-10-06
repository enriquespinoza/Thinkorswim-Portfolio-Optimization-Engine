from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURE_DATA_DIR = DATA_DIR / "features"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
GENERATED_REPORTS_DIR = REPORTS_DIR / "generated"

TRADING_DAYS_PER_YEAR = 252

DEFAULT_BENCHMARK = "SPY"

DEFAULT_UNIVERSE = [
    "SPY",
    "QQQ",
    "TLT",
    "GLD",
    "SCHD",
]
