from config.settings import (
    PROJECT_ROOT,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FEATURE_DATA_DIR,
    TRADING_DAYS_PER_YEAR,
)


def test_project_root_exists():
    assert PROJECT_ROOT.exists()


def test_data_directories_exist():
    assert RAW_DATA_DIR.exists()
    assert PROCESSED_DATA_DIR.exists()
    assert FEATURE_DATA_DIR.exists()


def test_trading_days_constant():
    assert TRADING_DAYS_PER_YEAR == 252
