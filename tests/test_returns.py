import numpy as np
import pandas as pd
import pytest

from src.portfolio.returns import (
    build_return_matrix,
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_cumulative_returns,
    calculate_daily_returns,
    calculate_log_returns,
    summarize_returns,
    validate_prices,
)


@pytest.fixture
def sample_prices():
    return pd.DataFrame(
        {
            "AAA": [100.0, 101.0, 103.02, 102.0],
            "BBB": [50.0, 50.5, 50.0, 51.0],
        },
        index=pd.to_datetime(
            [
                "2026-01-02",
                "2026-01-05",
                "2026-01-06",
                "2026-01-07",
            ]
        ),
    )


def test_validate_prices(sample_prices):
    cleaned = validate_prices(sample_prices)

    assert isinstance(cleaned, pd.DataFrame)
    assert list(cleaned.columns) == ["AAA", "BBB"]
    assert len(cleaned) == 4


def test_validate_prices_rejects_nonpositive_values():
    prices = pd.DataFrame(
        {"AAA": [100.0, 0.0]},
        index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
    )

    with pytest.raises(ValueError):
        validate_prices(prices)


def test_daily_returns(sample_prices):
    returns = calculate_daily_returns(sample_prices)

    assert len(returns) == 3

    expected_first_aaa = 101.0 / 100.0 - 1.0

    assert np.isclose(
        returns.iloc[0]["AAA"],
        expected_first_aaa,
    )


def test_log_returns(sample_prices):
    returns = calculate_log_returns(sample_prices)

    expected = np.log(101.0 / 100.0)

    assert np.isclose(
        returns.iloc[0]["AAA"],
        expected,
    )


def test_cumulative_returns():
    returns = pd.DataFrame(
        {
            "AAA": [0.10, -0.05],
        }
    )

    cumulative = calculate_cumulative_returns(returns)

    expected = (1.10 * 0.95) - 1.0

    assert np.isclose(
        cumulative.iloc[-1]["AAA"],
        expected,
    )


def test_annualized_return_constant_daily_gain():
    returns = pd.DataFrame(
        {
            "AAA": [0.01] * 252,
        }
    )

    annualized = calculate_annualized_return(returns)

    expected = (1.01 ** 252) - 1.0

    assert np.isclose(
        annualized["AAA"],
        expected,
    )


def test_annualized_volatility_constant_returns():
    returns = pd.DataFrame(
        {
            "AAA": [0.01] * 252,
        }
    )

    volatility = calculate_annualized_volatility(returns)

    assert np.isclose(
        volatility["AAA"],
        0.0,
    )


def test_build_return_matrix_removes_missing_rows():
    prices = pd.DataFrame(
        {
            "AAA": [100.0, 101.0, 102.0, 103.0],
            "BBB": [50.0, np.nan, 51.0, 52.0],
        },
        index=pd.date_range(
            "2026-01-01",
            periods=4,
            freq="D",
        ),
    )

    returns = build_return_matrix(prices)

    assert not returns.isna().any().any()


def test_summarize_returns(sample_prices):
    summary = summarize_returns(sample_prices)

    assert "total_return" in summary.columns
    assert "annualized_return" in summary.columns
    assert "annualized_volatility" in summary.columns

    assert "AAA" in summary.index
    assert "BBB" in summary.index
