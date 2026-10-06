import numpy as np
import pandas as pd

from src.backtest.metrics import (
    calculate_max_drawdown,
    summarize_backtest,
)


def test_max_drawdown():
    returns = pd.Series(
        [
            0.10,
            -0.20,
            0.05,
        ]
    )

    result = calculate_max_drawdown(
        returns
    )

    assert result < 0


def test_no_drawdown_for_positive_returns():
    returns = pd.Series(
        [
            0.01,
            0.02,
            0.03,
        ]
    )

    result = calculate_max_drawdown(
        returns
    )

    assert np.isclose(
        result,
        0.0,
    )


def test_summarize_backtest():
    results = pd.DataFrame(
        {
            "model": [
                "model_a",
                "model_a",
                "model_a",
                "model_b",
                "model_b",
                "model_b",
            ],
            "period_end": pd.to_datetime(
                [
                    "2026-01-31",
                    "2026-02-28",
                    "2026-03-31",
                    "2026-01-31",
                    "2026-02-28",
                    "2026-03-31",
                ]
            ),
            "net_return": [
                0.02,
                -0.01,
                0.03,
                0.01,
                0.01,
                0.01,
            ],
            "turnover": [
                1.00,
                0.10,
                0.15,
                1.00,
                0.05,
                0.05,
            ],
        }
    )

    summary = summarize_backtest(
        results
    )

    assert "annualized_return" in summary.columns
    assert "annualized_volatility" in summary.columns
    assert "sharpe" in summary.columns
    assert "max_drawdown" in summary.columns
    assert "average_turnover" in summary.columns

    assert "model_a" in summary.index
    assert "model_b" in summary.index
