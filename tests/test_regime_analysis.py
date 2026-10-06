import pandas as pd

from src.backtest.regime_analysis import (
    summarize_return_series,
)

from src.backtest.regime_analysis import (
    analyze_custom_regimes,
    rank_models_by_regime,
    summarize_period,
)


def test_summarize_period():
    returns = pd.Series(
        [
            0.02,
            -0.01,
            0.03,
            0.01,
        ]
    )

    result = summarize_return_series(
        returns
    )

    assert result[
        "periods"
    ] == 4

    assert (
        result[
            "annualized_return"
        ]
        > 0
    )

    assert (
        result[
            "annualized_volatility"
        ]
        > 0
    )


def test_analyze_custom_regimes():
    results = pd.DataFrame(
        {
            "model": [
                "a",
                "a",
                "b",
                "b",
            ],

            "period_end":
                pd.to_datetime(
                    [
                        "2020-01-31",
                        "2020-02-29",
                        "2020-01-31",
                        "2020-02-29",
                    ]
                ),

            "net_return": [
                0.02,
                -0.01,
                0.01,
                0.01,
            ],
        }
    )

    regimes = {
        "test": (
            "2020-01-01",
            "2020-12-31",
        )
    }

    analysis = (
        analyze_custom_regimes(
            results,
            regimes,
        )
    )

    assert set(
        analysis[
            "model"
        ]
    ) == {
        "a",
        "b",
    }


def test_rank_models_by_regime():
    data = pd.DataFrame(
        {
            "regime": [
                "test",
                "test",
            ],
            "model": [
                "a",
                "b",
            ],
            "sharpe": [
                1.2,
                0.8,
            ],
        }
    )

    ranked = (
        rank_models_by_regime(
            data,
            metric="sharpe",
        )
    )

    winner = ranked.iloc[
        0
    ]

    assert (
        winner[
            "model"
        ]
        == "a"
    )

    assert (
        winner[
            "rank"
        ]
        == 1
    )
