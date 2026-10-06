from __future__ import annotations

import pandas as pd

from config.settings import RAW_DATA_DIR

from src.backtest.regime_analysis import (
    analyze_custom_regimes,
    rank_models_by_regime,
)

from src.backtest.walk_forward import (
    run_walk_forward,
)

from src.portfolio.returns import (
    build_return_matrix,
)


def main() -> None:
    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        220,
    )

    prices = pd.read_csv(
        RAW_DATA_DIR
        / "portfolio_universe_daily.csv",
        index_col=0,
        parse_dates=True,
    )

    returns = (
        build_return_matrix(
            prices
        )
        .loc[
            :"2026-09-30"
        ]
    )

    backtest = (
        run_walk_forward(
            returns=returns,
            min_train_observations=756,
            max_weight=0.40,
            risk_free_rate=0.0,
            ewma_span=126,
            transaction_cost_bps=5.0,
        )
    )

    regimes = {
        "2018-2019 Pre-COVID": (
            "2018-01-01",
            "2019-12-31",
        ),

        "2020 Shock + Recovery": (
            "2020-01-01",
            "2020-12-31",
        ),

        "2021 Risk-On": (
            "2021-01-01",
            "2021-12-31",
        ),

        "2022 Rate Shock": (
            "2022-01-01",
            "2022-12-31",
        ),

        "2023-2024 Recovery": (
            "2023-01-01",
            "2024-12-31",
        ),

        "2025-2026": (
            "2025-01-01",
            "2026-09-30",
        ),
    }

    analysis = (
        analyze_custom_regimes(
            backtest.period_results,
            regimes,
        )
    )

    ranked = (
        rank_models_by_regime(
            analysis,
            metric="sharpe",
        )
    )

    display = (
        ranked.copy()
    )

    percentage_columns = [
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
    ]

    for column in (
        percentage_columns
    ):
        display[
            column
        ] *= 100

    print()
    print("=" * 115)
    print(
        "PORTFOLIO MODEL REGIME ANALYSIS"
    )
    print("=" * 115)

    for regime in regimes:
        print()
        print(regime.upper())
        print("-" * 115)

        regime_table = (
            display[
                display[
                    "regime"
                ]
                == regime
            ]
            .sort_values(
                "rank"
            )
            [
                [
                    "rank",
                    "model",
                    "periods",
                    "total_return",
                    "annualized_return",
                    "annualized_volatility",
                    "sharpe",
                    "sortino",
                    "max_drawdown",
                ]
            ]
        )

        print(
            regime_table
            .round(
                {
                    "total_return": 2,
                    "annualized_return": 2,
                    "annualized_volatility": 2,
                    "sharpe": 3,
                    "sortino": 3,
                    "max_drawdown": 2,
                }
            )
            .to_string(
                index=False
            )
        )

    print()
    print("=" * 115)
    print(
        "REGIME WINNERS BY SHARPE"
    )
    print("=" * 115)

    winners = (
        ranked[
            ranked[
                "rank"
            ] == 1
        ]
        [
            [
                "regime",
                "model",
                "sharpe",
                "annualized_return",
                "max_drawdown",
            ]
        ]
    )

    winners = (
        winners.copy()
    )

    winners[
        "annualized_return"
    ] *= 100

    winners[
        "max_drawdown"
    ] *= 100

    print(
        winners.round(
            {
                "sharpe": 3,
                "annualized_return": 2,
                "max_drawdown": 2,
            }
        )
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 115)
    print(
        "MODEL AVERAGE REGIME RANK"
    )
    print("=" * 115)

    average_rank = (
        ranked
        .groupby(
            "model"
        )[
            "rank"
        ]
        .mean()
        .sort_values()
    )

    print(
        average_rank.round(
            2
        )
    )


if __name__ == "__main__":
    main()
