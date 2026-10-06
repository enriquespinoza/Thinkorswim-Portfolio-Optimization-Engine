from __future__ import annotations

import pandas as pd

from config.settings import RAW_DATA_DIR

from src.backtest.metrics import (
    summarize_backtest,
)

from src.backtest.portfolio_blend import (
    build_blended_weight_history,
    simulate_weight_history,
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
        200,
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

    period_map = (
        backtest.period_results[
            [
                "rebalance_date",
                "period_end",
            ]
        ]
        .drop_duplicates()
    )

    blend_results = []

    # Fixed diagnostic allocations only.
    # We are deliberately NOT optimizing alpha.
    for alpha in [
        0.25,
        0.50,
        0.75,
    ]:
        name = (
            f"EW_{1-alpha:.0%}_MS_{alpha:.0%}"
        )

        blend_weights = (
            build_blended_weight_history(
                weights=
                    backtest.weights,

                model_a=
                    "equal_weight",

                model_b=
                    "maximum_sharpe",

                model_b_weight=
                    alpha,

                strategy_name=
                    name,
            )
        )

        simulated = (
            simulate_weight_history(
                returns=returns,

                target_weights=
                    blend_weights,

                period_map=
                    period_map,

                transaction_cost_bps=
                    5.0,
            )
        )

        blend_results.append(
            simulated
        )

    blends = pd.concat(
        blend_results,
        ignore_index=True,
    )

    base_models = (
        backtest.period_results[
            backtest.period_results[
                "model"
            ].isin(
                [
                    "equal_weight",
                    "maximum_sharpe",
                ]
            )
        ]
    )

    comparison = pd.concat(
        [
            base_models,
            blends,
        ],
        ignore_index=True,
    )

    summary = (
        summarize_backtest(
            comparison
        )
    )

    display = (
        summary.copy()
    )

    for column in [
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "average_turnover",
    ]:
        display[
            column
        ] *= 100

    print()
    print("=" * 105)
    print(
        "EQUAL WEIGHT / MAXIMUM SHARPE BLEND"
    )
    print("=" * 105)

    print()
    print(
        "Fixed blends only - no blend weight "
        "has been optimized."
    )

    print()
    print(
        display.round(
            {
                "total_return": 2,
                "annualized_return": 2,
                "annualized_volatility": 2,
                "sharpe": 3,
                "sortino": 3,
                "max_drawdown": 2,
                "average_turnover": 2,
            }
        )
    )


if __name__ == "__main__":
    main()
