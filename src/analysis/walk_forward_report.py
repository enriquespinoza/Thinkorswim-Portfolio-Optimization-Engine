from __future__ import annotations

import pandas as pd

from config.settings import RAW_DATA_DIR

from src.backtest.metrics import (
    summarize_backtest,
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

    returns = build_return_matrix(
        prices
    )

    backtest_returns = returns.loc[
        :"2026-09-30"
    ]

    result = run_walk_forward(
        returns=backtest_returns,

        # About three years of history before
        # making the first portfolio decision.
        min_train_observations=756,

        max_weight=0.40,

        # Keep at zero for our first controlled test.
        risk_free_rate=0.0,

        ewma_span=126,

        # Five basis points per unit of turnover.
        transaction_cost_bps=5.0,
    )

    summary = summarize_backtest(
        result.period_results
    )

    display = summary.copy()

    percentage_columns = [
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "average_turnover",
    ]

    for column in percentage_columns:
        display[
            column
        ] *= 100

    print()
    print("=" * 100)
    print(
        "PORTFOLIO OPTIMIZATION ENGINE"
    )
    print(
        "MONTHLY WALK-FORWARD BACKTEST"
    )
    print("=" * 100)

    print()
    print("DATA")
    print("-" * 100)

    print(
        "Return history:",
        returns.index.min().date(),
        "through",
        returns.index.max().date(),
    )

    print(
        "Assets:",
        ", ".join(
            returns.columns
        ),
    )

    print(
        "Minimum training observations:",
        756,
    )

    print(
        "Rebalance frequency:",
        "Monthly",
    )

    print(
        "Transaction cost:",
        "5 bps per unit of turnover",
    )

    print()
    print("OUT-OF-SAMPLE MODEL PERFORMANCE")
    print("-" * 100)

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

    print()
    print("LATEST MODEL WEIGHTS")
    print("-" * 100)

    latest_date = (
        result.weights[
            "rebalance_date"
        ].max()
    )

    latest_weights = (
        result.weights[
            result.weights[
                "rebalance_date"
            ]
            == latest_date
        ]
        .pivot(
            index="asset",
            columns="model",
            values="weight",
        )
        * 100
    )

    print(
        latest_weights.round(2)
    )

    print()
    print(
        "Latest rebalance:",
        latest_date.date(),
    )

    print()
    print("AVERAGE MONTHLY TURNOVER")
    print("-" * 100)

    turnover = (
        result.period_results
        .groupby("model")[
            "turnover"
        ]
        .mean()
        .sort_values()
        * 100
    )

    print(
        turnover.round(2)
    )

    print()
    print("BEST MODEL BY SHARPE")
    print("-" * 100)

    best_model = (
        summary[
            "sharpe"
        ].idxmax()
    )

    print(
        best_model
    )

    print()
    print("BACKTEST PERIOD")
    print("-" * 100)

    print(
        result.period_results[
            "rebalance_date"
        ].min().date(),
        "through",
        result.period_results[
            "period_end"
        ].max().date(),
    )

    print(
        "Out-of-sample monthly periods:",
        result.period_results[
            "period_end"
        ].nunique(),
    )


if __name__ == "__main__":
    main()
