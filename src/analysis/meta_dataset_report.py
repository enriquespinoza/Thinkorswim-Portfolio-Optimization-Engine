from __future__ import annotations

import pandas as pd

from config.settings import RAW_DATA_DIR

from src.backtest.walk_forward import (
    run_walk_forward,
)

from src.features.meta_dataset import (
    build_meta_learning_dataset,
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

    dataset = (
        build_meta_learning_dataset(
            returns=returns,
            period_results=
                backtest.period_results,
        )
    )

    print()
    print("=" * 100)
    print(
        "META-LEARNING DATASET"
    )
    print("=" * 100)

    print()
    print(
        "Rows:",
        len(dataset),
    )

    print(
        "Columns:",
        len(dataset.columns),
    )

    print(
        "First observation:",
        dataset.index.min().date(),
    )

    print(
        "Last observation:",
        dataset.index.max().date(),
    )

    print()
    print(
        "BEST MODEL FREQUENCY"
    )
    print("-" * 100)

    counts = (
        dataset[
            "best_model"
        ]
        .value_counts()
    )

    frequencies = (
        dataset[
            "best_model"
        ]
        .value_counts(
            normalize=True
        )
        * 100
    )

    class_table = pd.DataFrame(
        {
            "count":
                counts,

            "frequency_pct":
                frequencies,
        }
    )

    print(
        class_table.round(2)
    )

    print()
    print(
        "WINNER MARGIN"
    )
    print("-" * 100)

    print(
        (
            dataset[
                "winner_margin"
            ]
            * 100
        )
        .describe()
        .round(3)
    )

    print()
    print(
        "CROSS-MODEL RETURN DISPERSION"
    )
    print("-" * 100)

    print(
        (
            dataset[
                "model_return_dispersion"
            ]
            * 100
        )
        .describe()
        .round(3)
    )

    print()
    print(
        "AVERAGE FORWARD RETURN BY MODEL"
    )
    print("-" * 100)

    forward_columns = [
        column
        for column
        in dataset.columns
        if column.endswith(
            "_forward_return"
        )
    ]

    average_returns = (
        dataset[
            forward_columns
        ]
        .mean()
        .sort_values(
            ascending=False
        )
        * 100
    )

    print(
        average_returns.round(
            3
        )
    )

    print()
    print(
        "LATEST LABELED OBSERVATION"
    )
    print("-" * 100)

    latest = (
        dataset.iloc[
            -1
        ]
    )

    print(
        "Rebalance date:",
        dataset.index[
            -1
        ].date(),
    )

    print(
        "Best next-period model:",
        latest[
            "best_model"
        ],
    )

    print(
        "Winner margin:",
        f"{latest['winner_margin']:.2%}",
    )


if __name__ == "__main__":
    main()
