from __future__ import annotations

import pandas as pd

from config.settings import RAW_DATA_DIR

from src.backtest.walk_forward import (
    run_walk_forward,
)

from src.features.regime_features import (
    build_regime_feature_matrix,
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

    rebalance_dates = (
        backtest.period_results[
            "rebalance_date"
        ]
        .drop_duplicates()
        .sort_values()
    )

    features = (
        build_regime_feature_matrix(
            returns,
            rebalance_dates,
        )
    )

    print()
    print("=" * 100)
    print(
        "REGIME FEATURE DATASET"
    )
    print("=" * 100)

    print()
    print(
        "Rows:",
        len(features),
    )

    print(
        "Features:",
        len(features.columns),
    )

    print(
        "First date:",
        features.index.min().date(),
    )

    print(
        "Last date:",
        features.index.max().date(),
    )

    print()
    print(
        "LATEST MARKET STATE"
    )
    print("-" * 100)

    latest = (
        features
        .iloc[-1]
        .sort_index()
    )

    print(
        latest.round(4)
    )

    print()
    print(
        "FEATURE DISPERSION"
    )
    print("-" * 100)

    summary = pd.DataFrame(
        {
            "mean":
                features.mean(),

            "std":
                features.std(),

            "min":
                features.min(),

            "max":
                features.max(),
        }
    )

    print(
        summary.round(4)
    )


if __name__ == "__main__":
    main()
