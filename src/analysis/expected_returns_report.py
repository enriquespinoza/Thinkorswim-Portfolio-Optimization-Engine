from __future__ import annotations

import pandas as pd

from config.settings import RAW_DATA_DIR

from src.forecasts.expected_returns import (
    expected_return_table,
)

from src.portfolio.returns import (
    build_return_matrix,
)


def main() -> None:
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
    )

    estimates = (
        expected_return_table(
            returns,
            ewma_span=126,
        )
    )

    display = (
        estimates * 100
    )

    print()
    print("=" * 80)
    print(
        "EXPECTED RETURN BASELINE"
    )
    print("=" * 80)

    print()
    print(
        "Sample:",
        returns.index.min().date(),
        "through",
        returns.index.max().date(),
    )

    print(
        "Observations:",
        len(returns),
    )

    print()
    print(
        "ANNUALIZED EXPECTED RETURN ESTIMATES (%)"
    )
    print("-" * 80)

    print(
        display.round(2)
    )

    print()
    print(
        "BLENDED EXPECTED RETURN RANKING"
    )
    print("-" * 80)

    ranking = (
        display[
            "blended"
        ]
        .sort_values(
            ascending=False
        )
    )

    print(
        ranking.round(2)
    )

    print()
    print(
        "ESTIMATOR DISAGREEMENT"
    )
    print("-" * 80)

    print(
        display[
            "estimator_dispersion"
        ]
        .sort_values(
            ascending=False
        )
        .round(2)
    )


if __name__ == "__main__":
    main()
