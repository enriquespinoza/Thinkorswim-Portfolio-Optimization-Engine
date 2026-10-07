from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.metrics import (
    summarize_return_series,
)


def analyze_custom_regimes(
    period_results: pd.DataFrame,
    regimes: dict[
        str,
        tuple[str, str],
    ],
) -> pd.DataFrame:
    """
    Analyze model performance across user-defined date regimes.

    Parameters
    ----------
    period_results:
        Walk-forward period results.

    regimes:
        Dictionary such as:

        {
            "2018-2019": (
                "2018-01-01",
                "2019-12-31",
            )
        }

    Returns
    -------
    pd.DataFrame
        Multi-model regime statistics.
    """
    required = {
        "model",
        "period_end",
        "net_return",
    }

    missing = (
        required
        - set(
            period_results.columns
        )
    )

    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    results = (
        period_results.copy()
    )

    results[
        "period_end"
    ] = pd.to_datetime(
        results[
            "period_end"
        ]
    )

    records = []

    for (
        regime_name,
        (
            start_date,
            end_date,
        ),
    ) in regimes.items():

        start = pd.Timestamp(
            start_date
        )

        end = pd.Timestamp(
            end_date
        )

        regime_data = results[
            (
                results[
                    "period_end"
                ] >= start
            )
            & (
                results[
                    "period_end"
                ] <= end
            )
        ]

        for (
            model,
            group,
        ) in regime_data.groupby(
            "model"
        ):

            group = (
                group
                .sort_values(
                    "period_end"
                )
            )

            metrics = (
                summarize_return_series(
                    group[
                        "net_return"
                ],
                periods_per_year=12,
                risk_free_rate=0.0,
                )
            )

            records.append(
                {
                    "regime":
                        regime_name,

                    "model":
                        model,

                    **metrics,
                }
            )

    return pd.DataFrame(
        records
    )


def rank_models_by_regime(
    regime_results: pd.DataFrame,
    metric: str = "sharpe",
) -> pd.DataFrame:
    """
    Rank models within each regime.

    Rank 1 indicates the best result.
    """
    if metric not in (
        regime_results.columns
    ):
        raise ValueError(
            f"Unknown metric: {metric}"
        )

    ranked = (
        regime_results.copy()
    )

    ascending = (
        metric
        in {
            "annualized_volatility",
        }
    )

    ranked[
        "rank"
    ] = (
        ranked
        .groupby(
            "regime"
        )[metric]
        .rank(
            method="min",
            ascending=ascending,
        )
    )

    return ranked.sort_values(
        [
            "regime",
            "rank",
        ]
    )
