from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.metrics import (
    summarize_return_series,
)


def summarize_period(
    returns: pd.Series,
    periods_per_year: int = 12,
) -> dict[str, float]:
    """
    Calculate performance statistics for one return series.
    """
    if returns.empty:
        return {
            "periods": 0,
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "max_drawdown": np.nan,
        }

    total_growth = (
        1.0 + returns
    ).prod()

    periods = len(
        returns
    )

    annualized_return = (
        total_growth
        ** (
            periods_per_year
            / periods
        )
        - 1.0
    )

    if periods > 1:
        annualized_volatility = (
            returns.std(
                ddof=1
            )
            * np.sqrt(
                periods_per_year
            )
        )
    else:
        annualized_volatility = np.nan

    if (
        annualized_volatility > 0
        and np.isfinite(
            annualized_volatility
        )
    ):
        sharpe = (
            annualized_return
            / annualized_volatility
        )
    else:
        sharpe = np.nan

    downside = (
        returns[
            returns < 0
        ]
    )

    if len(
        downside
    ) > 1:
        downside_deviation = (
            downside.std(
                ddof=1
            )
            * np.sqrt(
                periods_per_year
            )
        )

        if (
            downside_deviation > 0
        ):
            sortino = (
                annualized_return
                / downside_deviation
            )
        else:
            sortino = np.nan
    else:
        sortino = np.nan

    return {
        "periods":
            periods,

        "total_return":
            total_growth - 1.0,

        "annualized_return":
            annualized_return,

        "annualized_volatility":
            annualized_volatility,

        "sharpe":
            sharpe,

        "sortino":
            sortino,

        "max_drawdown":
            calculate_max_drawdown(
                returns
            ),
    }


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
                ]
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
