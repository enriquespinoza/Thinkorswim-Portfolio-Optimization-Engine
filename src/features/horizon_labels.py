from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_PERIOD_COLUMNS = {
    "rebalance_date",
    "period_end",
    "model",
    "net_return",
}


def validate_period_results(
    period_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate monthly walk-forward period results.
    """
    if not isinstance(
        period_results,
        pd.DataFrame,
    ):
        raise TypeError(
            "period_results must be a pandas DataFrame."
        )

    missing = (
        REQUIRED_PERIOD_COLUMNS
        - set(
            period_results.columns
        )
    )

    if missing:
        raise ValueError(
            "period_results missing columns: "
            f"{sorted(missing)}"
        )

    data = (
        period_results.copy()
    )

    data[
        "rebalance_date"
    ] = pd.to_datetime(
        data[
            "rebalance_date"
        ]
    )

    data[
        "period_end"
    ] = pd.to_datetime(
        data[
            "period_end"
        ]
    )

    if data.empty:
        raise ValueError(
            "period_results cannot be empty."
        )

    return (
        data.sort_values(
            [
                "model",
                "rebalance_date",
            ]
        )
    )


def compound_returns(
    returns: pd.Series,
) -> float:
    """
    Compound a sequence of periodic returns.
    """
    values = (
        returns
        .astype(float)
        .to_numpy()
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "Returns contain non-finite values."
        )

    return float(
        np.prod(
            1.0
            + values
        )
        - 1.0
    )


def build_forward_horizon_returns(
    period_results: pd.DataFrame,
    horizon_periods: int = 3,
) -> pd.DataFrame:
    """
    Build compounded forward returns for every model.

    Example for horizon_periods=3:

        return_t
        + return_t+1
        + return_t+2

    are compounded into one forward three-period return.

    The output also records horizon_end so expanding-window
    models can purge observations whose future outcomes were
    not yet completely known.
    """
    if horizon_periods <= 0:
        raise ValueError(
            "horizon_periods must be positive."
        )

    data = (
        validate_period_results(
            period_results
        )
    )

    records = []

    for (
        model,
        group,
    ) in data.groupby(
        "model",
        sort=False,
    ):
        group = (
            group
            .sort_values(
                "rebalance_date"
            )
            .reset_index(
                drop=True
            )
        )

        for start_position in range(
            len(group)
            - horizon_periods
            + 1
        ):
            window = (
                group.iloc[
                    start_position:
                    start_position
                    + horizon_periods
                ]
            )

            records.append(
                {
                    "rebalance_date":
                        window[
                            "rebalance_date"
                        ].iloc[0],

                    "horizon_end":
                        window[
                            "period_end"
                        ].iloc[-1],

                    "model":
                        model,

                    "horizon_periods":
                        horizon_periods,

                    "forward_return":
                        compound_returns(
                            window[
                                "net_return"
                            ]
                        ),
                }
            )

    result = pd.DataFrame(
        records
    )

    if result.empty:
        raise ValueError(
            "No forward horizon returns generated."
        )

    return (
        result.sort_values(
            [
                "rebalance_date",
                "model",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def build_forward_horizon_labels(
    period_results: pd.DataFrame,
    horizon_periods: int = 3,
) -> pd.DataFrame:
    """
    Convert long model/horizon returns into one row per
    rebalance date.

    Produces:

    - <model>_forward_return
    - <model>_excess_vs_equal_weight
    - best_model
    - best_model_return
    - worst_model_return
    - winner_margin
    - model_return_dispersion
    - horizon_end
    """
    long_returns = (
        build_forward_horizon_returns(
            period_results=
                period_results,
            horizon_periods=
                horizon_periods,
        )
    )

    pivot = (
        long_returns
        .pivot(
            index=
                "rebalance_date",
            columns=
                "model",
            values=
                "forward_return",
        )
        .sort_index()
    )

    if (
        "equal_weight"
        not in pivot.columns
    ):
        raise ValueError(
            "equal_weight model is required."
        )

    horizon_end = (
        long_returns
        .groupby(
            "rebalance_date"
        )[
            "horizon_end"
        ]
        .max()
        .sort_index()
    )

    labels = pd.DataFrame(
        index=pivot.index
    )

    for model in (
        pivot.columns
    ):
        labels[
            f"{model}_forward_return"
        ] = pivot[
            model
        ]

        labels[
            f"{model}_excess_vs_equal_weight"
        ] = (
            pivot[
                model
            ]
            - pivot[
                "equal_weight"
            ]
        )

    labels[
        "best_model"
    ] = pivot.idxmax(
        axis=1
    )

    labels[
        "best_model_return"
    ] = pivot.max(
        axis=1
    )

    labels[
        "worst_model_return"
    ] = pivot.min(
        axis=1
    )

    sorted_returns = np.sort(
        pivot.to_numpy(),
        axis=1,
    )

    labels[
        "winner_margin"
    ] = (
        sorted_returns[
            :,
            -1
        ]
        - sorted_returns[
            :,
            -2
        ]
    )

    labels[
        "model_return_dispersion"
    ] = pivot.std(
        axis=1,
        ddof=0,
    )

    labels[
        "horizon_end"
    ] = horizon_end

    labels.index.name = (
        "rebalance_date"
    )

    return labels
