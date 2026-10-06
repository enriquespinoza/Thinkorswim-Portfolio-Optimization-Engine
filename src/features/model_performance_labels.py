from __future__ import annotations

import numpy as np
import pandas as pd


def validate_period_results(
    period_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate walk-forward model results used to build
    forward-performance labels.
    """
    required = {
        "model",
        "rebalance_date",
        "period_end",
        "net_return",
    }

    missing = (
        required
        - set(period_results.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    if period_results.empty:
        raise ValueError(
            "period_results cannot be empty."
        )

    cleaned = period_results.copy()

    cleaned[
        "rebalance_date"
    ] = pd.to_datetime(
        cleaned[
            "rebalance_date"
        ]
    )

    cleaned[
        "period_end"
    ] = pd.to_datetime(
        cleaned[
            "period_end"
        ]
    )

    cleaned[
        "net_return"
    ] = pd.to_numeric(
        cleaned[
            "net_return"
        ],
        errors="coerce",
    )

    if cleaned[
        "net_return"
    ].isna().any():
        raise ValueError(
            "net_return contains missing or "
            "non-numeric values."
        )

    if not np.isfinite(
        cleaned[
            "net_return"
        ].to_numpy()
    ).all():
        raise ValueError(
            "net_return contains non-finite values."
        )

    return cleaned


def build_forward_return_matrix(
    period_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one row per rebalance date and one column per model.

    The values are each model's realized return during the
    NEXT holding period.

    Example
    -------
    rebalance_date      equal_weight   hrp   maximum_sharpe
    2020-01-31              0.02       ...       0.03

    Features observed on 2020-01-31 may therefore be paired
    with these forward outcomes.
    """
    results = validate_period_results(
        period_results
    )

    duplicate_mask = (
        results.duplicated(
            subset=[
                "rebalance_date",
                "model",
            ],
            keep=False,
        )
    )

    if duplicate_mask.any():
        raise ValueError(
            "Multiple observations exist for the same "
            "rebalance_date and model."
        )

    matrix = (
        results.pivot(
            index="rebalance_date",
            columns="model",
            values="net_return",
        )
        .sort_index()
    )

    matrix.columns.name = None

    if matrix.isna().any().any():
        raise ValueError(
            "Some rebalance dates are missing model returns."
        )

    return matrix


def build_model_excess_returns(
    forward_returns: pd.DataFrame,
    benchmark_model: str = "equal_weight",
) -> pd.DataFrame:
    """
    Calculate each model's forward excess return relative
    to a benchmark model.
    """
    if benchmark_model not in (
        forward_returns.columns
    ):
        raise ValueError(
            f"Benchmark model '{benchmark_model}' "
            "is not present."
        )

    benchmark = (
        forward_returns[
            benchmark_model
        ]
    )

    excess = (
        forward_returns
        .subtract(
            benchmark,
            axis=0,
        )
    )

    excess.columns = [
        f"{column}_excess_vs_{benchmark_model}"
        for column in excess.columns
    ]

    return excess


def build_model_ranks(
    forward_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank models by next-period realized return.

    Rank 1 = highest realized net return.
    """
    ranks = (
        forward_returns.rank(
            axis=1,
            method="min",
            ascending=False,
        )
    )

    ranks.columns = [
        f"{column}_rank"
        for column in ranks.columns
    ]

    return ranks


def calculate_winner_margin(
    forward_returns: pd.DataFrame,
) -> pd.Series:
    """
    Difference between the best and second-best model return.

    A small margin means the winner label is weak/ambiguous.
    A large margin means model selection mattered more.
    """
    sorted_values = np.sort(
        forward_returns.to_numpy(),
        axis=1,
    )

    margin = (
        sorted_values[:, -1]
        - sorted_values[:, -2]
    )

    return pd.Series(
        margin,
        index=forward_returns.index,
        name="winner_margin",
    )


def build_forward_model_labels(
    period_results: pd.DataFrame,
    benchmark_model: str = "equal_weight",
) -> pd.DataFrame:
    """
    Build forward model-performance labels.

    Includes:
    - realized next-period return for every model
    - excess return versus equal weight
    - model ranks
    - best model
    - best model return
    - worst model return
    - winner margin
    - cross-model return dispersion
    """
    forward_returns = (
        build_forward_return_matrix(
            period_results
        )
    )

    excess_returns = (
        build_model_excess_returns(
            forward_returns,
            benchmark_model=
                benchmark_model,
        )
    )

    ranks = build_model_ranks(
        forward_returns
    )

    labels = pd.DataFrame(
        index=forward_returns.index
    )

    for model in (
        forward_returns.columns
    ):
        labels[
            f"{model}_forward_return"
        ] = forward_returns[
            model
        ]

    labels = labels.join(
        excess_returns
    )

    labels = labels.join(
        ranks
    )

    labels[
        "best_model"
    ] = forward_returns.idxmax(
        axis=1
    )

    labels[
        "best_model_return"
    ] = forward_returns.max(
        axis=1
    )

    labels[
        "worst_model_return"
    ] = forward_returns.min(
        axis=1
    )

    labels[
        "winner_margin"
    ] = calculate_winner_margin(
        forward_returns
    )

    labels[
        "model_return_dispersion"
    ] = forward_returns.std(
        axis=1,
        ddof=0,
    )

    labels.index.name = (
        "rebalance_date"
    )

    return labels
