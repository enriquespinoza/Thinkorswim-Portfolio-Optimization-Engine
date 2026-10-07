from __future__ import annotations

import pandas as pd

from src.features.model_performance_labels import (
    build_forward_model_labels,
)

from src.features.regime_features import (
    build_regime_feature_matrix,
)


def join_macro_features_to_meta_dataset(
    meta_dataset: pd.DataFrame,
    macro_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join availability-safe macro features onto the existing
    market-feature / forward-label dataset.

    The join is required to preserve the original number of
    meta-learning observations.
    """
    if not isinstance(
        meta_dataset.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "meta_dataset must use a DatetimeIndex."
        )

    if not isinstance(
        macro_features.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "macro_features must use a DatetimeIndex."
        )

    overlapping_columns = (
        set(
            meta_dataset.columns
        )
        & set(
            macro_features.columns
        )
    )

    if overlapping_columns:
        raise ValueError(
            "Macro features overlap existing columns: "
            f"{sorted(overlapping_columns)}"
        )

    original_rows = len(
        meta_dataset
    )

    joined = (
        meta_dataset
        .join(
            macro_features,
            how="inner",
        )
        .sort_index()
    )

    if len(joined) != original_rows:
        raise ValueError(
            "Joining macro features changed the number "
            "of meta-learning observations. "
            f"Before: {original_rows}, "
            f"after: {len(joined)}."
        )

    macro_columns = list(
        macro_features.columns
    )

    if joined[
        macro_columns
    ].isna().any().any():
        missing_counts = (
            joined[
                macro_columns
            ]
            .isna()
            .sum()
        )

        missing_counts = (
            missing_counts[
                missing_counts
                > 0
            ]
        )

        raise ValueError(
            "Joined macro features contain missing values: "
            f"{missing_counts.to_dict()}"
        )

    return joined


def build_meta_learning_dataset(
    returns: pd.DataFrame,
    period_results: pd.DataFrame,
    macro_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Combine observable market-state features at time t with
    realized portfolio-model outcomes over t -> t+1.

    Optional macro features may be joined using the same
    rebalance-date index.

    Time direction:

        X_t -> y_{t+1}
    """
    rebalance_dates = (
        period_results[
            "rebalance_date"
        ]
        .drop_duplicates()
        .sort_values()
    )

    features = (
        build_regime_feature_matrix(
            returns=returns,
            rebalance_dates=
                rebalance_dates,
        )
    )

    labels = (
        build_forward_model_labels(
            period_results
        )
    )

    dataset = (
        features
        .join(
            labels,
            how="inner",
        )
        .sort_index()
    )

    if dataset.empty:
        raise ValueError(
            "Meta-learning dataset is empty."
        )

    if macro_features is not None:
        dataset = (
            join_macro_features_to_meta_dataset(
                meta_dataset=
                    dataset,
                macro_features=
                    macro_features,
            )
        )

    return dataset
