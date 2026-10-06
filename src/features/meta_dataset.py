from __future__ import annotations

import pandas as pd

from src.features.model_performance_labels import (
    build_forward_model_labels,
)

from src.features.regime_features import (
    build_regime_feature_matrix,
)


def build_meta_learning_dataset(
    returns: pd.DataFrame,
    period_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine observable market-state features at time t with
    realized portfolio-model outcomes over t -> t+1.

    This preserves the time direction:

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
        features.join(
            labels,
            how="inner",
        )
        .sort_index()
    )

    if dataset.empty:
        raise ValueError(
            "Meta-learning dataset is empty."
        )

    return dataset
