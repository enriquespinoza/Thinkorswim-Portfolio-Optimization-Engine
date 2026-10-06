from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConsensusResult:
    """
    Result from combining multiple portfolio models.
    """

    model_weights: pd.DataFrame
    consensus_weights: pd.Series
    allocation_std: pd.Series
    allocation_range: pd.Series
    disagreement_score: pd.Series
    overall_disagreement: float


def validate_model_weights(
    model_weights: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate a matrix of portfolio model weights.

    Rows:
        Assets

    Columns:
        Portfolio models
    """
    if not isinstance(
        model_weights,
        pd.DataFrame,
    ):
        raise TypeError(
            "model_weights must be a pandas DataFrame."
        )

    if model_weights.empty:
        raise ValueError(
            "model_weights cannot be empty."
        )

    if model_weights.shape[1] < 2:
        raise ValueError(
            "At least two portfolio models are required."
        )

    cleaned = model_weights.copy()

    cleaned = cleaned.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if cleaned.isna().any().any():
        raise ValueError(
            "model_weights contains missing or "
            "non-numeric values."
        )

    if not np.isfinite(
        cleaned.to_numpy()
    ).all():
        raise ValueError(
            "model_weights contains non-finite values."
        )

    if (
        cleaned < -1e-12
    ).any().any():
        raise ValueError(
            "model_weights contains negative weights."
        )

    for model in cleaned.columns:
        total = cleaned[
            model
        ].sum()

        if not np.isclose(
            total,
            1.0,
            atol=1e-8,
        ):
            raise ValueError(
                f"Model '{model}' weights must sum to 1. "
                f"Observed total: {total}"
            )

    return cleaned


def calculate_consensus_weights(
    model_weights: pd.DataFrame,
    model_importance: pd.Series | None = None,
) -> pd.Series:
    """
    Calculate consensus target weights.

    By default, each portfolio model receives equal influence.

    Optional model_importance allows unequal model weighting.
    """
    model_weights = validate_model_weights(
        model_weights
    )

    if model_importance is None:
        importance = pd.Series(
            1.0 / len(
                model_weights.columns
            ),
            index=model_weights.columns,
            dtype=float,
        )

    else:
        if not isinstance(
            model_importance,
            pd.Series,
        ):
            raise TypeError(
                "model_importance must be a pandas Series."
            )

        missing = (
            set(model_weights.columns)
            - set(model_importance.index)
        )

        extra = (
            set(model_importance.index)
            - set(model_weights.columns)
        )

        if missing or extra:
            raise ValueError(
                "model_importance index must exactly match "
                "model columns."
            )

        importance = (
            model_importance
            .reindex(
                model_weights.columns
            )
            .astype(float)
        )

        if (
            importance < 0
        ).any():
            raise ValueError(
                "model importance cannot be negative."
            )

        if np.isclose(
            importance.sum(),
            0.0,
        ):
            raise ValueError(
                "model importance cannot sum to zero."
            )

        importance = (
            importance
            / importance.sum()
        )

    consensus = (
        model_weights
        .mul(
            importance,
            axis=1,
        )
        .sum(axis=1)
    )

    consensus = (
        consensus
        / consensus.sum()
    )

    consensus.name = (
        "consensus_weight"
    )

    return consensus


def calculate_allocation_disagreement(
    model_weights: pd.DataFrame,
) -> pd.DataFrame:
    """
    Measure disagreement among portfolio models.

    Metrics
    -------
    allocation_std:
        Standard deviation of model weights for each asset.

    allocation_range:
        Maximum model weight minus minimum model weight.

    disagreement_score:
        Relative disagreement using allocation range divided
        by mean allocation.

        Higher values indicate greater model disagreement.
    """
    model_weights = validate_model_weights(
        model_weights
    )

    mean_weight = (
        model_weights.mean(
            axis=1
        )
    )

    allocation_std = (
        model_weights.std(
            axis=1,
            ddof=0,
        )
    )

    allocation_range = (
        model_weights.max(
            axis=1
        )
        - model_weights.min(
            axis=1
        )
    )

    denominator = mean_weight.replace(
        0.0,
        np.nan,
    )

    disagreement_score = (
        allocation_range
        / denominator
    ).fillna(0.0)

    return pd.DataFrame(
        {
            "mean_weight":
                mean_weight,
            "allocation_std":
                allocation_std,
            "allocation_range":
                allocation_range,
            "disagreement_score":
                disagreement_score,
        }
    )


def build_consensus_portfolio(
    model_weights: pd.DataFrame,
    model_importance: pd.Series | None = None,
) -> ConsensusResult:
    """
    Build a consensus portfolio and associated disagreement metrics.
    """
    model_weights = validate_model_weights(
        model_weights
    )

    consensus = (
        calculate_consensus_weights(
            model_weights,
            model_importance=
                model_importance,
        )
    )

    disagreement = (
        calculate_allocation_disagreement(
            model_weights
        )
    )

    overall_disagreement = float(
        disagreement[
            "allocation_std"
        ].mean()
    )

    return ConsensusResult(
        model_weights=model_weights,
        consensus_weights=consensus,
        allocation_std=disagreement[
            "allocation_std"
        ],
        allocation_range=disagreement[
            "allocation_range"
        ],
        disagreement_score=disagreement[
            "disagreement_score"
        ],
        overall_disagreement=
            overall_disagreement,
    )
