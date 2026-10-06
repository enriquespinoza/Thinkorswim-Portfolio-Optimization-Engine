import numpy as np
import pandas as pd
import pytest

from src.ensemble.consensus import (
    build_consensus_portfolio,
    calculate_allocation_disagreement,
    calculate_consensus_weights,
    validate_model_weights,
)


@pytest.fixture
def model_weights():
    return pd.DataFrame(
        {
            "minimum_variance": [
                0.20,
                0.10,
                0.30,
                0.40,
            ],
            "risk_parity": [
                0.25,
                0.20,
                0.25,
                0.30,
            ],
            "hrp": [
                0.30,
                0.15,
                0.25,
                0.30,
            ],
        },
        index=[
            "AAA",
            "BBB",
            "CCC",
            "DDD",
        ],
    )


def test_validate_model_weights(
    model_weights,
):
    cleaned = (
        validate_model_weights(
            model_weights
        )
    )

    assert cleaned.shape == (
        4,
        3,
    )


def test_equal_weight_consensus(
    model_weights,
):
    consensus = (
        calculate_consensus_weights(
            model_weights
        )
    )

    expected = (
        model_weights.mean(
            axis=1
        )
    )

    assert np.allclose(
        consensus.to_numpy(),
        expected.to_numpy(),
    )


def test_consensus_sums_to_one(
    model_weights,
):
    consensus = (
        calculate_consensus_weights(
            model_weights
        )
    )

    assert np.isclose(
        consensus.sum(),
        1.0,
    )


def test_weighted_consensus(
    model_weights,
):
    importance = pd.Series(
        {
            "minimum_variance": 0.50,
            "risk_parity": 0.30,
            "hrp": 0.20,
        }
    )

    consensus = (
        calculate_consensus_weights(
            model_weights,
            model_importance=importance,
        )
    )

    assert np.isclose(
        consensus.sum(),
        1.0,
    )


def test_disagreement_metrics(
    model_weights,
):
    disagreement = (
        calculate_allocation_disagreement(
            model_weights
        )
    )

    assert (
        disagreement[
            "allocation_std"
        ] >= 0
    ).all()

    assert (
        disagreement[
            "allocation_range"
        ] >= 0
    ).all()


def test_identical_models_have_zero_disagreement():
    weights = pd.DataFrame(
        {
            "model_a": [
                0.25,
                0.25,
                0.25,
                0.25,
            ],
            "model_b": [
                0.25,
                0.25,
                0.25,
                0.25,
            ],
        },
        index=[
            "AAA",
            "BBB",
            "CCC",
            "DDD",
        ],
    )

    result = (
        build_consensus_portfolio(
            weights
        )
    )

    assert np.isclose(
        result.overall_disagreement,
        0.0,
    )


def test_build_consensus_portfolio(
    model_weights,
):
    result = (
        build_consensus_portfolio(
            model_weights
        )
    )

    assert np.isclose(
        result.consensus_weights.sum(),
        1.0,
    )

    assert (
        result.overall_disagreement
        >= 0
    )


def test_model_weights_must_sum_to_one():
    invalid = pd.DataFrame(
        {
            "model_a": [
                0.50,
                0.40,
            ],
            "model_b": [
                0.50,
                0.50,
            ],
        },
        index=[
            "AAA",
            "BBB",
        ],
    )

    with pytest.raises(
        ValueError
    ):
        validate_model_weights(
            invalid
        )
