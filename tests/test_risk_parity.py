import numpy as np
import pandas as pd
import pytest

from src.optimizers.risk_parity import (
    optimize_risk_parity,
    percentage_risk_contributions_array,
    risk_parity_objective,
)


@pytest.fixture
def sample_covariance():
    return pd.DataFrame(
        [
            [0.0400, 0.0060, 0.0040],
            [0.0060, 0.0225, 0.0030],
            [0.0040, 0.0030, 0.0100],
        ],
        index=[
            "AAA",
            "BBB",
            "CCC",
        ],
        columns=[
            "AAA",
            "BBB",
            "CCC",
        ],
    )


def test_percentage_risk_contributions_sum_to_one(
    sample_covariance,
):
    weights = np.array(
        [
            1 / 3,
            1 / 3,
            1 / 3,
        ]
    )

    contributions = (
        percentage_risk_contributions_array(
            weights,
            sample_covariance.to_numpy(),
        )
    )

    assert np.isclose(
        contributions.sum(),
        1.0,
    )


def test_risk_parity_objective_nonnegative(
    sample_covariance,
):
    weights = np.array(
        [
            1 / 3,
            1 / 3,
            1 / 3,
        ]
    )

    target = np.array(
        [
            1 / 3,
            1 / 3,
            1 / 3,
        ]
    )

    objective = (
        risk_parity_objective(
            weights,
            sample_covariance.to_numpy(),
            target,
        )
    )

    assert objective >= 0


def test_risk_parity_weights_sum_to_one(
    sample_covariance,
):
    result = optimize_risk_parity(
        sample_covariance
    )

    assert np.isclose(
        result.weights.sum(),
        1.0,
    )


def test_risk_parity_weights_are_nonnegative(
    sample_covariance,
):
    result = optimize_risk_parity(
        sample_covariance
    )

    assert (
        result.weights >= 0
    ).all()


def test_risk_parity_equalizes_risk(
    sample_covariance,
):
    result = optimize_risk_parity(
        sample_covariance
    )

    target = (
        1.0
        / len(sample_covariance)
    )

    contributions = (
        result.percentage_risk_contribution
    )

    assert np.allclose(
        contributions.to_numpy(),
        target,
        atol=1e-5,
    )


def test_risk_parity_respects_max_weight(
    sample_covariance,
):
    result = optimize_risk_parity(
        covariance=sample_covariance,
        max_weight=0.50,
    )

    assert (
        result.weights
        <= 0.50 + 1e-10
    ).all()


def test_risk_parity_respects_min_weight(
    sample_covariance,
):
    result = optimize_risk_parity(
        covariance=sample_covariance,
        min_weight=0.10,
    )

    assert (
        result.weights
        >= 0.10 - 1e-10
    ).all()


def test_risk_parity_reports_success(
    sample_covariance,
):
    result = optimize_risk_parity(
        sample_covariance
    )

    assert result.success is True
    assert result.iterations > 0
    assert result.portfolio_volatility > 0
    assert result.risk_contribution_error >= 0


def test_diagonal_covariance_inverse_volatility_relationship():
    covariance = pd.DataFrame(
        [
            [0.04, 0.00, 0.00],
            [0.00, 0.01, 0.00],
            [0.00, 0.00, 0.0225],
        ],
        index=[
            "AAA",
            "BBB",
            "CCC",
        ],
        columns=[
            "AAA",
            "BBB",
            "CCC",
        ],
    )

    result = optimize_risk_parity(
        covariance
    )

    volatility = np.array(
        [
            0.20,
            0.10,
            0.15,
        ]
    )

    expected = (
        1 / volatility
    )

    expected = (
        expected
        / expected.sum()
    )

    assert np.allclose(
        result.weights.to_numpy(),
        expected,
        atol=1e-5,
    )


def test_infeasible_max_weight_raises(
    sample_covariance,
):
    with pytest.raises(
        ValueError
    ):
        optimize_risk_parity(
            covariance=sample_covariance,
            max_weight=0.20,
        )


def test_infeasible_min_weight_raises(
    sample_covariance,
):
    with pytest.raises(
        ValueError
    ):
        optimize_risk_parity(
            covariance=sample_covariance,
            min_weight=0.40,
        )
