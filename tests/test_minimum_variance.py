import numpy as np
import pandas as pd
import pytest

from src.optimizers.minimum_variance import (
    build_equal_weight_portfolio,
    compare_minimum_variance_to_equal_weight,
    optimize_minimum_variance,
    portfolio_variance_objective,
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


def test_variance_objective():
    weights = np.array(
        [0.5, 0.5]
    )

    covariance = np.array(
        [
            [0.04, 0.01],
            [0.01, 0.02],
        ]
    )

    result = portfolio_variance_objective(
        weights,
        covariance,
    )

    expected = float(
        weights.T
        @ covariance
        @ weights
    )

    assert np.isclose(
        result,
        expected,
    )


def test_equal_weight_portfolio(
    sample_covariance,
):
    weights = (
        build_equal_weight_portfolio(
            sample_covariance
        )
    )

    assert np.isclose(
        weights.sum(),
        1.0,
    )

    assert np.allclose(
        weights.to_numpy(),
        np.array(
            [
                1 / 3,
                1 / 3,
                1 / 3,
            ]
        ),
    )


def test_minimum_variance_weights_sum_to_one(
    sample_covariance,
):
    result = optimize_minimum_variance(
        sample_covariance
    )

    assert np.isclose(
        result.weights.sum(),
        1.0,
    )


def test_minimum_variance_weights_are_nonnegative(
    sample_covariance,
):
    result = optimize_minimum_variance(
        sample_covariance
    )

    assert (
        result.weights >= 0
    ).all()


def test_minimum_variance_respects_max_weight(
    sample_covariance,
):
    result = optimize_minimum_variance(
        covariance=sample_covariance,
        max_weight=0.50,
    )

    assert (
        result.weights
        <= 0.50 + 1e-10
    ).all()


def test_minimum_variance_respects_min_weight(
    sample_covariance,
):
    result = optimize_minimum_variance(
        covariance=sample_covariance,
        min_weight=0.10,
    )

    assert (
        result.weights
        >= 0.10 - 1e-10
    ).all()


def test_minimum_variance_reduces_or_matches_equal_weight_risk(
    sample_covariance,
):
    comparison = (
        compare_minimum_variance_to_equal_weight(
            sample_covariance
        )
    )

    equal_volatility = comparison.loc[
        "portfolio_volatility",
        "equal_weight",
    ]

    optimized_volatility = comparison.loc[
        "portfolio_volatility",
        "minimum_variance",
    ]

    assert (
        optimized_volatility
        <= equal_volatility + 1e-10
    )


def test_optimizer_reports_success(
    sample_covariance,
):
    result = optimize_minimum_variance(
        sample_covariance
    )

    assert result.success is True

    assert result.iterations > 0

    assert (
        result.portfolio_volatility
        > 0
    )


def test_infeasible_max_weight_raises(
    sample_covariance,
):
    with pytest.raises(ValueError):
        optimize_minimum_variance(
            covariance=sample_covariance,
            max_weight=0.20,
        )


def test_infeasible_min_weight_raises(
    sample_covariance,
):
    with pytest.raises(ValueError):
        optimize_minimum_variance(
            covariance=sample_covariance,
            min_weight=0.40,
        )
