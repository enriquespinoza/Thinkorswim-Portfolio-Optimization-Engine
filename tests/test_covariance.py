import numpy as np
import pandas as pd
import pytest

from src.risk.covariance import (
    calculate_condition_number,
    calculate_ledoit_wolf_covariance,
    calculate_sample_covariance,
    compare_covariance_estimators,
    covariance_eigenvalues,
    covariance_to_correlation,
    validate_returns,
)


@pytest.fixture
def sample_returns():
    return pd.DataFrame(
        {
            "AAA": [
                0.010,
                -0.005,
                0.012,
                0.004,
                -0.003,
                0.008,
            ],
            "BBB": [
                0.006,
                -0.002,
                0.009,
                0.003,
                -0.001,
                0.005,
            ],
            "CCC": [
                -0.002,
                0.004,
                -0.003,
                0.006,
                0.002,
                -0.001,
            ],
        }
    )


def test_validate_returns(sample_returns):
    cleaned = validate_returns(sample_returns)

    assert cleaned.shape == sample_returns.shape
    assert not cleaned.isna().any().any()


def test_validate_returns_rejects_empty():
    with pytest.raises(ValueError):
        validate_returns(pd.DataFrame())


def test_sample_covariance_shape(sample_returns):
    covariance = calculate_sample_covariance(
        sample_returns
    )

    assert covariance.shape == (3, 3)

    assert list(covariance.columns) == [
        "AAA",
        "BBB",
        "CCC",
    ]

    assert list(covariance.index) == [
        "AAA",
        "BBB",
        "CCC",
    ]


def test_sample_covariance_is_symmetric(sample_returns):
    covariance = calculate_sample_covariance(
        sample_returns
    )

    assert np.allclose(
        covariance.to_numpy(),
        covariance.to_numpy().T,
    )


def test_ledoit_wolf_covariance_shape(sample_returns):
    covariance, shrinkage = (
        calculate_ledoit_wolf_covariance(
            sample_returns
        )
    )

    assert covariance.shape == (3, 3)

    assert 0.0 <= shrinkage <= 1.0


def test_ledoit_wolf_covariance_is_symmetric(
    sample_returns,
):
    covariance, _ = (
        calculate_ledoit_wolf_covariance(
            sample_returns
        )
    )

    assert np.allclose(
        covariance.to_numpy(),
        covariance.to_numpy().T,
    )


def test_covariance_to_correlation(sample_returns):
    covariance = calculate_sample_covariance(
        sample_returns
    )

    correlation = covariance_to_correlation(
        covariance
    )

    assert np.allclose(
        np.diag(correlation),
        1.0,
    )

    assert (
        correlation.to_numpy() <= 1.0
    ).all()

    assert (
        correlation.to_numpy() >= -1.0
    ).all()


def test_condition_number_positive(sample_returns):
    covariance = calculate_sample_covariance(
        sample_returns
    )

    condition_number = calculate_condition_number(
        covariance
    )

    assert condition_number > 0


def test_eigenvalues_count(sample_returns):
    covariance = calculate_sample_covariance(
        sample_returns
    )

    eigenvalues = covariance_eigenvalues(
        covariance
    )

    assert len(eigenvalues) == 3


def test_ledoit_wolf_positive_semidefinite(
    sample_returns,
):
    covariance, _ = (
        calculate_ledoit_wolf_covariance(
            sample_returns
        )
    )

    eigenvalues = covariance_eigenvalues(
        covariance
    )

    assert eigenvalues.min() >= -1e-12


def test_compare_covariance_estimators(
    sample_returns,
):
    comparison = compare_covariance_estimators(
        sample_returns
    )

    assert (
        comparison.sample_condition_number
        > 0
    )

    assert (
        comparison.ledoit_wolf_condition_number
        > 0
    )

    assert 0.0 <= comparison.shrinkage <= 1.0

    assert comparison.frobenius_difference >= 0
