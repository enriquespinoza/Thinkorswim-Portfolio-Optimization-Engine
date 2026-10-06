import numpy as np
import pandas as pd
import pytest

from src.optimizers.hrp import (
    calculate_cluster_variance,
    correlation_to_distance,
    hierarchical_cluster_order,
    inverse_variance_weights,
    optimize_hrp,
    recursive_bisection,
)


@pytest.fixture
def sample_covariance():
    return pd.DataFrame(
        [
            [0.0400, 0.0180, 0.0020, 0.0010],
            [0.0180, 0.0324, 0.0020, 0.0010],
            [0.0020, 0.0020, 0.0144, 0.0060],
            [0.0010, 0.0010, 0.0060, 0.0100],
        ],
        index=[
            "AAA",
            "BBB",
            "CCC",
            "DDD",
        ],
        columns=[
            "AAA",
            "BBB",
            "CCC",
            "DDD",
        ],
    )


def test_correlation_to_distance():
    correlation = pd.DataFrame(
        [
            [1.0, 0.5],
            [0.5, 1.0],
        ],
        index=["AAA", "BBB"],
        columns=["AAA", "BBB"],
    )

    distance = (
        correlation_to_distance(
            correlation
        )
    )

    expected = np.sqrt(
        (1.0 - 0.5) / 2.0
    )

    assert np.isclose(
        distance.loc[
            "AAA",
            "BBB",
        ],
        expected,
    )

    assert np.isclose(
        distance.loc[
            "AAA",
            "AAA",
        ],
        0.0,
    )


def test_distance_matrix_is_symmetric():
    correlation = pd.DataFrame(
        [
            [1.0, 0.4],
            [0.4, 1.0],
        ],
        index=["AAA", "BBB"],
        columns=["AAA", "BBB"],
    )

    distance = (
        correlation_to_distance(
            correlation
        )
    )

    assert np.allclose(
        distance.to_numpy(),
        distance.to_numpy().T,
    )


def test_inverse_variance_weights_sum_to_one(
    sample_covariance,
):
    weights = (
        inverse_variance_weights(
            sample_covariance
        )
    )

    assert np.isclose(
        weights.sum(),
        1.0,
    )


def test_inverse_variance_favors_lower_variance():
    covariance = pd.DataFrame(
        [
            [0.04, 0.00],
            [0.00, 0.01],
        ],
        index=[
            "HIGH_VOL",
            "LOW_VOL",
        ],
        columns=[
            "HIGH_VOL",
            "LOW_VOL",
        ],
    )

    weights = (
        inverse_variance_weights(
            covariance
        )
    )

    assert (
        weights["LOW_VOL"]
        > weights["HIGH_VOL"]
    )


def test_cluster_variance_positive(
    sample_covariance,
):
    variance = (
        calculate_cluster_variance(
            sample_covariance,
            ["AAA", "BBB"],
        )
    )

    assert variance > 0


def test_hierarchical_cluster_order_contains_all_assets(
    sample_covariance,
):
    ordered_assets, linkage_matrix = (
        hierarchical_cluster_order(
            sample_covariance
        )
    )

    assert set(
        ordered_assets
    ) == set(
        sample_covariance.columns
    )

    assert (
        linkage_matrix.shape[0]
        == len(sample_covariance) - 1
    )


def test_recursive_bisection_weights_sum_to_one(
    sample_covariance,
):
    ordered_assets, _ = (
        hierarchical_cluster_order(
            sample_covariance
        )
    )

    weights = recursive_bisection(
        covariance=sample_covariance,
        ordered_assets=ordered_assets,
    )

    assert np.isclose(
        weights.sum(),
        1.0,
    )


def test_recursive_bisection_nonnegative(
    sample_covariance,
):
    ordered_assets, _ = (
        hierarchical_cluster_order(
            sample_covariance
        )
    )

    weights = recursive_bisection(
        covariance=sample_covariance,
        ordered_assets=ordered_assets,
    )

    assert (
        weights >= 0
    ).all()


def test_hrp_weights_sum_to_one(
    sample_covariance,
):
    result = optimize_hrp(
        sample_covariance
    )

    assert np.isclose(
        result.weights.sum(),
        1.0,
    )


def test_hrp_weights_are_nonnegative(
    sample_covariance,
):
    result = optimize_hrp(
        sample_covariance
    )

    assert (
        result.weights >= 0
    ).all()


def test_hrp_reports_success(
    sample_covariance,
):
    result = optimize_hrp(
        sample_covariance
    )

    assert result.success is True

    assert (
        result.portfolio_volatility
        > 0
    )

    assert (
        result.risk_concentration_index
        > 0
    )


def test_hrp_preserves_asset_names(
    sample_covariance,
):
    result = optimize_hrp(
        sample_covariance
    )

    assert set(
        result.weights.index
    ) == set(
        sample_covariance.columns
    )


def test_single_asset_hrp():
    covariance = pd.DataFrame(
        [[0.04]],
        index=["AAA"],
        columns=["AAA"],
    )

    result = optimize_hrp(
        covariance
    )

    assert np.isclose(
        result.weights["AAA"],
        1.0,
    )

    assert (
        result.ordered_assets
        == ("AAA",)
    )

    assert (
        result.linkage_matrix
        is None
    )
