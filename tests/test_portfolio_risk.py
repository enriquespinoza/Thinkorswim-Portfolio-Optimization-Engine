import numpy as np
import pandas as pd
import pytest

from src.risk.portfolio_risk import (
    align_weights,
    calculate_component_risk_contribution,
    calculate_marginal_risk_contribution,
    calculate_percentage_risk_contribution,
    calculate_portfolio_variance,
    calculate_portfolio_volatility,
    risk_concentration_index,
    risk_contribution_table,
    validate_covariance,
)


@pytest.fixture
def sample_covariance():
    return pd.DataFrame(
        [
            [0.0400, 0.0060],
            [0.0060, 0.0100],
        ],
        index=["AAA", "BBB"],
        columns=["AAA", "BBB"],
    )


@pytest.fixture
def sample_weights():
    return pd.Series(
        {
            "AAA": 0.60,
            "BBB": 0.40,
        }
    )


def test_validate_covariance(
    sample_covariance,
):
    cleaned = validate_covariance(
        sample_covariance
    )

    assert cleaned.shape == (2, 2)


def test_validate_covariance_rejects_nonsquare():
    covariance = pd.DataFrame(
        [
            [0.04, 0.01, 0.02],
            [0.01, 0.03, 0.01],
        ]
    )

    with pytest.raises(ValueError):
        validate_covariance(
            covariance
        )


def test_align_weights(
    sample_weights,
    sample_covariance,
):
    weights = align_weights(
        sample_weights,
        sample_covariance,
    )

    assert list(weights.index) == [
        "AAA",
        "BBB",
    ]


def test_align_weights_reorders_assets(
    sample_covariance,
):
    weights = pd.Series(
        {
            "BBB": 0.40,
            "AAA": 0.60,
        }
    )

    aligned = align_weights(
        weights,
        sample_covariance,
    )

    assert list(aligned.index) == [
        "AAA",
        "BBB",
    ]

    assert np.isclose(
        aligned["AAA"],
        0.60,
    )


def test_portfolio_variance(
    sample_weights,
    sample_covariance,
):
    calculated = calculate_portfolio_variance(
        sample_weights,
        sample_covariance,
    )

    w = sample_weights.reindex(
        sample_covariance.columns
    ).to_numpy()

    expected = float(
        w.T
        @ sample_covariance.to_numpy()
        @ w
    )

    assert np.isclose(
        calculated,
        expected,
    )


def test_portfolio_volatility(
    sample_weights,
    sample_covariance,
):
    variance = calculate_portfolio_variance(
        sample_weights,
        sample_covariance,
    )

    volatility = calculate_portfolio_volatility(
        sample_weights,
        sample_covariance,
    )

    assert np.isclose(
        volatility,
        np.sqrt(variance),
    )


def test_marginal_risk_contribution(
    sample_weights,
    sample_covariance,
):
    marginal = (
        calculate_marginal_risk_contribution(
            sample_weights,
            sample_covariance,
        )
    )

    assert len(marginal) == 2

    assert list(marginal.index) == [
        "AAA",
        "BBB",
    ]


def test_component_risk_sums_to_volatility(
    sample_weights,
    sample_covariance,
):
    component = (
        calculate_component_risk_contribution(
            sample_weights,
            sample_covariance,
        )
    )

    volatility = calculate_portfolio_volatility(
        sample_weights,
        sample_covariance,
    )

    assert np.isclose(
        component.sum(),
        volatility,
    )


def test_percentage_risk_sums_to_one(
    sample_weights,
    sample_covariance,
):
    percentage = (
        calculate_percentage_risk_contribution(
            sample_weights,
            sample_covariance,
        )
    )

    assert np.isclose(
        percentage.sum(),
        1.0,
    )


def test_risk_contribution_table(
    sample_weights,
    sample_covariance,
):
    table = risk_contribution_table(
        sample_weights,
        sample_covariance,
    )

    expected_columns = {
        "weight",
        "asset_volatility",
        "marginal_risk_contribution",
        "component_risk_contribution",
        "percentage_risk_contribution",
    }

    assert expected_columns.issubset(
        table.columns
    )

    assert len(table) == 2


def test_risk_concentration_index(
    sample_weights,
    sample_covariance,
):
    concentration = (
        risk_concentration_index(
            sample_weights,
            sample_covariance,
        )
    )

    assert concentration > 0

def test_effective_number_of_risk_positions(
    sample_weights,
    sample_covariance,
):
    from src.risk.portfolio_risk import (
        effective_number_of_risk_positions,
    )

    effective = (
        effective_number_of_risk_positions(
            sample_weights,
            sample_covariance,
        )
    )

    assert effective > 0
    assert effective <= len(
        sample_weights
    ) + 1e-10