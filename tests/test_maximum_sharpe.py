import numpy as np
import pandas as pd
import pytest

from src.optimizers.maximum_sharpe import (
    calculate_portfolio_expected_return,
    negative_sharpe_objective,
    optimize_maximum_sharpe,
    validate_expected_returns,
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


@pytest.fixture
def sample_expected_returns():
    return pd.Series(
        {
            "AAA": 0.15,
            "BBB": 0.10,
            "CCC": 0.06,
        }
    )


def test_validate_expected_returns(
    sample_expected_returns,
    sample_covariance,
):
    result = (
        validate_expected_returns(
            sample_expected_returns,
            sample_covariance,
        )
    )

    assert list(
        result.index
    ) == list(
        sample_covariance.columns
    )


def test_portfolio_expected_return():
    weights = pd.Series(
        {
            "AAA": 0.60,
            "BBB": 0.40,
        }
    )

    expected_returns = pd.Series(
        {
            "AAA": 0.10,
            "BBB": 0.05,
        }
    )

    result = (
        calculate_portfolio_expected_return(
            weights,
            expected_returns,
        )
    )

    expected = (
        0.60 * 0.10
        + 0.40 * 0.05
    )

    assert np.isclose(
        result,
        expected,
    )


def test_negative_sharpe_objective():
    weights = np.array(
        [
            0.50,
            0.50,
        ]
    )

    expected_returns = np.array(
        [
            0.10,
            0.06,
        ]
    )

    covariance = np.array(
        [
            [0.04, 0.01],
            [0.01, 0.02],
        ]
    )

    result = (
        negative_sharpe_objective(
            weights,
            expected_returns,
            covariance,
            0.0,
        )
    )

    assert np.isfinite(
        result
    )


def test_maximum_sharpe_weights_sum_to_one(
    sample_expected_returns,
    sample_covariance,
):
    result = (
        optimize_maximum_sharpe(
            expected_returns=
                sample_expected_returns,
            covariance=
                sample_covariance,
        )
    )

    assert np.isclose(
        result.weights.sum(),
        1.0,
    )


def test_maximum_sharpe_nonnegative_weights(
    sample_expected_returns,
    sample_covariance,
):
    result = (
        optimize_maximum_sharpe(
            expected_returns=
                sample_expected_returns,
            covariance=
                sample_covariance,
        )
    )

    assert (
        result.weights >= 0
    ).all()


def test_maximum_sharpe_respects_max_weight(
    sample_expected_returns,
    sample_covariance,
):
    result = (
        optimize_maximum_sharpe(
            expected_returns=
                sample_expected_returns,
            covariance=
                sample_covariance,
            max_weight=0.50,
        )
    )

    assert (
        result.weights
        <= 0.50 + 1e-10
    ).all()


def test_maximum_sharpe_reports_success(
    sample_expected_returns,
    sample_covariance,
):
    result = (
        optimize_maximum_sharpe(
            expected_returns=
                sample_expected_returns,
            covariance=
                sample_covariance,
        )
    )

    assert result.success is True

    assert (
        result.portfolio_volatility
        > 0
    )

    assert np.isfinite(
        result.sharpe_ratio
    )


def test_maximum_sharpe_beats_equal_weight_sharpe(
    sample_expected_returns,
    sample_covariance,
):
    result = (
        optimize_maximum_sharpe(
            expected_returns=
                sample_expected_returns,
            covariance=
                sample_covariance,
        )
    )

    equal_weights = np.array(
        [
            1 / 3,
            1 / 3,
            1 / 3,
        ]
    )

    equal_negative_sharpe = (
        negative_sharpe_objective(
            equal_weights,
            sample_expected_returns.to_numpy(),
            sample_covariance.to_numpy(),
            0.0,
        )
    )

    equal_sharpe = (
        -equal_negative_sharpe
    )

    assert (
        result.sharpe_ratio
        >= equal_sharpe - 1e-8
    )


def test_missing_expected_return_raises(
    sample_covariance,
):
    expected_returns = pd.Series(
        {
            "AAA": 0.10,
            "BBB": 0.08,
        }
    )

    with pytest.raises(
        ValueError
    ):
        optimize_maximum_sharpe(
            expected_returns=
                expected_returns,
            covariance=
                sample_covariance,
        )


def test_infeasible_max_weight_raises(
    sample_expected_returns,
    sample_covariance,
):
    with pytest.raises(
        ValueError
    ):
        optimize_maximum_sharpe(
            expected_returns=
                sample_expected_returns,
            covariance=
                sample_covariance,
            max_weight=0.20,
        )
