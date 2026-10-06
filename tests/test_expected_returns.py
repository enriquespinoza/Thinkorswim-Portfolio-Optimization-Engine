import numpy as np
import pandas as pd
import pytest

from src.forecasts.expected_returns import (
    calculate_arithmetic_expected_return,
    calculate_blended_expected_return,
    calculate_estimator_dispersion,
    calculate_exponentially_weighted_expected_return,
    calculate_geometric_expected_return,
    estimate_expected_returns,
    expected_return_table,
)


@pytest.fixture
def sample_returns():
    return pd.DataFrame(
        {
            "AAA": [
                0.010,
                0.005,
                -0.002,
                0.008,
                0.003,
            ],
            "BBB": [
                -0.004,
                0.002,
                0.001,
                0.003,
                0.005,
            ],
        }
    )


def test_geometric_expected_return(
    sample_returns,
):
    result = (
        calculate_geometric_expected_return(
            sample_returns,
            trading_days=252,
        )
    )

    assert set(
        result.index
    ) == {
        "AAA",
        "BBB",
    }

    assert np.isfinite(
        result.to_numpy()
    ).all()


def test_arithmetic_expected_return(
    sample_returns,
):
    result = (
        calculate_arithmetic_expected_return(
            sample_returns,
            trading_days=252,
        )
    )

    expected_aaa = (
        sample_returns[
            "AAA"
        ].mean()
        * 252
    )

    assert np.isclose(
        result["AAA"],
        expected_aaa,
    )


def test_ewma_expected_return(
    sample_returns,
):
    result = (
        calculate_exponentially_weighted_expected_return(
            sample_returns,
            span=3,
            trading_days=252,
        )
    )

    assert len(
        result
    ) == 2

    assert np.isfinite(
        result.to_numpy()
    ).all()


def test_ewma_invalid_span(
    sample_returns,
):
    with pytest.raises(
        ValueError
    ):
        calculate_exponentially_weighted_expected_return(
            sample_returns,
            span=1,
        )


def test_equal_weight_blend():
    geometric = pd.Series(
        {
            "AAA": 0.10,
            "BBB": 0.05,
        }
    )

    arithmetic = pd.Series(
        {
            "AAA": 0.12,
            "BBB": 0.07,
        }
    )

    ewma = pd.Series(
        {
            "AAA": 0.20,
            "BBB": 0.08,
        }
    )

    blended = (
        calculate_blended_expected_return(
            geometric,
            arithmetic,
            ewma,
        )
    )

    assert np.isclose(
        blended["AAA"],
        (
            0.10
            + 0.12
            + 0.20
        )
        / 3,
    )


def test_custom_blend_weights():
    geometric = pd.Series(
        {
            "AAA": 0.10,
        }
    )

    arithmetic = pd.Series(
        {
            "AAA": 0.20,
        }
    )

    ewma = pd.Series(
        {
            "AAA": 0.30,
        }
    )

    blended = (
        calculate_blended_expected_return(
            geometric,
            arithmetic,
            ewma,
            geometric_weight=0.20,
            arithmetic_weight=0.30,
            exponentially_weighted_weight=0.50,
        )
    )

    expected = (
        0.10 * 0.20
        + 0.20 * 0.30
        + 0.30 * 0.50
    )

    assert np.isclose(
        blended["AAA"],
        expected,
    )


def test_blend_weights_are_normalized():
    geometric = pd.Series(
        {
            "AAA": 0.10,
        }
    )

    arithmetic = pd.Series(
        {
            "AAA": 0.20,
        }
    )

    ewma = pd.Series(
        {
            "AAA": 0.30,
        }
    )

    result = (
        calculate_blended_expected_return(
            geometric,
            arithmetic,
            ewma,
            geometric_weight=2,
            arithmetic_weight=3,
            exponentially_weighted_weight=5,
        )
    )

    expected = (
        0.10 * 0.20
        + 0.20 * 0.30
        + 0.30 * 0.50
    )

    assert np.isclose(
        result["AAA"],
        expected,
    )


def test_estimator_dispersion():
    geometric = pd.Series(
        {
            "AAA": 0.10,
        }
    )

    arithmetic = pd.Series(
        {
            "AAA": 0.10,
        }
    )

    ewma = pd.Series(
        {
            "AAA": 0.10,
        }
    )

    dispersion = (
        calculate_estimator_dispersion(
            geometric,
            arithmetic,
            ewma,
        )
    )

    assert np.isclose(
        dispersion["AAA"],
        0.0,
    )


def test_complete_expected_return_result(
    sample_returns,
):
    result = (
        estimate_expected_returns(
            sample_returns,
            ewma_span=3,
        )
    )

    assert len(
        result.blended
    ) == 2

    assert (
        result.blended.index.equals(
            sample_returns.columns
        )
    )


def test_expected_return_table(
    sample_returns,
):
    table = (
        expected_return_table(
            sample_returns,
            ewma_span=3,
        )
    )

    expected_columns = {
        "geometric",
        "arithmetic",
        "ewma",
        "blended",
        "estimator_dispersion",
    }

    assert expected_columns.issubset(
        table.columns
    )
