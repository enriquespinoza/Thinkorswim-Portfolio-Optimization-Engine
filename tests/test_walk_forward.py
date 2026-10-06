import numpy as np
import pandas as pd

from src.backtest.walk_forward import (
    calculate_drifted_weights,
    calculate_period_asset_returns,
    calculate_turnover,
    generate_month_end_dates,
)


def test_generate_month_end_dates():
    dates = pd.bdate_range(
        "2026-01-01",
        "2026-03-31",
    )

    returns = pd.DataFrame(
        {
            "AAA": 0.001,
            "BBB": 0.002,
        },
        index=dates,
    )

    month_ends = generate_month_end_dates(
        returns
    )

    assert len(month_ends) == 3


def test_period_asset_returns():
    returns = pd.DataFrame(
        {
            "AAA": [
                0.01,
                0.02,
                0.03,
            ],
        },
        index=pd.to_datetime(
            [
                "2026-01-30",
                "2026-02-02",
                "2026-02-03",
            ]
        ),
    )

    result = calculate_period_asset_returns(
        returns=returns,
        start_date=pd.Timestamp(
            "2026-01-30"
        ),
        end_date=pd.Timestamp(
            "2026-02-03"
        ),
    )

    expected = (
        1.02
        * 1.03
        - 1.0
    )

    assert np.isclose(
        result["AAA"],
        expected,
    )


def test_turnover_identical_portfolios():
    weights = pd.Series(
        {
            "AAA": 0.5,
            "BBB": 0.5,
        }
    )

    turnover = calculate_turnover(
        current_weights=weights,
        previous_weights=weights,
    )

    assert np.isclose(
        turnover,
        0.0,
    )


def test_turnover_reallocation():
    old_weights = pd.Series(
        {
            "AAA": 0.5,
            "BBB": 0.5,
        }
    )

    new_weights = pd.Series(
        {
            "AAA": 0.7,
            "BBB": 0.3,
        }
    )

    turnover = calculate_turnover(
        current_weights=new_weights,
        previous_weights=old_weights,
    )

    expected = (
        0.5
        * (
            abs(0.7 - 0.5)
            + abs(0.3 - 0.5)
        )
    )

    assert np.isclose(
        turnover,
        expected,
    )


def test_initial_turnover_is_one():
    weights = pd.Series(
        {
            "AAA": 0.6,
            "BBB": 0.4,
        }
    )

    turnover = calculate_turnover(
        current_weights=weights,
        previous_weights=None,
    )

    assert np.isclose(
        turnover,
        1.0,
    )
def test_calculate_drifted_weights():
    starting_weights = pd.Series(
        {
            "AAA": 0.50,
            "BBB": 0.50,
        }
    )

    asset_returns = pd.Series(
        {
            "AAA": 0.10,
            "BBB": 0.00,
        }
    )

    drifted = (
        calculate_drifted_weights(
            starting_weights,
            asset_returns,
        )
    )

    expected_aaa = (
        0.50 * 1.10
    ) / (
        0.50 * 1.10
        + 0.50
    )

    assert np.isclose(
        drifted["AAA"],
        expected_aaa,
    )

    assert np.isclose(
        drifted.sum(),
        1.0,
    )


def test_rebalance_after_drift():
    target = pd.Series(
        {
            "AAA": 0.50,
            "BBB": 0.50,
        }
    )

    asset_returns = pd.Series(
        {
            "AAA": 0.10,
            "BBB": 0.00,
        }
    )

    drifted = (
        calculate_drifted_weights(
            target,
            asset_returns,
        )
    )

    turnover = (
        calculate_turnover(
            current_weights=target,
            previous_weights=drifted,
        )
    )

    assert turnover > 0