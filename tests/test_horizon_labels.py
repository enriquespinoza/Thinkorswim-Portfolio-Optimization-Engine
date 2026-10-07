import pandas as pd

from src.features.horizon_labels import (
    build_forward_horizon_labels,
    build_forward_horizon_returns,
    compound_returns,
)


def build_period_results():
    dates = pd.date_range(
        "2026-01-31",
        periods=5,
        freq="ME",
    )

    rows = []

    for model, returns in {
        "equal_weight": [
            0.01,
            0.02,
            -0.01,
            0.03,
            0.01,
        ],
        "maximum_sharpe": [
            0.02,
            0.01,
            0.00,
            0.04,
            -0.01,
        ],
    }.items():
        for index, value in enumerate(
            returns
        ):
            rows.append(
                {
                    "rebalance_date":
                        dates[
                            index
                        ],

                    "period_end":
                        dates[
                            index
                        ]
                        + pd.offsets.MonthEnd(
                            1
                        ),

                    "model":
                        model,

                    "net_return":
                        value,
                }
            )

    return pd.DataFrame(
        rows
    )


def test_compound_returns():
    returns = pd.Series(
        [
            0.10,
            -0.05,
            0.02,
        ]
    )

    result = compound_returns(
        returns
    )

    expected = (
        1.10
        * 0.95
        * 1.02
        - 1.0
    )

    assert abs(
        result
        - expected
    ) < 1e-12


def test_three_period_returns():
    data = (
        build_period_results()
    )

    result = (
        build_forward_horizon_returns(
            data,
            horizon_periods=3,
        )
    )

    counts = (
        result
        .groupby(
            "model"
        )
        .size()
    )

    assert (
        counts
        == 3
    ).all()


def test_forward_horizon_labels():
    data = (
        build_period_results()
    )

    labels = (
        build_forward_horizon_labels(
            data,
            horizon_periods=3,
        )
    )

    assert len(
        labels
    ) == 3

    assert (
        "maximum_sharpe_excess_vs_equal_weight"
        in labels.columns
    )

    assert (
        "horizon_end"
        in labels.columns
    )
