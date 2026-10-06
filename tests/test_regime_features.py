import numpy as np
import pandas as pd
import pytest

from src.features.regime_features import (
    build_regime_feature_matrix,
    build_regime_features_for_date,
    calculate_average_pairwise_correlation,
    calculate_current_drawdown,
    calculate_pair_correlation,
    calculate_trailing_return,
    calculate_trailing_volatility,
)


@pytest.fixture
def sample_returns():
    np.random.seed(
        42
    )

    dates = pd.bdate_range(
        "2024-01-01",
        periods=300,
    )

    return pd.DataFrame(
        {
            "SPY":
                np.random.normal(
                    0.0005,
                    0.01,
                    len(dates),
                ),

            "QQQ":
                np.random.normal(
                    0.0007,
                    0.013,
                    len(dates),
                ),

            "SCHD":
                np.random.normal(
                    0.0004,
                    0.009,
                    len(dates),
                ),

            "TLT":
                np.random.normal(
                    0.0001,
                    0.008,
                    len(dates),
                ),

            "GLD":
                np.random.normal(
                    0.0003,
                    0.009,
                    len(dates),
                ),
        },
        index=dates,
    )


def test_trailing_return(
    sample_returns,
):
    result = (
        calculate_trailing_return(
            sample_returns,
            lookback=21,
        )
    )

    expected = (
        (
            1.0
            + sample_returns.tail(
                21
            )
        )
        .prod()
        - 1.0
    )

    assert np.allclose(
        result.to_numpy(),
        expected.to_numpy(),
    )


def test_trailing_volatility(
    sample_returns,
):
    result = (
        calculate_trailing_volatility(
            sample_returns,
            lookback=21,
        )
    )

    assert (
        result > 0
    ).all()


def test_current_drawdown(
    sample_returns,
):
    result = (
        calculate_current_drawdown(
            sample_returns,
            lookback=126,
        )
    )

    assert (
        result <= 0
    ).all()


def test_pair_correlation(
    sample_returns,
):
    result = (
        calculate_pair_correlation(
            sample_returns,
            "SPY",
            "TLT",
            lookback=63,
        )
    )

    assert (
        -1.0
        <= result
        <= 1.0
    )


def test_average_pairwise_correlation(
    sample_returns,
):
    result = (
        calculate_average_pairwise_correlation(
            sample_returns,
            lookback=63,
        )
    )

    assert (
        -1.0
        <= result
        <= 1.0
    )


def test_regime_features_contain_expected_fields(
    sample_returns,
):
    date = (
        sample_returns.index[
            -1
        ]
    )

    result = (
        build_regime_features_for_date(
            sample_returns,
            date,
        )
    )

    expected = {
        "SPY_return_21d",
        "SPY_return_63d",
        "SPY_return_126d",
        "SPY_vol_21d",
        "SPY_vol_63d",
        "SPY_drawdown_126d",
        "equity_bond_corr_63d",
        "equity_bond_momentum_spread_63d",
        "average_pairwise_corr_63d",
    }

    assert expected.issubset(
        result.index
    )


def test_no_lookahead(
    sample_returns,
):
    as_of_date = (
        sample_returns.index[
            199
        ]
    )

    original = (
        build_regime_features_for_date(
            sample_returns,
            as_of_date,
        )
    )

    modified = (
        sample_returns.copy()
    )

    modified.loc[
        modified.index
        > as_of_date
    ] = 999.0

    recalculated = (
        build_regime_features_for_date(
            modified,
            as_of_date,
        )
    )

    assert np.allclose(
        original.to_numpy(),
        recalculated.to_numpy(),
    )


def test_feature_matrix(
    sample_returns,
):
    dates = pd.DatetimeIndex(
        [
            sample_returns.index[
                150
            ],
            sample_returns.index[
                200
            ],
            sample_returns.index[
                250
            ],
        ]
    )

    result = (
        build_regime_feature_matrix(
            sample_returns,
            dates,
        )
    )

    assert len(
        result
    ) == 3

    assert (
        result.index.name
        == "rebalance_date"
    )
