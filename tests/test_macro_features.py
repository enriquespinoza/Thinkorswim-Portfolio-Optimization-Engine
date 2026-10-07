import numpy as np
import pandas as pd

from src.features.macro_features import (
    align_macro_to_calendar,
    build_macro_feature_matrix,
    build_macro_features_for_date,
)


def build_sample_macro_data():
    dates = pd.bdate_range(
        "2023-01-02",
        periods=500,
    )

    x = np.arange(
        len(dates),
        dtype=float,
    )

    return pd.DataFrame(
        {
            "yield_2y":
                4.0
                + x * 0.0005,

            "yield_10y":
                4.2
                + x * 0.0004,

            "yield_30y":
                4.5
                + x * 0.0003,

            "fed_funds":
                4.1
                + x * 0.0002,

            "breakeven_10y":
                2.2
                + x * 0.0001,

            "real_yield_10y":
                2.0
                + x * 0.0002,

            "unemployment":
                4.0
                + x * 0.0001,

            "initial_claims":
                200000
                + x * 10,

            "payrolls":
                155000
                + x * 20,

            "retail_sales":
                700000
                + x * 50,
        },
        index=dates,
    )


def test_align_macro_to_calendar():
    macro = (
        build_sample_macro_data()
    )

    calendar = macro.index[
        100:200
    ]

    result = (
        align_macro_to_calendar(
            macro,
            calendar,
        )
    )

    assert result.index.equals(
        calendar
    )


def test_build_macro_features():
    macro = (
        build_sample_macro_data()
    )

    date = macro.index[
        300
    ]

    result = (
        build_macro_features_for_date(
            macro_data=macro,
            calendar_index=
                macro.index,
            as_of_date=date,
        )
    )

    expected = {
        "yield_2y_level",
        "yield_10y_level",
        "yield_30y_level",
        "curve_10y_2y_bp",
        "curve_30y_10y_bp",
        "policy_10y_ff_bp",
        "yield_10y_change_21d_bp",
        "yield_10y_change_63d_bp",
        "yield_10y_change_std_21d_bp",
        "unemployment_change_126d_pp",
        "initial_claims_change_63d_pct",
        "payrolls_change_126d_pct",
        "retail_sales_change_126d_pct",
    }

    assert expected.issubset(
        result.index
    )


def test_macro_no_lookahead():
    macro = (
        build_sample_macro_data()
    )

    as_of_date = macro.index[
        300
    ]

    original = (
        build_macro_features_for_date(
            macro_data=macro,
            calendar_index=
                macro.index,
            as_of_date=
                as_of_date,
        )
    )

    modified = (
        macro.copy()
    )

    modified.loc[
        modified.index
        > as_of_date
    ] = 999999.0

    recalculated = (
        build_macro_features_for_date(
            macro_data=modified,
            calendar_index=
                macro.index,
            as_of_date=
                as_of_date,
        )
    )

    assert np.allclose(
        original.to_numpy(),
        recalculated.to_numpy(),
        equal_nan=True,
    )


def test_macro_feature_matrix():
    macro = (
        build_sample_macro_data()
    )

    dates = pd.DatetimeIndex(
        [
            macro.index[
                250
            ],
            macro.index[
                300
            ],
            macro.index[
                350
            ],
        ]
    )

    matrix = (
        build_macro_feature_matrix(
            macro_data=macro,
            calendar_index=
                macro.index,
            rebalance_dates=
                dates,
        )
    )

    assert len(
        matrix
    ) == 3

    assert (
        matrix.index.name
        == "rebalance_date"
    )
