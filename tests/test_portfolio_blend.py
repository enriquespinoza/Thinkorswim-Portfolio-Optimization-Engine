import numpy as np
import pandas as pd

from src.backtest.portfolio_blend import (
    build_blended_weight_history,
)


def test_build_half_blend():
    weights = pd.DataFrame(
        {
            "model": [
                "equal_weight",
                "equal_weight",
                "maximum_sharpe",
                "maximum_sharpe",
            ],

            "rebalance_date":
                pd.to_datetime(
                    [
                        "2026-01-31",
                        "2026-01-31",
                        "2026-01-31",
                        "2026-01-31",
                    ]
                ),

            "asset": [
                "AAA",
                "BBB",
                "AAA",
                "BBB",
            ],

            "weight": [
                0.50,
                0.50,
                0.80,
                0.20,
            ],
        }
    )

    result = (
        build_blended_weight_history(
            weights=weights,
            model_a="equal_weight",
            model_b="maximum_sharpe",
            model_b_weight=0.50,
        )
    )

    result = (
        result.set_index(
            "asset"
        )
    )

    assert np.isclose(
        result.loc[
            "AAA",
            "weight",
        ],
        0.65,
    )

    assert np.isclose(
        result.loc[
            "BBB",
            "weight",
        ],
        0.35,
    )


def test_blend_sums_to_one():
    weights = pd.DataFrame(
        {
            "model": [
                "a",
                "a",
                "b",
                "b",
            ],

            "rebalance_date":
                pd.to_datetime(
                    [
                        "2026-01-31",
                        "2026-01-31",
                        "2026-01-31",
                        "2026-01-31",
                    ]
                ),

            "asset": [
                "AAA",
                "BBB",
                "AAA",
                "BBB",
            ],

            "weight": [
                0.60,
                0.40,
                0.20,
                0.80,
            ],
        }
    )

    result = (
        build_blended_weight_history(
            weights,
            "a",
            "b",
            model_b_weight=0.25,
        )
    )

    assert np.isclose(
        result[
            "weight"
        ].sum(),
        1.0,
    )
