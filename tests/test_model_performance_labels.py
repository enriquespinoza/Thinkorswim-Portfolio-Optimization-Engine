import numpy as np
import pandas as pd

from src.features.model_performance_labels import (
    build_forward_model_labels,
    build_forward_return_matrix,
    build_model_excess_returns,
    build_model_ranks,
    calculate_winner_margin,
)


def sample_period_results():
    return pd.DataFrame(
        {
            "model": [
                "equal_weight",
                "model_a",
                "model_b",
                "equal_weight",
                "model_a",
                "model_b",
            ],

            "rebalance_date":
                pd.to_datetime(
                    [
                        "2026-01-31",
                        "2026-01-31",
                        "2026-01-31",
                        "2026-02-28",
                        "2026-02-28",
                        "2026-02-28",
                    ]
                ),

            "period_end":
                pd.to_datetime(
                    [
                        "2026-02-28",
                        "2026-02-28",
                        "2026-02-28",
                        "2026-03-31",
                        "2026-03-31",
                        "2026-03-31",
                    ]
                ),

            "net_return": [
                0.02,
                0.03,
                0.01,
                0.01,
                -0.01,
                0.04,
            ],
        }
    )


def test_forward_return_matrix():
    matrix = (
        build_forward_return_matrix(
            sample_period_results()
        )
    )

    assert matrix.shape == (
        2,
        3,
    )

    assert np.isclose(
        matrix.loc[
            pd.Timestamp(
                "2026-01-31"
            ),
            "model_a",
        ],
        0.03,
    )


def test_model_excess_returns():
    matrix = (
        build_forward_return_matrix(
            sample_period_results()
        )
    )

    excess = (
        build_model_excess_returns(
            matrix
        )
    )

    assert np.isclose(
        excess.iloc[
            0
        ][
            "model_a_excess_vs_equal_weight"
        ],
        0.01,
    )


def test_model_ranks():
    matrix = (
        build_forward_return_matrix(
            sample_period_results()
        )
    )

    ranks = build_model_ranks(
        matrix
    )

    assert (
        ranks.iloc[
            0
        ][
            "model_a_rank"
        ]
        == 1
    )


def test_winner_margin():
    matrix = (
        build_forward_return_matrix(
            sample_period_results()
        )
    )

    margin = (
        calculate_winner_margin(
            matrix
        )
    )

    assert np.isclose(
        margin.iloc[0],
        0.01,
    )


def test_forward_labels():
    labels = (
        build_forward_model_labels(
            sample_period_results()
        )
    )

    assert (
        labels.iloc[
            0
        ][
            "best_model"
        ]
        == "model_a"
    )

    assert (
        labels.iloc[
            1
        ][
            "best_model"
        ]
        == "model_b"
    )

    assert (
        "winner_margin"
        in labels.columns
    )

    assert (
        "model_return_dispersion"
        in labels.columns
    )
