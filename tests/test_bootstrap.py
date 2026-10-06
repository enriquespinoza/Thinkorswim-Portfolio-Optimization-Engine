import numpy as np
import pandas as pd

from src.backtest.bootstrap import (
    bootstrap_strategy_comparison,
    circular_block_bootstrap_indices,
    validate_paired_returns,
)


def test_paired_returns():
    challenger = pd.Series(
        [0.01] * 20
    )

    benchmark = pd.Series(
        [0.005] * 20
    )

    a, b = validate_paired_returns(
        challenger,
        benchmark,
    )

    assert len(a) == 20
    assert len(b) == 20


def test_block_indices_length():
    rng = np.random.default_rng(
        42
    )

    indices = (
        circular_block_bootstrap_indices(
            number_of_observations=100,
            block_length=6,
            rng=rng,
        )
    )

    assert len(
        indices
    ) == 100

    assert (
        indices >= 0
    ).all()

    assert (
        indices < 100
    ).all()


def test_bootstrap_comparison():
    rng = np.random.default_rng(
        42
    )

    benchmark = pd.Series(
        rng.normal(
            0.005,
            0.02,
            60,
        )
    )

    challenger = (
        benchmark
        + 0.002
    )

    result = (
        bootstrap_strategy_comparison(
            challenger=
                challenger,

            benchmark=
                benchmark,

            n_bootstrap=250,

            block_length=3,
        )
    )

    assert len(
        result.samples
    ) == 250

    assert (
        "probability_sharpe_improvement"
        in result.summary.index
    )

    assert (
        result.summary[
            "mean_return_difference"
        ]
        > 0
    )
