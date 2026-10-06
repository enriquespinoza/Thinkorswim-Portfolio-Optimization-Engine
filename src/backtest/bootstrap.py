from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.metrics import (
    calculate_annualized_return,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
)


@dataclass(frozen=True)
class BootstrapComparisonResult:
    """
    Bootstrap comparison between a challenger strategy
    and a benchmark strategy.
    """

    samples: pd.DataFrame
    summary: pd.Series


def validate_paired_returns(
    challenger: pd.Series,
    benchmark: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Align and validate two return series.
    """
    if not isinstance(
        challenger,
        pd.Series,
    ):
        raise TypeError(
            "challenger must be a pandas Series."
        )

    if not isinstance(
        benchmark,
        pd.Series,
    ):
        raise TypeError(
            "benchmark must be a pandas Series."
        )

    aligned = pd.concat(
        [
            challenger.rename(
                "challenger"
            ),
            benchmark.rename(
                "benchmark"
            ),
        ],
        axis=1,
        join="inner",
    ).dropna()

    if aligned.empty:
        raise ValueError(
            "No overlapping return observations."
        )

    if len(aligned) < 12:
        raise ValueError(
            "At least 12 paired observations are required."
        )

    values = aligned.to_numpy()

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "Return series contain non-finite values."
        )

    return (
        aligned[
            "challenger"
        ],
        aligned[
            "benchmark"
        ],
    )


def circular_block_bootstrap_indices(
    number_of_observations: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate one circular block-bootstrap sample.

    Circular sampling allows blocks to wrap from the end
    of the series back to the beginning.
    """
    if number_of_observations <= 0:
        raise ValueError(
            "number_of_observations must be positive."
        )

    if block_length <= 0:
        raise ValueError(
            "block_length must be positive."
        )

    if block_length > number_of_observations:
        raise ValueError(
            "block_length cannot exceed sample size."
        )

    number_of_blocks = int(
        np.ceil(
            number_of_observations
            / block_length
        )
    )

    indices = []

    for _ in range(
        number_of_blocks
    ):
        start = int(
            rng.integers(
                0,
                number_of_observations,
            )
        )

        block = (
            start
            + np.arange(
                block_length
            )
        ) % number_of_observations

        indices.extend(
            block.tolist()
        )

    return np.asarray(
        indices[
            :number_of_observations
        ],
        dtype=int,
    )


def bootstrap_strategy_comparison(
    challenger: pd.Series,
    benchmark: pd.Series,
    challenger_name: str = "challenger",
    benchmark_name: str = "benchmark",
    n_bootstrap: int = 5000,
    block_length: int = 6,
    periods_per_year: int = 12,
    random_seed: int = 42,
) -> BootstrapComparisonResult:
    """
    Paired circular block-bootstrap comparison.

    For every bootstrap sample, both strategies use the
    exact same resampled months.

    This preserves the paired nature of the comparison.
    """
    challenger, benchmark = (
        validate_paired_returns(
            challenger,
            benchmark,
        )
    )

    if n_bootstrap <= 0:
        raise ValueError(
            "n_bootstrap must be positive."
        )

    if periods_per_year <= 0:
        raise ValueError(
            "periods_per_year must be positive."
        )

    number_of_observations = len(
        challenger
    )

    rng = np.random.default_rng(
        random_seed
    )

    challenger_array = (
        challenger.to_numpy()
    )

    benchmark_array = (
        benchmark.to_numpy()
    )

    records = []

    for iteration in range(
        n_bootstrap
    ):
        indices = (
            circular_block_bootstrap_indices(
                number_of_observations=
                    number_of_observations,

                block_length=
                    block_length,

                rng=rng,
            )
        )

        challenger_sample = pd.Series(
            challenger_array[
                indices
            ]
        )

        benchmark_sample = pd.Series(
            benchmark_array[
                indices
            ]
        )

        challenger_return = (
            calculate_annualized_return(
                challenger_sample,
                periods_per_year=
                    periods_per_year,
            )
        )

        benchmark_return = (
            calculate_annualized_return(
                benchmark_sample,
                periods_per_year=
                    periods_per_year,
            )
        )

        challenger_sharpe = (
            calculate_sharpe_ratio(
                challenger_sample,
                risk_free_rate=0.0,
                periods_per_year=
                    periods_per_year,
            )
        )

        benchmark_sharpe = (
            calculate_sharpe_ratio(
                benchmark_sample,
                risk_free_rate=0.0,
                periods_per_year=
                    periods_per_year,
            )
        )

        challenger_drawdown = (
            calculate_max_drawdown(
                challenger_sample
            )
        )

        benchmark_drawdown = (
            calculate_max_drawdown(
                benchmark_sample
            )
        )

        records.append(
            {
                "iteration":
                    iteration,

                "challenger_annualized_return":
                    challenger_return,

                "benchmark_annualized_return":
                    benchmark_return,

                "annualized_return_difference":
                    challenger_return
                    - benchmark_return,

                "challenger_sharpe":
                    challenger_sharpe,

                "benchmark_sharpe":
                    benchmark_sharpe,

                "sharpe_difference":
                    challenger_sharpe
                    - benchmark_sharpe,

                "challenger_max_drawdown":
                    challenger_drawdown,

                "benchmark_max_drawdown":
                    benchmark_drawdown,

                # Positive means challenger had a
                # shallower drawdown.
                "drawdown_improvement":
                    challenger_drawdown
                    - benchmark_drawdown,
            }
        )

    samples = pd.DataFrame(
        records
    )

    sharpe_difference = samples[
        "sharpe_difference"
    ]

    return_difference = samples[
        "annualized_return_difference"
    ]

    drawdown_difference = samples[
        "drawdown_improvement"
    ]

    summary = pd.Series(
        {
            "observations":
                number_of_observations,

            "bootstrap_samples":
                n_bootstrap,

            "block_length":
                block_length,

            "mean_sharpe_difference":
                sharpe_difference.mean(),

            "median_sharpe_difference":
                sharpe_difference.median(),

            "sharpe_difference_ci_low":
                sharpe_difference.quantile(
                    0.025
                ),

            "sharpe_difference_ci_high":
                sharpe_difference.quantile(
                    0.975
                ),

            "probability_sharpe_improvement":
                (
                    sharpe_difference
                    > 0
                ).mean(),

            "mean_return_difference":
                return_difference.mean(),

            "return_difference_ci_low":
                return_difference.quantile(
                    0.025
                ),

            "return_difference_ci_high":
                return_difference.quantile(
                    0.975
                ),

            "probability_return_improvement":
                (
                    return_difference
                    > 0
                ).mean(),

            "mean_drawdown_improvement":
                drawdown_difference.mean(),

            "probability_drawdown_improvement":
                (
                    drawdown_difference
                    > 0
                ).mean(),

            "challenger":
                challenger_name,

            "benchmark":
                benchmark_name,
        }
    )

    return BootstrapComparisonResult(
        samples=samples,
        summary=summary,
    )
