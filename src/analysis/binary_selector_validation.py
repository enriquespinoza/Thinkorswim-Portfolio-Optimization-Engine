from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import (
    FEATURE_DATA_DIR,
    RAW_DATA_DIR,
)

from src.analysis.binary_meta_3m_report import (
    run_binary_predictions,
)

from src.analysis.macro_meta_3m_report import (
    MARKET_FEATURES,
    run_purged_predictions,
)

from src.backtest.walk_forward import (
    run_walk_forward,
)

from src.features.horizon_labels import (
    build_forward_horizon_labels,
)

from src.portfolio.returns import (
    build_return_matrix,
)


BOOTSTRAP_SAMPLES = 10000
RANDOM_SEED = 42


def build_predictions(
    stored: pd.DataFrame,
    period_results: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    labels = (
        build_forward_horizon_labels(
            period_results,
            horizon_periods=3,
        )
    )

    dataset = (
        stored[
            MARKET_FEATURES
        ]
        .join(
            labels,
            how="inner",
        )
        .sort_index()
    )

    binary = (
        run_binary_predictions(
            dataset
        )
    )

    multimodel = (
        run_purged_predictions(
            dataset=
                dataset,

            feature_set_name=
                "market_only",

            features=
                MARKET_FEATURES,
        )
    )

    return (
        binary,
        multimodel,
    )


def verify_decision_equivalence(
    binary: pd.DataFrame,
    multimodel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare binary decisions with the previous multi-model
    selector on every common prediction date.
    """
    comparison = (
        binary[
            [
                "rebalance_date",
                "selected_model",
            ]
        ]
        .rename(
            columns={
                "selected_model":
                    "binary_model"
            }
        )
        .merge(
            multimodel[
                [
                    "rebalance_date",
                    "selected_model",
                ]
            ]
            .rename(
                columns={
                    "selected_model":
                        "multimodel_model"
                }
            ),
            on="rebalance_date",
            how="inner",
        )
    )

    comparison[
        "match"
    ] = (
        comparison[
            "binary_model"
        ]
        == comparison[
            "multimodel_model"
        ]
    )

    return comparison


def build_nonoverlapping_blocks(
    binary: pd.DataFrame,
    phase: int,
) -> pd.DataFrame:
    """
    Convert overlapping monthly predictions into one of
    three non-overlapping quarterly sequences.
    """
    data = (
        binary
        .sort_values(
            "rebalance_date"
        )
        .reset_index(
            drop=True
        )
        .iloc[
            phase::3
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    data[
        "year"
    ] = (
        pd.to_datetime(
            data[
                "rebalance_date"
            ]
        )
        .dt.year
    )

    return data


def bootstrap_mean_excess(
    excess: np.ndarray,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """
    Paired bootstrap over complete three-month decision
    blocks.

    Each observation is already one non-overlapping
    quarterly decision.
    """
    rng = (
        np.random.default_rng(
            seed
        )
    )

    n = len(
        excess
    )

    means = np.empty(
        samples,
        dtype=float,
    )

    for iteration in range(
        samples
    ):
        sample = rng.choice(
            excess,
            size=n,
            replace=True,
        )

        means[
            iteration
        ] = sample.mean()

    return {
        "observations":
            n,

        "mean_excess":
            float(
                excess.mean()
            ),

        "median_excess":
            float(
                np.median(
                    excess
                )
            ),

        "positive_quarter_rate":
            float(
                (
                    excess > 0
                ).mean()
            ),

        "bootstrap_probability_positive":
            float(
                (
                    means > 0
                ).mean()
            ),

        "ci_low":
            float(
                np.quantile(
                    means,
                    0.025,
                )
            ),

        "ci_high":
            float(
                np.quantile(
                    means,
                    0.975,
                )
            ),
    }


def main() -> None:
    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        220,
    )

    stored = pd.read_csv(
        FEATURE_DATA_DIR
        / "meta_learning_dataset_with_macro.csv",
        index_col=0,
        parse_dates=True,
    )

    prices = pd.read_csv(
        RAW_DATA_DIR
        / "portfolio_universe_daily.csv",
        index_col=0,
        parse_dates=True,
    )

    returns = (
        build_return_matrix(
            prices
        )
        .loc[
            :"2026-09-30"
        ]
    )

    backtest = (
        run_walk_forward(
            returns=returns,
            min_train_observations=756,
            max_weight=0.40,
            risk_free_rate=0.0,
            ewma_span=126,
            transaction_cost_bps=5.0,
        )
    )

    binary, multimodel = (
        build_predictions(
            stored=
                stored,

            period_results=
                backtest.period_results,
        )
    )

    comparison = (
        verify_decision_equivalence(
            binary,
            multimodel,
        )
    )

    print()
    print("=" * 125)
    print(
        "BINARY META-ALLOCATION V1 VALIDATION"
    )
    print("=" * 125)

    print()
    print(
        "DECISION EQUIVALENCE"
    )
    print("-" * 125)

    print(
        "Common observations:",
        len(
            comparison
        ),
    )

    print(
        "Matching decisions:",
        int(
            comparison[
                "match"
            ].sum()
        ),
    )

    print(
        "Match rate (%):",
        round(
            comparison[
                "match"
            ].mean()
            * 100,
            2,
        ),
    )

    mismatches = (
        comparison[
            ~comparison[
                "match"
            ]
        ]
    )

    if mismatches.empty:
        print(
            "PASS: binary gate exactly reproduces "
            "the previous selector."
        )
    else:
        print()
        print(
            "MISMATCHES"
        )

        print(
            mismatches.to_string(
                index=False
            )
        )

    bootstrap_records = []

    for phase in [
        0,
        1,
        2,
    ]:
        quarterly = (
            build_nonoverlapping_blocks(
                binary,
                phase=phase,
            )
        )

        print()
        print("=" * 125)
        print(
            f"PHASE {phase} QUARTERLY SIGNAL ATTRIBUTION"
        )
        print("=" * 125)

        display = (
            quarterly[
                [
                    "rebalance_date",
                    "selected_model",
                    "predicted_excess",
                    "actual_excess",
                    "selected_excess",
                ]
            ]
            .copy()
        )

        for column in [
            "predicted_excess",
            "actual_excess",
            "selected_excess",
        ]:
            display[
                column
            ] *= 100

        print(
            display.round(
                {
                    "predicted_excess":
                        3,

                    "actual_excess":
                        3,

                    "selected_excess":
                        3,
                }
            )
            .to_string(
                index=False
            )
        )

        print()
        print(
            "YEAR ATTRIBUTION"
        )
        print("-" * 125)

        year_summary = (
            quarterly
            .groupby(
                "year"
            )
            .agg(
                decisions=(
                    "selected_excess",
                    "size",
                ),

                maximum_sharpe_calls=(
                    "selected_model",
                    lambda x:
                        (
                            x
                            == "maximum_sharpe"
                        ).sum(),
                ),

                mean_selected_excess=(
                    "selected_excess",
                    "mean",
                ),

                total_selected_excess=(
                    "selected_excess",
                    "sum",
                ),
            )
        )

        year_display = (
            year_summary.copy()
        )

        year_display[
            "mean_selected_excess"
        ] *= 100

        year_display[
            "total_selected_excess"
        ] *= 100

        print(
            year_display.round(
                3
            ).to_string()
        )

        stats = (
            bootstrap_mean_excess(
                quarterly[
                    "selected_excess"
                ].to_numpy(),
                seed=
                    RANDOM_SEED
                    + phase,
            )
        )

        bootstrap_records.append(
            {
                "phase":
                    phase,

                **stats,
            }
        )

    bootstrap = pd.DataFrame(
        bootstrap_records
    )

    display_bootstrap = (
        bootstrap.copy()
    )

    for column in [
        "mean_excess",
        "median_excess",
        "ci_low",
        "ci_high",
    ]:
        display_bootstrap[
            column
        ] *= 100

    for column in [
        "positive_quarter_rate",
        "bootstrap_probability_positive",
    ]:
        display_bootstrap[
            column
        ] *= 100

    print()
    print("=" * 125)
    print(
        "QUARTERLY BLOCK BOOTSTRAP"
    )
    print("=" * 125)

    print(
        display_bootstrap.round(
            {
                "mean_excess": 3,
                "median_excess": 3,
                "positive_quarter_rate": 1,
                "bootstrap_probability_positive": 1,
                "ci_low": 3,
                "ci_high": 3,
            }
        )
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
