from __future__ import annotations

import pandas as pd

from config.settings import (
    FEATURE_DATA_DIR,
    RAW_DATA_DIR,
)

from src.analysis.macro_meta_3m_report import (
    MARKET_FEATURES,
    run_purged_predictions,
)

from src.backtest.metrics import (
    summarize_backtest,
)

from src.backtest.portfolio_blend import (
    simulate_weight_history,
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


STRATEGY_NAME = (
    "market_meta_3m"
)


def build_quarterly_decisions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert monthly eligible predictions into strictly
    non-overlapping quarterly decisions.

    Every third prediction is retained.
    """
    predictions = (
        predictions
        .sort_values(
            "rebalance_date"
        )
        .reset_index(
            drop=True
        )
    )

    quarterly = (
        predictions
        .iloc[
            ::3
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    return quarterly


def build_model_assignments(
    decisions: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    months_per_decision: int = 3,
) -> pd.DataFrame:
    """
    Assign the selected optimizer to each monthly rebalance
    inside its three-month holding window.

    The selected optimizer family remains fixed for three
    months, but that optimizer may generate updated monthly
    target weights using information available at each month.
    """
    rebalance_dates = (
        pd.DatetimeIndex(
            rebalance_dates
        )
        .sort_values()
    )

    records = []

    for _, row in (
        decisions.iterrows()
    ):
        decision_date = pd.Timestamp(
            row[
                "rebalance_date"
            ]
        )

        eligible_dates = (
            rebalance_dates[
                rebalance_dates
                >= decision_date
            ][
                :months_per_decision
            ]
        )

        if (
            len(
                eligible_dates
            )
            < months_per_decision
        ):
            continue

        for rebalance_date in (
            eligible_dates
        ):
            records.append(
                {
                    "decision_date":
                        decision_date,

                    "rebalance_date":
                        rebalance_date,

                    "selected_model":
                        row[
                            "selected_model"
                        ],
                }
            )

    assignments = pd.DataFrame(
        records
    )

    if assignments.empty:
        raise ValueError(
            "No quarterly model assignments generated."
        )

    duplicated = (
        assignments[
            "rebalance_date"
        ]
        .duplicated()
        .any()
    )

    if duplicated:
        raise ValueError(
            "Quarterly holding windows overlap."
        )

    return assignments


def build_selector_weight_history(
    assignments: pd.DataFrame,
    model_weights: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct the actual target-weight history implied by
    quarterly optimizer selection.
    """
    required = {
        "rebalance_date",
        "model",
        "asset",
        "weight",
    }

    missing = (
        required
        - set(
            model_weights.columns
        )
    )

    if missing:
        raise ValueError(
            "Weight history missing columns: "
            f"{sorted(missing)}"
        )

    rows = []

    for _, assignment in (
        assignments.iterrows()
    ):
        date = pd.Timestamp(
            assignment[
                "rebalance_date"
            ]
        )

        model = (
            assignment[
                "selected_model"
            ]
        )

        selected = (
            model_weights[
                (
                    model_weights[
                        "rebalance_date"
                    ]
                    == date
                )
                &
                (
                    model_weights[
                        "model"
                    ]
                    == model
                )
            ]
            .copy()
        )

        if selected.empty:
            raise ValueError(
                f"No weights found for "
                f"{model} on {date.date()}."
            )

        selected[
            "model"
        ] = STRATEGY_NAME

        rows.append(
            selected[
                [
                    "rebalance_date",
                    "model",
                    "asset",
                    "weight",
                ]
            ]
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


def build_static_benchmark_weights(
    model_weights: pd.DataFrame,
    model: str,
    rebalance_dates: pd.Series,
) -> pd.DataFrame:
    """
    Restrict one existing optimizer's target weights to the
    same monthly evaluation window as the selector.
    """
    weights = (
        model_weights[
            (
                model_weights[
                    "model"
                ]
                == model
            )
            &
            (
                model_weights[
                    "rebalance_date"
                ].isin(
                    rebalance_dates
                )
            )
        ]
        .copy()
    )

    return weights


def main() -> None:
    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        220,
    )

    stored_dataset = pd.read_csv(
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

    labels = (
        build_forward_horizon_labels(
            backtest.period_results,
            horizon_periods=3,
        )
    )

    dataset = (
        stored_dataset[
            MARKET_FEATURES
        ]
        .join(
            labels,
            how="inner",
        )
        .sort_index()
    )

    monthly_predictions = (
        run_purged_predictions(
            dataset=dataset,
            feature_set_name=
                "market_only",
            features=
                MARKET_FEATURES,
        )
    )

    decisions = (
        build_quarterly_decisions(
            monthly_predictions
        )
    )

    all_rebalance_dates = (
        backtest.period_results[
            "rebalance_date"
        ]
        .drop_duplicates()
        .sort_values()
    )

    assignments = (
        build_model_assignments(
            decisions=
                decisions,

            rebalance_dates=
                pd.DatetimeIndex(
                    all_rebalance_dates
                ),

            months_per_decision=3,
        )
    )

    selector_weights = (
        build_selector_weight_history(
            assignments=
                assignments,

            model_weights=
                backtest.weights,
        )
    )

    evaluation_dates = (
        assignments[
            "rebalance_date"
        ]
        .drop_duplicates()
        .sort_values()
    )

    period_map = (
        backtest.period_results[
            [
                "rebalance_date",
                "period_end",
            ]
        ]
        .drop_duplicates()
    )

    period_map = (
        period_map[
            period_map[
                "rebalance_date"
            ].isin(
                evaluation_dates
            )
        ]
        .copy()
    )

    selector_results = (
        simulate_weight_history(
            returns=returns,
            target_weights=
                selector_weights,
            period_map=
                period_map,
            transaction_cost_bps=5.0,
        )
    )

    benchmark_models = [
        "equal_weight",
        "maximum_sharpe",
        "minimum_variance",
        "hrp",
        "risk_parity",
        "risk_consensus",
    ]

    result_frames = [
        selector_results
    ]

    for model in (
        benchmark_models
    ):
        weights = (
            build_static_benchmark_weights(
                model_weights=
                    backtest.weights,

                model=model,

                rebalance_dates=
                    evaluation_dates,
            )
        )

        results = (
            simulate_weight_history(
                returns=returns,
                target_weights=
                    weights,
                period_map=
                    period_map,
                transaction_cost_bps=5.0,
            )
        )

        result_frames.append(
            results
        )

    combined_results = pd.concat(
        result_frames,
        ignore_index=True,
    )

    summary = (
        summarize_backtest(
            combined_results
        )
    )

    display = (
        summary.copy()
    )

    for column in [
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "average_turnover",
    ]:
        display[
            column
        ] *= 100

    print()
    print("=" * 125)
    print(
        "NON-OVERLAPPING QUARTERLY META-SELECTOR BACKTEST"
    )
    print("=" * 125)

    print()
    print(
        "Feature set: Market Only"
    )

    print(
        "Decision horizon: 3 months"
    )

    print(
        "Transaction cost: 5 bps"
    )

    print(
        "Quarterly decisions:",
        len(
            decisions
        ),
    )

    print(
        "Monthly portfolio periods:",
        len(
            evaluation_dates
        ),
    )

    print(
        "First decision:",
        decisions[
            "rebalance_date"
        ].min(),
    )

    print(
        "Last decision:",
        decisions[
            "rebalance_date"
        ].max(),
    )

    print()
    print("=" * 125)
    print(
        "PERFORMANCE"
    )
    print("=" * 125)

    print(
        display.round(
            {
                "total_return": 2,
                "annualized_return": 2,
                "annualized_volatility": 2,
                "sharpe": 3,
                "sortino": 3,
                "max_drawdown": 2,
                "average_turnover": 2,
            }
        )
        .sort_values(
            "sharpe",
            ascending=False,
        )
        .to_string()
    )

    print()
    print("=" * 125)
    print(
        "QUARTERLY MODEL DECISIONS"
    )
    print("=" * 125)

    print(
        decisions[
            [
                "rebalance_date",
                "selected_model",
                "actual_best_model",
                "training_labels",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 125)
    print(
        "SELECTION FREQUENCY"
    )
    print("=" * 125)

    frequency = (
        decisions[
            "selected_model"
        ]
        .value_counts()
        .rename_axis(
            "model"
        )
        .to_frame(
            "count"
        )
    )

    frequency[
        "frequency"
    ] = (
        frequency[
            "count"
        ]
        / frequency[
            "count"
        ].sum()
        * 100
    )

    print(
        frequency.round(
            {
                "frequency": 1,
            }
        )
        .to_string()
    )


if __name__ == "__main__":
    main()
