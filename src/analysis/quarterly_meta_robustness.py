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

from src.analysis.quarterly_meta_backtest import (
    build_model_assignments,
    build_selector_weight_history,
    build_static_benchmark_weights,
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


TRANSACTION_COSTS_BPS = [
    0.0,
    5.0,
    10.0,
    25.0,
]

PHASE_OFFSETS = [
    0,
    1,
    2,
]


def build_phase_decisions(
    predictions: pd.DataFrame,
    phase_offset: int,
) -> pd.DataFrame:
    """
    Build one of the three possible non-overlapping
    quarterly decision schedules.

    phase_offset=0:
        prediction rows 0, 3, 6, ...

    phase_offset=1:
        prediction rows 1, 4, 7, ...

    phase_offset=2:
        prediction rows 2, 5, 8, ...
    """
    if phase_offset not in {
        0,
        1,
        2,
    }:
        raise ValueError(
            "phase_offset must be 0, 1, or 2."
        )

    predictions = (
        predictions
        .sort_values(
            "rebalance_date"
        )
        .reset_index(
            drop=True
        )
    )

    decisions = (
        predictions
        .iloc[
            phase_offset::3
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    return decisions


def summarize_one_phase(
    returns: pd.DataFrame,
    backtest,
    monthly_predictions: pd.DataFrame,
    phase_offset: int,
    transaction_cost_bps: float,
) -> tuple[
    dict,
    pd.DataFrame,
]:
    """
    Simulate one quarterly phase under one transaction-cost
    assumption.
    """
    decisions = (
        build_phase_decisions(
            monthly_predictions,
            phase_offset=
                phase_offset,
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

    # A trailing quarterly decision may be removed if the
    # full three-month holding window is unavailable.
    valid_decision_dates = (
        assignments[
            "decision_date"
        ]
        .drop_duplicates()
    )

    decisions = (
        decisions[
            decisions[
                "rebalance_date"
            ].isin(
                valid_decision_dates
            )
        ]
        .copy()
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
            transaction_cost_bps=
                transaction_cost_bps,
        )
    )

    equal_weight_weights = (
        build_static_benchmark_weights(
            model_weights=
                backtest.weights,

            model=
                "equal_weight",

            rebalance_dates=
                evaluation_dates,
        )
    )

    maximum_sharpe_weights = (
        build_static_benchmark_weights(
            model_weights=
                backtest.weights,

            model=
                "maximum_sharpe",

            rebalance_dates=
                evaluation_dates,
        )
    )

    equal_weight_results = (
        simulate_weight_history(
            returns=returns,
            target_weights=
                equal_weight_weights,
            period_map=
                period_map,
            transaction_cost_bps=
                transaction_cost_bps,
        )
    )

    maximum_sharpe_results = (
        simulate_weight_history(
            returns=returns,
            target_weights=
                maximum_sharpe_weights,
            period_map=
                period_map,
            transaction_cost_bps=
                transaction_cost_bps,
        )
    )

    combined = pd.concat(
        [
            selector_results,
            equal_weight_results,
            maximum_sharpe_results,
        ],
        ignore_index=True,
    )

    summary = (
        summarize_backtest(
            combined
        )
    )

    selector = (
        summary.loc[
            "market_meta_3m"
        ]
    )

    equal_weight = (
        summary.loc[
            "equal_weight"
        ]
    )

    maximum_sharpe = (
        summary.loc[
            "maximum_sharpe"
        ]
    )

    exact_accuracy = (
        decisions[
            "selected_model"
        ]
        == decisions[
            "actual_best_model"
        ]
    ).mean()

    frequency = (
        decisions[
            "selected_model"
        ]
        .value_counts(
            normalize=True
        )
    )

    record = {
        "phase":
            phase_offset,

        "cost_bps":
            transaction_cost_bps,

        "decisions":
            len(
                decisions
            ),

        "monthly_periods":
            len(
                evaluation_dates
            ),

        "first_decision":
            decisions[
                "rebalance_date"
            ].min(),

        "last_decision":
            decisions[
                "rebalance_date"
            ].max(),

        "exact_selection_accuracy":
            exact_accuracy,

        "equal_weight_frequency":
            frequency.get(
                "equal_weight",
                0.0,
            ),

        "maximum_sharpe_frequency":
            frequency.get(
                "maximum_sharpe",
                0.0,
            ),

        "selector_total_return":
            selector[
                "total_return"
            ],

        "selector_annualized_return":
            selector[
                "annualized_return"
            ],

        "selector_volatility":
            selector[
                "annualized_volatility"
            ],

        "selector_sharpe":
            selector[
                "sharpe"
            ],

        "selector_sortino":
            selector[
                "sortino"
            ],

        "selector_max_drawdown":
            selector[
                "max_drawdown"
            ],

        "selector_turnover":
            selector[
                "average_turnover"
            ],

        "equal_weight_return":
            equal_weight[
                "annualized_return"
            ],

        "equal_weight_sharpe":
            equal_weight[
                "sharpe"
            ],

        "maximum_sharpe_return":
            maximum_sharpe[
                "annualized_return"
            ],

        "maximum_sharpe_sharpe":
            maximum_sharpe[
                "sharpe"
            ],

        "sharpe_advantage_vs_ew":
            (
                selector[
                    "sharpe"
                ]
                - equal_weight[
                    "sharpe"
                ]
            ),

        "sharpe_advantage_vs_ms":
            (
                selector[
                    "sharpe"
                ]
                - maximum_sharpe[
                    "sharpe"
                ]
            ),

        "return_advantage_vs_ew":
            (
                selector[
                    "annualized_return"
                ]
                - equal_weight[
                    "annualized_return"
                ]
            ),

        "return_advantage_vs_ms":
            (
                selector[
                    "annualized_return"
                ]
                - maximum_sharpe[
                    "annualized_return"
                ]
            ),
    }

    return (
        record,
        decisions,
    )


def main() -> None:
    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        260,
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

    # Keep the research selector trained using the original
    # 5-bp assumptions. Cost sensitivity below changes
    # execution friction without re-tuning the model.
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
            dataset=
                dataset,

            feature_set_name=
                "market_only",

            features=
                MARKET_FEATURES,
        )
    )

    records = []

    for phase in (
        PHASE_OFFSETS
    ):
        for cost in (
            TRANSACTION_COSTS_BPS
        ):
            record, _ = (
                summarize_one_phase(
                    returns=
                        returns,

                    backtest=
                        backtest,

                    monthly_predictions=
                        monthly_predictions,

                    phase_offset=
                        phase,

                    transaction_cost_bps=
                        cost,
                )
            )

            records.append(
                record
            )

    report = pd.DataFrame(
        records
    )

    display = (
        report.copy()
    )

    percentage_columns = [
        "exact_selection_accuracy",
        "equal_weight_frequency",
        "maximum_sharpe_frequency",
        "selector_total_return",
        "selector_annualized_return",
        "selector_volatility",
        "selector_max_drawdown",
        "selector_turnover",
        "equal_weight_return",
        "maximum_sharpe_return",
        "return_advantage_vs_ew",
        "return_advantage_vs_ms",
    ]

    for column in (
        percentage_columns
    ):
        display[
            column
        ] *= 100

    print()
    print("=" * 160)
    print(
        "QUARTERLY META-SELECTOR "
        "PHASE AND COST ROBUSTNESS"
    )
    print("=" * 160)

    print()
    print(
        "Selector training assumption: "
        "5 bps"
    )

    print(
        "Execution cost sensitivity: "
        "0 / 5 / 10 / 25 bps"
    )

    print()
    print(
        display[
            [
                "phase",
                "cost_bps",
                "decisions",
                "monthly_periods",
                "selector_annualized_return",
                "selector_volatility",
                "selector_sharpe",
                "selector_sortino",
                "selector_max_drawdown",
                "selector_turnover",
                "equal_weight_return",
                "equal_weight_sharpe",
                "maximum_sharpe_return",
                "maximum_sharpe_sharpe",
                "return_advantage_vs_ew",
                "return_advantage_vs_ms",
                "sharpe_advantage_vs_ew",
                "sharpe_advantage_vs_ms",
            ]
        ]
        .round(
            {
                "selector_annualized_return":
                    2,

                "selector_volatility":
                    2,

                "selector_sharpe":
                    3,

                "selector_sortino":
                    3,

                "selector_max_drawdown":
                    2,

                "selector_turnover":
                    2,

                "equal_weight_return":
                    2,

                "equal_weight_sharpe":
                    3,

                "maximum_sharpe_return":
                    2,

                "maximum_sharpe_sharpe":
                    3,

                "return_advantage_vs_ew":
                    2,

                "return_advantage_vs_ms":
                    2,

                "sharpe_advantage_vs_ew":
                    3,

                "sharpe_advantage_vs_ms":
                    3,
            }
        )
        .to_string(
            index=False
        )
    )

    five_bp = (
        display[
            display[
                "cost_bps"
            ]
            == 5.0
        ]
        .copy()
    )

    print()
    print("=" * 160)
    print(
        "5-BP PHASE SUMMARY"
    )
    print("=" * 160)

    print(
        five_bp[
            [
                "phase",
                "first_decision",
                "last_decision",
                "decisions",
                "exact_selection_accuracy",
                "equal_weight_frequency",
                "maximum_sharpe_frequency",
                "selector_annualized_return",
                "selector_sharpe",
                "return_advantage_vs_ew",
                "sharpe_advantage_vs_ew",
                "return_advantage_vs_ms",
                "sharpe_advantage_vs_ms",
            ]
        ]
        .round(
            {
                "exact_selection_accuracy":
                    1,

                "equal_weight_frequency":
                    1,

                "maximum_sharpe_frequency":
                    1,

                "selector_annualized_return":
                    2,

                "selector_sharpe":
                    3,

                "return_advantage_vs_ew":
                    2,

                "sharpe_advantage_vs_ew":
                    3,

                "return_advantage_vs_ms":
                    2,

                "sharpe_advantage_vs_ms":
                    3,
            }
        )
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
