from __future__ import annotations

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
)

from src.analysis.quarterly_meta_robustness import (
    summarize_one_phase,
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


PHASES = [0, 1, 2]

TRANSACTION_COSTS_BPS = [
    0.0,
    5.0,
    10.0,
    25.0,
]


def main() -> None:
    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        260,
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

    labels = (
        build_forward_horizon_labels(
            backtest.period_results,
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

    # summarize_one_phase also reports exact winner accuracy,
    # so attach the hindsight best-model label.
    binary = (
        binary.merge(
            dataset[
                [
                    "best_model"
                ]
            ],
            left_on=
                "rebalance_date",
            right_index=True,
            how="left",
        )
        .rename(
            columns={
                "best_model":
                    "actual_best_model"
            }
        )
    )

    records = []

    for phase in PHASES:
        for cost in TRANSACTION_COSTS_BPS:

            record, _ = (
                summarize_one_phase(
                    returns=
                        returns,

                    backtest=
                        backtest,

                    monthly_predictions=
                        binary,

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
        "selector_annualized_return",
        "selector_volatility",
        "selector_max_drawdown",
        "selector_turnover",
        "equal_weight_return",
        "maximum_sharpe_return",
        "return_advantage_vs_ew",
        "return_advantage_vs_ms",
    ]

    for column in percentage_columns:
        display[
            column
        ] *= 100

    print()
    print("=" * 165)
    print(
        "BINARY META-ALLOCATION V1 "
        "PHASE / COST ROBUSTNESS"
    )
    print("=" * 165)

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
                "selector_annualized_return": 2,
                "selector_volatility": 2,
                "selector_sharpe": 3,
                "selector_sortino": 3,
                "selector_max_drawdown": 2,
                "selector_turnover": 2,
                "equal_weight_return": 2,
                "equal_weight_sharpe": 3,
                "maximum_sharpe_return": 2,
                "maximum_sharpe_sharpe": 3,
                "return_advantage_vs_ew": 2,
                "return_advantage_vs_ms": 2,
                "sharpe_advantage_vs_ew": 3,
                "sharpe_advantage_vs_ms": 3,
            }
        )
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 165)
    print(
        "5-BP BINARY GATE SUMMARY"
    )
    print("=" * 165)

    five_bp = (
        display[
            display[
                "cost_bps"
            ]
            == 5.0
        ]
        .copy()
    )

    print(
        five_bp[
            [
                "phase",
                "decisions",
                "selector_annualized_return",
                "selector_sharpe",
                "selector_max_drawdown",
                "selector_turnover",
                "return_advantage_vs_ew",
                "sharpe_advantage_vs_ew",
                "return_advantage_vs_ms",
                "sharpe_advantage_vs_ms",
            ]
        ]
        .round(
            {
                "selector_annualized_return": 2,
                "selector_sharpe": 3,
                "selector_max_drawdown": 2,
                "selector_turnover": 2,
                "return_advantage_vs_ew": 2,
                "sharpe_advantage_vs_ew": 3,
                "return_advantage_vs_ms": 2,
                "sharpe_advantage_vs_ms": 3,
            }
        )
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
