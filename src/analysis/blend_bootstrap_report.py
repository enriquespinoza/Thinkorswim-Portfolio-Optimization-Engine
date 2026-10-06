from __future__ import annotations

import pandas as pd

from config.settings import RAW_DATA_DIR

from src.backtest.bootstrap import (
    bootstrap_strategy_comparison,
)

from src.backtest.portfolio_blend import (
    build_blended_weight_history,
    simulate_weight_history,
)

from src.backtest.walk_forward import (
    run_walk_forward,
)

from src.portfolio.returns import (
    build_return_matrix,
)


def main() -> None:
    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        220,
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

    period_map = (
        backtest.period_results[
            [
                "rebalance_date",
                "period_end",
            ]
        ]
        .drop_duplicates()
    )

    equal_weight_returns = (
        backtest.period_results[
            backtest.period_results[
                "model"
            ]
            == "equal_weight"
        ]
        .set_index(
            "period_end"
        )[
            "net_return"
        ]
        .sort_index()
    )

    blend_results = {}

    for alpha in [
        0.25,
        0.50,
        0.75,
    ]:
        strategy_name = (
            f"EW_{1-alpha:.0%}_MS_{alpha:.0%}"
        )

        weights = (
            build_blended_weight_history(
                weights=
                    backtest.weights,

                model_a=
                    "equal_weight",

                model_b=
                    "maximum_sharpe",

                model_b_weight=
                    alpha,

                strategy_name=
                    strategy_name,
            )
        )

        simulated = (
            simulate_weight_history(
                returns=returns,

                target_weights=
                    weights,

                period_map=
                    period_map,

                transaction_cost_bps=
                    5.0,
            )
        )

        blend_results[
            strategy_name
        ] = (
            simulated
            .set_index(
                "period_end"
            )[
                "net_return"
            ]
            .sort_index()
        )

    records = []

    for (
        strategy_name,
        strategy_returns,
    ) in blend_results.items():

        for block_length in [
            3,
            6,
            12,
        ]:
            result = (
                bootstrap_strategy_comparison(
                    challenger=
                        strategy_returns,

                    benchmark=
                        equal_weight_returns,

                    challenger_name=
                        strategy_name,

                    benchmark_name=
                        "equal_weight",

                    n_bootstrap=5000,

                    block_length=
                        block_length,

                    periods_per_year=12,

                    random_seed=
                        42
                        + block_length,
                )
            )

            summary = (
                result.summary
            )

            records.append(
                {
                    "strategy":
                        strategy_name,

                    "block_months":
                        block_length,

                    "mean_sharpe_diff":
                        summary[
                            "mean_sharpe_difference"
                        ],

                    "sharpe_ci_low":
                        summary[
                            "sharpe_difference_ci_low"
                        ],

                    "sharpe_ci_high":
                        summary[
                            "sharpe_difference_ci_high"
                        ],

                    "prob_sharpe_better":
                        summary[
                            "probability_sharpe_improvement"
                        ],

                    "mean_return_diff":
                        summary[
                            "mean_return_difference"
                        ],

                    "return_ci_low":
                        summary[
                            "return_difference_ci_low"
                        ],

                    "return_ci_high":
                        summary[
                            "return_difference_ci_high"
                        ],

                    "prob_return_better":
                        summary[
                            "probability_return_improvement"
                        ],

                    "mean_drawdown_improvement":
                        summary[
                            "mean_drawdown_improvement"
                        ],

                    "prob_drawdown_better":
                        summary[
                            "probability_drawdown_improvement"
                        ],
                }
            )

    report = pd.DataFrame(
        records
    )

    display = (
        report.copy()
    )

    for column in [
        "prob_sharpe_better",
        "prob_return_better",
        "prob_drawdown_better",
    ]:
        display[
            column
        ] *= 100

    for column in [
        "mean_return_diff",
        "return_ci_low",
        "return_ci_high",
        "mean_drawdown_improvement",
    ]:
        display[
            column
        ] *= 100

    print()
    print("=" * 130)
    print(
        "BLEND BLOCK-BOOTSTRAP ROBUSTNESS"
    )
    print("=" * 130)

    print()
    print(
        "Benchmark: Equal Weight"
    )

    print(
        "Transaction costs: 5 bps"
    )

    print(
        "Bootstrap samples: 5,000 per comparison"
    )

    print()
    print(
        display.round(
            {
                "mean_sharpe_diff": 4,
                "sharpe_ci_low": 4,
                "sharpe_ci_high": 4,
                "prob_sharpe_better": 1,
                "mean_return_diff": 3,
                "return_ci_low": 3,
                "return_ci_high": 3,
                "prob_return_better": 1,
                "mean_drawdown_improvement": 3,
                "prob_drawdown_better": 1,
            }
        )
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
