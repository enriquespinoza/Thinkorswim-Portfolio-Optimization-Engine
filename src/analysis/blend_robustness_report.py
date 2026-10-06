from __future__ import annotations

import pandas as pd

from config.settings import RAW_DATA_DIR

from src.backtest.metrics import (
    summarize_backtest,
)

from src.backtest.portfolio_blend import (
    build_blended_weight_history,
    simulate_weight_history,
)

from src.backtest.regime_analysis import (
    analyze_custom_regimes,
    rank_models_by_regime,
)

from src.backtest.walk_forward import (
    run_walk_forward,
)

from src.portfolio.returns import (
    build_return_matrix,
)


BLEND_WEIGHTS = [
    0.25,
    0.50,
    0.75,
]

TRANSACTION_COSTS_BPS = [
    0.0,
    5.0,
    10.0,
    25.0,
]

REGIMES = {
    "2018-2019 Pre-COVID": (
        "2018-01-01",
        "2019-12-31",
    ),

    "2020 Shock + Recovery": (
        "2020-01-01",
        "2020-12-31",
    ),

    "2021 Risk-On": (
        "2021-01-01",
        "2021-12-31",
    ),

    "2022 Rate Shock": (
        "2022-01-01",
        "2022-12-31",
    ),

    "2023-2024 Recovery": (
        "2023-01-01",
        "2024-12-31",
    ),

    "2025-2026": (
        "2025-01-01",
        "2026-09-30",
    ),
}


def build_strategy_results(
    returns: pd.DataFrame,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    """
    Build equal-weight, maximum-Sharpe and fixed blend results
    using the same transaction-cost assumption.
    """
    backtest = run_walk_forward(
        returns=returns,
        min_train_observations=756,
        max_weight=0.40,
        risk_free_rate=0.0,
        ewma_span=126,
        transaction_cost_bps=
            transaction_cost_bps,
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

    base_results = (
        backtest.period_results[
            backtest.period_results[
                "model"
            ].isin(
                [
                    "equal_weight",
                    "maximum_sharpe",
                ]
            )
        ]
        .copy()
    )

    strategy_results = [
        base_results
    ]

    for alpha in BLEND_WEIGHTS:
        strategy_name = (
            f"EW_{1-alpha:.0%}_MS_{alpha:.0%}"
        )

        blend_weights = (
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

        blend_results = (
            simulate_weight_history(
                returns=returns,

                target_weights=
                    blend_weights,

                period_map=
                    period_map,

                transaction_cost_bps=
                    transaction_cost_bps,
            )
        )

        strategy_results.append(
            blend_results
        )

    return pd.concat(
        strategy_results,
        ignore_index=True,
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

    cost_records = []

    five_bp_results = None

    for cost_bps in (
        TRANSACTION_COSTS_BPS
    ):
        results = (
            build_strategy_results(
                returns=returns,
                transaction_cost_bps=
                    cost_bps,
            )
        )

        if cost_bps == 5.0:
            five_bp_results = (
                results.copy()
            )

        summary = (
            summarize_backtest(
                results
            )
        )

        for (
            model,
            row,
        ) in summary.iterrows():
            cost_records.append(
                {
                    "transaction_cost_bps":
                        cost_bps,

                    "model":
                        model,

                    "annualized_return":
                        row[
                            "annualized_return"
                        ],

                    "annualized_volatility":
                        row[
                            "annualized_volatility"
                        ],

                    "sharpe":
                        row[
                            "sharpe"
                        ],

                    "sortino":
                        row[
                            "sortino"
                        ],

                    "max_drawdown":
                        row[
                            "max_drawdown"
                        ],

                    "average_turnover":
                        row[
                            "average_turnover"
                        ],
                }
            )

    cost_table = pd.DataFrame(
        cost_records
    )

    print()
    print("=" * 115)
    print(
        "BLEND ROBUSTNESS ANALYSIS"
    )
    print("=" * 115)

    print()
    print(
        "TRANSACTION-COST SENSITIVITY"
    )
    print("-" * 115)

    display = (
        cost_table.copy()
    )

    for column in [
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "average_turnover",
    ]:
        display[
            column
        ] *= 100

    print(
        display.round(
            {
                "annualized_return": 2,
                "annualized_volatility": 2,
                "sharpe": 3,
                "sortino": 3,
                "max_drawdown": 2,
                "average_turnover": 2,
            }
        )
        .sort_values(
            [
                "transaction_cost_bps",
                "sharpe",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .to_string(
            index=False
        )
    )

    if five_bp_results is None:
        return

    regime_results = (
        analyze_custom_regimes(
            period_results=
                five_bp_results,
            regimes=REGIMES,
        )
    )

    ranked = (
        rank_models_by_regime(
            regime_results,
            metric="sharpe",
        )
    )

    regime_display = (
        ranked.copy()
    )

    for column in [
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
    ]:
        regime_display[
            column
        ] *= 100

    print()
    print("=" * 115)
    print(
        "5-BP REGIME PERFORMANCE"
    )
    print("=" * 115)

    for regime in REGIMES:
        print()
        print(
            regime.upper()
        )
        print("-" * 115)

        table = (
            regime_display[
                regime_display[
                    "regime"
                ]
                == regime
            ]
            .sort_values(
                "rank"
            )
            [
                [
                    "rank",
                    "model",
                    "annualized_return",
                    "annualized_volatility",
                    "sharpe",
                    "sortino",
                    "max_drawdown",
                ]
            ]
        )

        print(
            table.round(
                {
                    "annualized_return": 2,
                    "annualized_volatility": 2,
                    "sharpe": 3,
                    "sortino": 3,
                    "max_drawdown": 2,
                }
            )
            .to_string(
                index=False
            )
        )

    print()
    print("=" * 115)
    print(
        "BEST STRATEGY BY REGIME"
    )
    print("=" * 115)

    winners = (
        ranked[
            ranked[
                "rank"
            ] == 1
        ]
        [
            [
                "regime",
                "model",
                "sharpe",
                "annualized_return",
                "max_drawdown",
            ]
        ]
        .copy()
    )

    winners[
        "annualized_return"
    ] *= 100

    winners[
        "max_drawdown"
    ] *= 100

    print(
        winners.round(
            {
                "sharpe": 3,
                "annualized_return": 2,
                "max_drawdown": 2,
            }
        )
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
