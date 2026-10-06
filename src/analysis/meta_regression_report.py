from __future__ import annotations

import pandas as pd

from config.settings import RAW_DATA_DIR

from src.backtest.walk_forward import (
    run_walk_forward,
)

from src.features.meta_dataset import (
    build_meta_learning_dataset,
)

from src.models.meta_regression import (
    DEFAULT_META_FEATURES,
    run_meta_regression,
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

    dataset = (
        build_meta_learning_dataset(
            returns=returns,
            period_results=
                backtest.period_results,
        )
    )

    result = (
        run_meta_regression(
            dataset=dataset,

            feature_columns=
                DEFAULT_META_FEATURES,

            min_train_periods=36,

            alpha=10.0,

            # Zero for the first research run.
            # Later this becomes an implementation hurdle.
            selection_threshold=0.0,
        )
    )

    print()
    print("=" * 105)
    print(
        "META-MODEL RIDGE REGRESSION"
    )
    print("=" * 105)

    print()
    print(
        "Dataset observations:",
        len(dataset),
    )

    print(
        "Features used:",
        len(
            DEFAULT_META_FEATURES
        ),
    )

    print(
        "Initial training periods:",
        36,
    )

    print(
        "Out-of-sample predictions:",
        len(
            result.predictions
        ),
    )

    print()
    print(
        "FEATURES"
    )
    print("-" * 105)

    for feature in (
        DEFAULT_META_FEATURES
    ):
        print(feature)

    print()
    print(
        "MODEL EXCESS-RETURN PREDICTION QUALITY"
    )
    print("-" * 105)

    metrics = (
        result.model_metrics
        .copy()
    )

    metrics[
        "mae"
    ] *= 100

    metrics[
        "rmse"
    ] *= 100

    metrics[
        "directional_accuracy"
    ] *= 100

    print(
        metrics.round(
            {
                "mae": 3,
                "rmse": 3,
                "correlation": 3,
                "directional_accuracy": 2,
            }
        )
    )

    print()
    print(
        "META-SELECTION RESULTS"
    )
    print("-" * 105)

    selection = (
        result.selection_metrics
        .copy()
    )

    percentage_fields = [
        "selection_accuracy",
        "positive_excess_rate",
        "active_positive_rate",
        "average_selected_excess",
        "proxy_total_return",
        "equal_weight_total_return",
        "proxy_excess_growth",
    ]

    for field in (
        percentage_fields
    ):
        selection[
            field
        ] *= 100

    print(
        selection.round(3)
    )

    print()
    print(
        "MODEL SELECTION FREQUENCY"
    )
    print("-" * 105)

    frequencies = (
        result.predictions[
            "selected_model"
        ]
        .value_counts()
    )

    frequencies_pct = (
        result.predictions[
            "selected_model"
        ]
        .value_counts(
            normalize=True
        )
        * 100
    )

    frequency_table = pd.DataFrame(
        {
            "count":
                frequencies,

            "frequency_pct":
                frequencies_pct,
        }
    )

    print(
        frequency_table.round(2)
    )

    print()
    print(
        "LATEST META-MODEL DECISION"
    )
    print("-" * 105)

    latest = (
        result.predictions
        .iloc[-1]
    )

    print(
        "Date:",
        result.predictions.index[
            -1
        ].date(),
    )

    print(
        "Selected model:",
        latest[
            "selected_model"
        ],
    )

    print(
        "Predicted excess return:",
        f"{latest['selected_predicted_excess']:.3%}",
    )

    print(
        "Realized excess return:",
        f"{latest['selected_realized_excess']:.3%}",
    )

    print(
        "Actual best model:",
        latest[
            "actual_best_model"
        ],
    )


if __name__ == "__main__":
    main()
