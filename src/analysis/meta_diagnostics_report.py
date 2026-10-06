from __future__ import annotations

import pandas as pd

from config.settings import RAW_DATA_DIR

from src.backtest.walk_forward import (
    run_walk_forward,
)

from src.features.meta_dataset import (
    build_meta_learning_dataset,
)

from src.models.meta_diagnostics import (
    prediction_calibration_table,
    prediction_information_coefficients,
    selection_confusion_table,
    sign_diagnostics,
    strategy_benchmark_table,
)

from src.models.meta_regression import (
    DEFAULT_CANDIDATE_MODELS,
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

    meta = (
        run_meta_regression(
            dataset=dataset,
            feature_columns=
                DEFAULT_META_FEATURES,
            candidate_models=
                DEFAULT_CANDIDATE_MODELS,
            min_train_periods=36,
            alpha=10.0,
            selection_threshold=0.0,
        )
    )

    predictions = (
        meta.predictions
    )

    print()
    print("=" * 110)
    print(
        "META-MODEL SIGNAL DIAGNOSTICS"
    )
    print("=" * 110)

    print()
    print(
        "INFORMATION COEFFICIENTS"
    )
    print("-" * 110)

    ic = (
        prediction_information_coefficients(
            predictions,
            DEFAULT_CANDIDATE_MODELS,
        )
    )

    print(
        ic.round(3)
    )

    print()
    print(
        "PREDICTION SIGN DIAGNOSTICS"
    )
    print("-" * 110)

    signs = sign_diagnostics(
        predictions,
        DEFAULT_CANDIDATE_MODELS,
    )

    display_signs = (
        signs.copy()
    )

    display_signs[
        "mean_realized_excess"
    ] *= 100

    display_signs[
        "median_realized_excess"
    ] *= 100

    display_signs[
        "positive_realized_rate"
    ] *= 100

    print(
        display_signs.round(3)
        .to_string(index=False)
    )

    print()
    print(
        "PREDICTION QUARTILE CALIBRATION"
    )
    print("-" * 110)

    calibration = (
        prediction_calibration_table(
            predictions,
            DEFAULT_CANDIDATE_MODELS,
            bins=4,
        )
    )

    display_calibration = (
        calibration.copy()
    )

    for column in [
        "mean_predicted_excess",
        "mean_realized_excess",
        "positive_realized_rate",
    ]:
        display_calibration[
            column
        ] *= 100

    print(
        display_calibration.round(3)
        .to_string(index=False)
    )

    print()
    print(
        "SAME-WINDOW STRATEGY BENCHMARKS"
    )
    print("-" * 110)

    benchmarks = (
        strategy_benchmark_table(
            predictions,
            DEFAULT_CANDIDATE_MODELS,
        )
    )

    display_benchmarks = (
        benchmarks.copy()
    )

    for column in [
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
    ]:
        display_benchmarks[
            column
        ] *= 100

    print(
        display_benchmarks.round(
            {
                "total_return": 2,
                "annualized_return": 2,
                "annualized_volatility": 2,
                "sharpe": 3,
                "sortino": 3,
                "max_drawdown": 2,
            }
        )
    )

    print()
    print(
        "SELECTION CONFUSION MATRIX"
    )
    print("-" * 110)

    print(
        selection_confusion_table(
            predictions
        )
    )


if __name__ == "__main__":
    main()
