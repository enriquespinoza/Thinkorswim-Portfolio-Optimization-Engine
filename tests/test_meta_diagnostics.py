import numpy as np
import pandas as pd

from src.models.meta_diagnostics import (
    prediction_calibration_table,
    prediction_information_coefficients,
    selection_confusion_table,
    sign_diagnostics,
    strategy_benchmark_table,
)


def sample_predictions():
    dates = pd.date_range(
        "2024-01-31",
        periods=12,
        freq="ME",
    )

    equal_returns = np.array(
        [
            0.01,
            0.02,
            -0.01,
            0.03,
            0.01,
            -0.02,
            0.02,
            0.01,
            0.00,
            0.03,
            -0.01,
            0.02,
        ]
    )

    predicted = np.linspace(
        -0.01,
        0.01,
        12,
    )

    realized = (
        predicted
        + np.linspace(
            -0.002,
            0.002,
            12,
        )
    )

    return pd.DataFrame(
        {
            "equal_weight_forward_return":
                equal_returns,

            "model_a_predicted_excess":
                predicted,

            "model_a_realized_excess":
                realized,

            "selected_forward_return":
                equal_returns + np.maximum(
                    realized,
                    0,
                ),

            "selected_model":
                np.where(
                    predicted > 0,
                    "model_a",
                    "equal_weight",
                ),

            "actual_best_model":
                np.where(
                    realized > 0,
                    "model_a",
                    "equal_weight",
                ),
        },
        index=dates,
    )


def test_information_coefficients():
    result = (
        prediction_information_coefficients(
            sample_predictions(),
            ["model_a"],
        )
    )

    assert (
        result.loc[
            "model_a",
            "pearson_ic",
        ]
        > 0
    )


def test_calibration_table():
    result = (
        prediction_calibration_table(
            sample_predictions(),
            ["model_a"],
            bins=4,
        )
    )

    assert len(
        result
    ) == 4


def test_sign_diagnostics():
    result = (
        sign_diagnostics(
            sample_predictions(),
            ["model_a"],
        )
    )

    assert set(
        result[
            "signal"
        ]
    ) == {
        "predicted_positive",
        "predicted_nonpositive",
    }


def test_strategy_benchmarks():
    result = (
        strategy_benchmark_table(
            sample_predictions(),
            ["model_a"],
        )
    )

    assert (
        "equal_weight"
        in result.index
    )

    assert (
        "oracle_best"
        in result.index
    )


def test_confusion_table():
    result = (
        selection_confusion_table(
            sample_predictions()
        )
    )

    assert (
        "All"
        in result.index
    )
