import numpy as np
import pandas as pd

from src.models.meta_regression import (
    build_ridge_pipeline,
    evaluate_model_predictions,
    evaluate_model_selection,
    walk_forward_meta_regression,
)


def build_sample_dataset():
    np.random.seed(
        42
    )

    dates = pd.date_range(
        "2020-01-31",
        periods=60,
        freq="ME",
    )

    feature_a = np.random.normal(
        size=60
    )

    feature_b = np.random.normal(
        size=60
    )

    equal_return = np.random.normal(
        0.01,
        0.03,
        size=60,
    )

    model_excess = (
        0.002 * feature_a
        - 0.001 * feature_b
        + np.random.normal(
            0,
            0.002,
            size=60,
        )
    )

    model_return = (
        equal_return
        + model_excess
    )

    best_model = np.where(
        model_return
        > equal_return,
        "model_a",
        "equal_weight",
    )

    return pd.DataFrame(
        {
            "feature_a":
                feature_a,

            "feature_b":
                feature_b,

            "equal_weight_forward_return":
                equal_return,

            "model_a_forward_return":
                model_return,

            "model_a_excess_vs_equal_weight":
                model_excess,

            "best_model":
                best_model,
        },
        index=dates,
    )


def test_ridge_pipeline():
    pipeline = (
        build_ridge_pipeline(
            alpha=10.0
        )
    )

    assert pipeline is not None


def test_walk_forward_predictions():
    dataset = (
        build_sample_dataset()
    )

    predictions = (
        walk_forward_meta_regression(
            dataset=dataset,

            feature_columns=[
                "feature_a",
                "feature_b",
            ],

            candidate_models=[
                "model_a",
            ],

            min_train_periods=24,
            alpha=1.0,
        )
    )

    assert len(
        predictions
    ) == 36

    assert (
        "model_a_predicted_excess"
        in predictions.columns
    )

    assert (
        "model_a_realized_excess"
        in predictions.columns
    )


def test_prediction_metrics():
    dataset = (
        build_sample_dataset()
    )

    predictions = (
        walk_forward_meta_regression(
            dataset=dataset,

            feature_columns=[
                "feature_a",
                "feature_b",
            ],

            candidate_models=[
                "model_a",
            ],

            min_train_periods=24,
            alpha=1.0,
        )
    )

    metrics = (
        evaluate_model_predictions(
            predictions,
            candidate_models=[
                "model_a",
            ],
        )
    )

    assert (
        "correlation"
        in metrics.columns
    )

    assert (
        "directional_accuracy"
        in metrics.columns
    )


def test_selection_metrics():
    dataset = (
        build_sample_dataset()
    )

    predictions = (
        walk_forward_meta_regression(
            dataset=dataset,

            feature_columns=[
                "feature_a",
                "feature_b",
            ],

            candidate_models=[
                "model_a",
            ],

            min_train_periods=24,
            alpha=1.0,
        )
    )

    metrics = (
        evaluate_model_selection(
            predictions
        )
    )

    assert (
        "selection_accuracy"
        in metrics.index
    )

    assert (
        "average_selected_excess"
        in metrics.index
    )
