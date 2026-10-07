import pandas as pd
import pytest

from src.features.meta_dataset import (
    join_macro_features_to_meta_dataset,
)


def test_macro_join_preserves_rows():
    dates = pd.date_range(
        "2026-01-31",
        periods=3,
        freq="ME",
    )

    meta = pd.DataFrame(
        {
            "market_feature":
                [
                    1.0,
                    2.0,
                    3.0,
                ],

            "best_model":
                [
                    "a",
                    "b",
                    "a",
                ],
        },
        index=dates,
    )

    macro = pd.DataFrame(
        {
            "macro_feature":
                [
                    10.0,
                    20.0,
                    30.0,
                ],
        },
        index=dates,
    )

    joined = (
        join_macro_features_to_meta_dataset(
            meta,
            macro,
        )
    )

    assert len(
        joined
    ) == 3

    assert (
        "macro_feature"
        in joined.columns
    )


def test_macro_join_rejects_missing_date():
    dates = pd.date_range(
        "2026-01-31",
        periods=3,
        freq="ME",
    )

    meta = pd.DataFrame(
        {
            "market_feature":
                [
                    1.0,
                    2.0,
                    3.0,
                ],
        },
        index=dates,
    )

    macro = pd.DataFrame(
        {
            "macro_feature":
                [
                    10.0,
                    20.0,
                ],
        },
        index=dates[
            :2
        ],
    )

    with pytest.raises(
        ValueError
    ):
        join_macro_features_to_meta_dataset(
            meta,
            macro,
        )
