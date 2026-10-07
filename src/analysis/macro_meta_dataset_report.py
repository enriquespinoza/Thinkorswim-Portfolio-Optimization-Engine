from __future__ import annotations

import pandas as pd

from config.settings import (
    FEATURE_DATA_DIR,
    RAW_DATA_DIR,
)

from src.backtest.walk_forward import (
    run_walk_forward,
)

from src.data.macro_data import (
    download_macro_dataset,
    save_macro_data,
)

from src.features.macro_features import (
    build_macro_feature_matrix,
)

from src.features.meta_dataset import (
    build_meta_learning_dataset,
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

    print()
    print(
        "Downloading macroeconomic data..."
    )

    macro_data = (
        download_macro_dataset(
            start="2014-01-01",
            end="2026-09-30",
        )
    )

    macro_path = (
        save_macro_data(
            macro_data
        )
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

    rebalance_dates = (
        backtest.period_results[
            "rebalance_date"
        ]
        .drop_duplicates()
        .sort_values()
    )

    macro_features = (
        build_macro_feature_matrix(
            macro_data=
                macro_data,

            calendar_index=
                returns.index,

            rebalance_dates=
                rebalance_dates,
        )
    )

    base_dataset = (
        build_meta_learning_dataset(
            returns=returns,
            period_results=
                backtest.period_results,
        )
    )

    combined_dataset = (
        build_meta_learning_dataset(
            returns=returns,
            period_results=
                backtest.period_results,
            macro_features=
                macro_features,
        )
    )

    FEATURE_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        FEATURE_DATA_DIR
        / "meta_learning_dataset_with_macro.csv"
    )

    combined_dataset.to_csv(
        output_path
    )

    print()
    print("=" * 110)
    print(
        "MACRO-ENHANCED META-LEARNING DATASET"
    )
    print("=" * 110)

    print()
    print(
        "Base rows:",
        len(
            base_dataset
        ),
    )

    print(
        "Combined rows:",
        len(
            combined_dataset
        ),
    )

    print(
        "Base columns:",
        len(
            base_dataset.columns
        ),
    )

    print(
        "Macro features:",
        len(
            macro_features.columns
        ),
    )

    print(
        "Combined columns:",
        len(
            combined_dataset.columns
        ),
    )

    print(
        "First observation:",
        combined_dataset
        .index
        .min()
        .date(),
    )

    print(
        "Last observation:",
        combined_dataset
        .index
        .max()
        .date(),
    )

    print()
    print(
        "Raw macro data:",
        macro_path,
    )

    print(
        "Combined dataset:",
        output_path,
    )

    print()
    print(
        "MISSING MACRO VALUES"
    )
    print("-" * 110)

    missing = (
        macro_features
        .isna()
        .sum()
    )

    missing = (
        missing[
            missing > 0
        ]
    )

    if missing.empty:
        print(
            "None"
        )
    else:
        print(
            missing
        )

    print()
    print(
        "LATEST MACRO STATE"
    )
    print("-" * 110)

    print(
        macro_features
        .iloc[-1]
        .sort_index()
        .round(4)
    )

    print()
    print(
        "MACRO FEATURE DISPERSION"
    )
    print("-" * 110)

    summary = pd.DataFrame(
        {
            "mean":
                macro_features.mean(),

            "std":
                macro_features.std(),

            "min":
                macro_features.min(),

            "max":
                macro_features.max(),
        }
    )

    print(
        summary.round(
            4
        )
    )

    print()
    print(
        "ROW PRESERVATION CHECK"
    )
    print("-" * 110)

    if (
        len(base_dataset)
        == 104
        and len(
            combined_dataset
        )
        == 104
    ):
        print(
            "PASS: all 104 existing "
            "meta-learning observations preserved."
        )
    else:
        print(
            "CHECK REQUIRED:",
            f"base={len(base_dataset)}, "
            f"combined={len(combined_dataset)}",
        )


if __name__ == "__main__":
    main()
