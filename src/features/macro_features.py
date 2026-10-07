from __future__ import annotations

import numpy as np
import pandas as pd


RATE_COLUMNS = (
    "yield_2y",
    "yield_10y",
    "yield_30y",
    "fed_funds",
    "breakeven_10y",
    "real_yield_10y",
)

YIELD_COLUMNS = (
    "yield_2y",
    "yield_10y",
    "yield_30y",
)

REQUIRED_MACRO_COLUMNS = {
    "yield_2y",
    "yield_10y",
    "yield_30y",
    "fed_funds",
    "breakeven_10y",
    "real_yield_10y",
    "unemployment",
    "initial_claims",
    "payrolls",
    "retail_sales",
}


def validate_macro_data(
    macro_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate availability-dated macroeconomic data.
    """
    if not isinstance(
        macro_data,
        pd.DataFrame,
    ):
        raise TypeError(
            "macro_data must be a pandas DataFrame."
        )

    if macro_data.empty:
        raise ValueError(
            "macro_data cannot be empty."
        )

    if not isinstance(
        macro_data.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "macro_data must use a DatetimeIndex."
        )

    missing = (
        REQUIRED_MACRO_COLUMNS
        - set(
            macro_data.columns
        )
    )

    if missing:
        raise ValueError(
            f"Missing macro columns: "
            f"{sorted(missing)}"
        )

    cleaned = (
        macro_data
        .sort_index()
        .copy()
    )

    for column in (
        REQUIRED_MACRO_COLUMNS
    ):
        cleaned[
            column
        ] = pd.to_numeric(
            cleaned[
                column
            ],
            errors="coerce",
        )

    return cleaned


def align_macro_to_calendar(
    macro_data: pd.DataFrame,
    calendar_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Forward-fill availability-dated macro data onto the market
    trading calendar.

    No backfilling is performed.

    Therefore a macro observation cannot appear before its
    information-availability date.
    """
    macro_data = (
        validate_macro_data(
            macro_data
        )
    )

    if not isinstance(
        calendar_index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "calendar_index must be a DatetimeIndex."
        )

    calendar_index = (
        pd.DatetimeIndex(
            calendar_index
        )
        .sort_values()
        .unique()
    )

    combined_index = (
        macro_data.index
        .union(
            calendar_index
        )
        .sort_values()
    )

    aligned = (
        macro_data
        .reindex(
            combined_index
        )
        .ffill()
        .reindex(
            calendar_index
        )
    )

    aligned.index.name = (
        "date"
    )

    return aligned


def _level_change(
    series: pd.Series,
    lookback: int,
) -> float:
    """
    Difference in raw level between today and lookback
    trading observations ago.
    """
    if len(series) <= lookback:
        raise ValueError(
            f"At least {lookback + 1} observations "
            "are required."
        )

    return float(
        series.iloc[-1]
        - series.iloc[
            -lookback - 1
        ]
    )


def _percentage_change(
    series: pd.Series,
    lookback: int,
) -> float:
    """
    Percentage change from lookback observations ago.
    """
    if len(series) <= lookback:
        raise ValueError(
            f"At least {lookback + 1} observations "
            "are required."
        )

    previous = float(
        series.iloc[
            -lookback - 1
        ]
    )

    current = float(
        series.iloc[-1]
    )

    if np.isclose(
        previous,
        0.0,
    ):
        return np.nan

    return (
        current
        / previous
        - 1.0
    )


def _change_volatility_bp(
    series: pd.Series,
    lookback: int,
) -> float:
    """
    Standard deviation of daily level changes expressed
    in basis points.
    """
    if len(series) <= lookback:
        raise ValueError(
            f"At least {lookback + 1} observations "
            "are required."
        )

    changes = (
        series
        .diff()
        .tail(
            lookback
        )
        .dropna()
    )

    if changes.empty:
        return np.nan

    return float(
        changes.std(
            ddof=1
        )
        * 100.0
    )


def build_macro_features_for_date(
    macro_data: pd.DataFrame,
    calendar_index: pd.DatetimeIndex,
    as_of_date: pd.Timestamp | str,
) -> pd.Series:
    """
    Build macro/rates features using only information available
    on or before as_of_date.
    """
    macro_data = (
        validate_macro_data(
            macro_data
        )
    )

    as_of_date = pd.Timestamp(
        as_of_date
    )

    historical_calendar = (
        pd.DatetimeIndex(
            calendar_index
        )
    )

    historical_calendar = (
        historical_calendar[
            historical_calendar
            <= as_of_date
        ]
    )

    if len(
        historical_calendar
    ) < 127:
        raise ValueError(
            "At least 127 trading-calendar observations "
            "are required for macro features."
        )

    aligned = (
        align_macro_to_calendar(
            macro_data,
            historical_calendar,
        )
    )

    history = (
        aligned.loc[
            aligned.index
            <= as_of_date
        ]
    )

    required_latest = history[
        list(
            REQUIRED_MACRO_COLUMNS
        )
    ].iloc[-1]

    if required_latest.isna().any():
        missing = (
            required_latest[
                required_latest.isna()
            ]
            .index
            .tolist()
        )

        raise ValueError(
            f"Macro values unavailable as of "
            f"{as_of_date.date()}: "
            f"{missing}"
        )

    latest = (
        history.iloc[-1]
    )

    features: dict[
        str,
        float,
    ] = {}

    # Current rates / macro levels.
    for column in [
        "yield_2y",
        "yield_10y",
        "yield_30y",
        "fed_funds",
        "breakeven_10y",
        "real_yield_10y",
        "unemployment",
    ]:
        features[
            f"{column}_level"
        ] = float(
            latest[
                column
            ]
        )

    # Yield-curve / policy structure.
    features[
        "curve_10y_2y_bp"
    ] = float(
        (
            latest[
                "yield_10y"
            ]
            - latest[
                "yield_2y"
            ]
        )
        * 100.0
    )

    features[
        "curve_30y_10y_bp"
    ] = float(
        (
            latest[
                "yield_30y"
            ]
            - latest[
                "yield_10y"
            ]
        )
        * 100.0
    )

    features[
        "policy_10y_ff_bp"
    ] = float(
        (
            latest[
                "yield_10y"
            ]
            - latest[
                "fed_funds"
            ]
        )
        * 100.0
    )

    # Rate momentum.
    for column in RATE_COLUMNS:
        series = (
            history[
                column
            ]
        )

        for lookback in [
            21,
            63,
        ]:
            features[
                f"{column}_change_{lookback}d_bp"
            ] = (
                _level_change(
                    series,
                    lookback,
                )
                * 100.0
            )

    # Treasury-rate volatility.
    for column in YIELD_COLUMNS:
        series = (
            history[
                column
            ]
        )

        for lookback in [
            21,
            63,
        ]:
            features[
                f"{column}_change_std_{lookback}d_bp"
            ] = (
                _change_volatility_bp(
                    series,
                    lookback,
                )
            )

    # Labor-market trends.
    unemployment = history[
        "unemployment"
    ]

    features[
        "unemployment_change_63d_pp"
    ] = _level_change(
        unemployment,
        63,
    )

    features[
        "unemployment_change_126d_pp"
    ] = _level_change(
        unemployment,
        126,
    )

    claims = history[
        "initial_claims"
    ]

    features[
        "initial_claims_change_21d_pct"
    ] = _percentage_change(
        claims,
        21,
    )

    features[
        "initial_claims_change_63d_pct"
    ] = _percentage_change(
        claims,
        63,
    )

    payrolls = history[
        "payrolls"
    ]

    features[
        "payrolls_change_63d_pct"
    ] = _percentage_change(
        payrolls,
        63,
    )

    features[
        "payrolls_change_126d_pct"
    ] = _percentage_change(
        payrolls,
        126,
    )

    retail_sales = history[
        "retail_sales"
    ]

    features[
        "retail_sales_change_63d_pct"
    ] = _percentage_change(
        retail_sales,
        63,
    )

    features[
        "retail_sales_change_126d_pct"
    ] = _percentage_change(
        retail_sales,
        126,
    )

    feature_series = pd.Series(
        features,
        dtype=float,
        name=as_of_date,
    )

    if not np.isfinite(
        feature_series
        .dropna()
        .to_numpy()
    ).all():
        raise ValueError(
            "Macro feature vector contains "
            "non-finite values."
        )

    return feature_series


def build_macro_feature_matrix(
    macro_data: pd.DataFrame,
    calendar_index: pd.DatetimeIndex,
    rebalance_dates: pd.DatetimeIndex | list,
) -> pd.DataFrame:
    """
    Build one macro/rates feature row for each rebalance date.
    """
    rows = []

    for date in pd.to_datetime(
        rebalance_dates
    ):
        features = (
            build_macro_features_for_date(
                macro_data=
                    macro_data,
                calendar_index=
                    calendar_index,
                as_of_date=
                    date,
            )
        )

        features.name = date

        rows.append(
            features
        )

    if not rows:
        raise ValueError(
            "No macro feature rows were generated."
        )

    matrix = pd.DataFrame(
        rows
    )

    matrix.index.name = (
        "rebalance_date"
    )

    return (
        matrix
        .sort_index()
    )
