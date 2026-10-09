from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import pandas as pd


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from config.settings import (
    PROCESSED_DATA_DIR,
)


DEFAULT_PROCESSED_ROOT = (
    PROCESSED_DATA_DIR
    / "schwab"
)

CAPTURE_RUNS_DIRECTORY = "capture_runs"
VALIDATION_SCHEMA_VERSION = "1.0.0"

DEFAULT_MAX_QUOTE_AGE_MINUTES = 30.0
DEFAULT_WIDE_SPREAD_PCT = 25.0
DEFAULT_EXTREME_PERCENT_CHANGE = 1000.0

OPTION_RESEARCH_FIELDS = (
    "bid",
    "ask",
    "mark",
    "last",
    "volatility",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "openInterest",
    "totalVolume",
    "timeValue",
    "intrinsicValue",
    "extrinsicValue",
    "theoreticalOptionValue",
    "theoreticalVolatility",
)


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    code: str
    dataset: str
    message: str
    count: int | None = None
    percentage: float | None = None


def _canonical_json_bytes(
    value: Any,
) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode(
        "utf-8"
    )


def _sha256_json(
    value: Any,
) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(
            value
        )
    ).hexdigest()


def _read_json(
    path: Path,
    *,
    context: str,
) -> dict[str, Any]:
    try:
        with Path(path).open(
            "r",
            encoding="utf-8",
        ) as handle:
            value = json.load(
                handle
            )

    except FileNotFoundError:
        raise

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{context} contains invalid JSON: {path}"
        ) from exc

    if not isinstance(
        value,
        Mapping,
    ):
        raise RuntimeError(
            f"{context} must contain a JSON object: {path}"
        )

    return dict(
        value
    )


def _latest_capture_summary(
    processed_root: Path,
) -> Path:
    run_dir = (
        Path(
            processed_root
        )
        / CAPTURE_RUNS_DIRECTORY
    )

    if not run_dir.is_dir():
        raise FileNotFoundError(
            "Schwab capture-run directory not found: "
            f"{run_dir}"
        )

    candidates = sorted(
        run_dir.glob(
            "*.json"
        )
    )

    if not candidates:
        raise FileNotFoundError(
            "No Schwab capture-run summaries found in "
            f"{run_dir}"
        )

    return candidates[-1]


def _load_capture_summary(
    *,
    processed_root: Path,
    run_summary: Path | None,
) -> tuple[
    Path,
    dict[str, Any],
]:
    if run_summary is None:
        summary_path = (
            _latest_capture_summary(
                processed_root
            )
        )

    else:
        summary_path = Path(
            run_summary
        )

        if not summary_path.is_absolute():
            project_candidate = (
                PROJECT_ROOT
                / summary_path
            )

            if project_candidate.is_file():
                summary_path = project_candidate

    summary = _read_json(
        summary_path,
        context=
            "Schwab capture-run summary",
    )

    if not str(
        summary.get(
            "run_id",
            "",
        )
    ).strip():
        raise RuntimeError(
            "Capture-run summary has no run_id."
        )

    artifacts = summary.get(
        "artifacts"
    )

    if not isinstance(
        artifacts,
        list,
    ):
        raise RuntimeError(
            "Capture-run summary artifacts must be a list."
        )

    return (
        summary_path,
        summary,
    )


def _artifact_paths_by_dataset(
    *,
    processed_root: Path,
    summary: Mapping[
        str,
        Any,
    ],
) -> dict[
    str,
    list[
        tuple[
            str,
            Path,
        ]
    ],
]:
    grouped: dict[
        str,
        list[
            tuple[
                str,
                Path,
            ]
        ],
    ] = {}

    for index, artifact in enumerate(
        summary.get(
            "artifacts",
            [],
        )
    ):
        if not isinstance(
            artifact,
            Mapping,
        ):
            raise RuntimeError(
                "Capture-run artifact "
                f"{index} is not an object."
            )

        dataset = str(
            artifact.get(
                "dataset",
                "",
            )
        ).strip()

        scope = str(
            artifact.get(
                "scope",
                "",
            )
        ).strip()

        normalized = artifact.get(
            "normalized"
        )

        if not isinstance(
            normalized,
            Mapping,
        ):
            raise RuntimeError(
                "Capture-run artifact "
                f"{index} has no normalized metadata."
            )

        relative_path = str(
            normalized.get(
                "relative_path",
                "",
            )
        ).strip()

        if not dataset:
            raise RuntimeError(
                "Capture-run artifact "
                f"{index} has no dataset."
            )

        if not relative_path:
            raise RuntimeError(
                "Capture-run artifact "
                f"{index} has no normalized relative_path."
            )

        path = (
            Path(
                processed_root
            )
            / relative_path
        )

        if not path.is_file():
            raise FileNotFoundError(
                "Normalized Schwab artifact is missing: "
                f"{path}"
            )

        grouped.setdefault(
            dataset,
            [],
        ).append(
            (
                scope,
                path,
            )
        )

    return grouped


def _load_dataset(
    artifacts: Iterable[
        tuple[
            str,
            Path,
        ]
    ],
) -> pd.DataFrame:
    frames: list[
        pd.DataFrame
    ] = []

    for scope, path in artifacts:
        frame = pd.read_csv(
            path,
            low_memory=False,
        )

        frame[
            "_capture_scope"
        ] = scope

        frames.append(
            frame
        )

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


def _numeric(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(
            index=frame.index,
            dtype="float64",
        )

    return pd.to_numeric(
        frame[
            column
        ],
        errors="coerce",
    )


def _percentage(
    count: int,
    denominator: int,
) -> float:
    if denominator <= 0:
        return 0.0

    return round(
        100.0
        * count
        / denominator,
        4,
    )


def _append_issue(
    issues: list[ValidationIssue],
    *,
    level: str,
    code: str,
    dataset: str,
    message: str,
    count: int | None = None,
    denominator: int | None = None,
) -> None:
    percentage = None

    if (
        count is not None
        and denominator is not None
    ):
        percentage = _percentage(
            count,
            denominator,
        )

    issues.append(
        ValidationIssue(
            level=
                level,

            code=
                code,

            dataset=
                dataset,

            message=
                message,

            count=
                count,

            percentage=
                percentage,
        )
    )


def _quantiles(
    series: pd.Series,
) -> dict[str, float] | None:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return None

    quantile_values = numeric.quantile(
        [
            0.0,
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
            1.0,
        ]
    )

    return {
        "min":
            float(
                quantile_values.loc[
                    0.0
                ]
            ),

        "p01":
            float(
                quantile_values.loc[
                    0.01
                ]
            ),

        "p05":
            float(
                quantile_values.loc[
                    0.05
                ]
            ),

        "p25":
            float(
                quantile_values.loc[
                    0.25
                ]
            ),

        "median":
            float(
                quantile_values.loc[
                    0.50
                ]
            ),

        "p75":
            float(
                quantile_values.loc[
                    0.75
                ]
            ),

        "p95":
            float(
                quantile_values.loc[
                    0.95
                ]
            ),

        "p99":
            float(
                quantile_values.loc[
                    0.99
                ]
            ),

        "max":
            float(
                quantile_values.loc[
                    1.0
                ]
            ),
    }


def _epoch_millis_to_utc(
    series: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    return pd.to_datetime(
        numeric,
        unit="ms",
        errors="coerce",
        utc=True,
    )


def _capture_times(
    frame: pd.DataFrame,
) -> pd.Series:
    if (
        "raw_captured_at_utc"
        not in frame.columns
    ):
        return pd.Series(
            pd.NaT,
            index=frame.index,
            dtype="datetime64[ns, UTC]",
        )

    return pd.to_datetime(
        frame[
            "raw_captured_at_utc"
        ],
        errors="coerce",
        utc=True,
    )


def _quote_age_minutes(
    frame: pd.DataFrame,
    *,
    quote_time_column: str,
) -> pd.Series:
    if quote_time_column not in frame.columns:
        return pd.Series(
            index=frame.index,
            dtype="float64",
        )

    quote_time = (
        _epoch_millis_to_utc(
            frame[
                quote_time_column
            ]
        )
    )

    capture_time = (
        _capture_times(
            frame
        )
    )

    age = (
        capture_time
        - quote_time
    )

    return (
        age.dt.total_seconds()
        / 60.0
    )


def _spot_map(
    quotes: pd.DataFrame,
) -> dict[str, float]:
    if (
        quotes.empty
        or "symbol"
        not in quotes.columns
    ):
        return {}

    price_column = next(
        (
            column
            for column in (
                "quote__mark",
                "quote__lastPrice",
                "regular__regularMarketLastPrice",
                "quote__closePrice",
            )
            if column
            in quotes.columns
        ),
        None,
    )

    if price_column is None:
        return {}

    values = pd.to_numeric(
        quotes[
            price_column
        ],
        errors="coerce",
    )

    result: dict[
        str,
        float,
    ] = {}

    for symbol, value in zip(
        quotes[
            "symbol"
        ],
        values,
    ):
        if (
            pd.notna(
                value
            )
            and float(
                value
            )
            > 0
        ):
            result[
                str(
                    symbol
                ).upper()
            ] = float(
                value
            )

    return result


def _moneyness_bucket(
    signed_pct: float,
) -> str:
    if signed_pct <= -10.0:
        return "deep_otm"

    if signed_pct < -2.0:
        return "otm"

    if signed_pct <= 2.0:
        return "atm"

    if signed_pct < 10.0:
        return "itm"

    return "deep_itm"


def _dte_bucket(
    value: float,
) -> str:
    if value <= 0:
        return "0_dte"

    if value <= 7:
        return "1_7"

    if value <= 30:
        return "8_30"

    if value <= 90:
        return "31_90"

    if value <= 365:
        return "91_365"

    return "366_plus"


def _value_counts_dict(
    series: pd.Series,
) -> dict[str, int]:
    counts = (
        series
        .dropna()
        .astype(str)
        .value_counts()
    )

    return {
        str(
            key
        ):
            int(
                value
            )
        for key, value
        in counts.items()
    }


def _option_coverage(
    options: pd.DataFrame,
    *,
    spots: Mapping[
        str,
        float,
    ],
    spread_pct: pd.Series,
    stale_mask: pd.Series,
) -> dict[
    str,
    dict[
        str,
        Any,
    ]
]:
    if (
        options.empty
        or "underlying_symbol"
        not in options.columns
    ):
        return {}

    result: dict[
        str,
        dict[
            str,
            Any,
        ]
    ] = {}

    symbols = (
        options[
            "underlying_symbol"
        ]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
    )

    for symbol in sorted(
        symbols
    ):
        mask = (
            options[
                "underlying_symbol"
            ]
            .astype(str)
            .str.upper()
            == symbol
        )

        subset = options.loc[
            mask
        ].copy()

        row_count = int(
            len(
                subset
            )
        )

        dte = _numeric(
            subset,
            "expiration_dte",
        )

        dte_buckets = dte.map(
            lambda value:
                _dte_bucket(
                    float(
                        value
                    )
                )
                if pd.notna(
                    value
                )
                else None
        )

        moneyness_counts: dict[
            str,
            int,
        ] = {}

        spot = spots.get(
            symbol
        )

        if (
            spot is not None
            and spot > 0
            and "strike"
            in subset.columns
            and "put_call"
            in subset.columns
        ):
            strike = _numeric(
                subset,
                "strike",
            )

            side = (
                subset[
                    "put_call"
                ]
                .astype(str)
                .str.upper()
            )

            signed = pd.Series(
                index=subset.index,
                dtype="float64",
            )

            call_mask = (
                side
                == "CALL"
            )

            put_mask = (
                side
                == "PUT"
            )

            signed.loc[
                call_mask
            ] = (
                (
                    spot
                    - strike.loc[
                        call_mask
                    ]
                )
                / spot
                * 100.0
            )

            signed.loc[
                put_mask
            ] = (
                (
                    strike.loc[
                        put_mask
                    ]
                    - spot
                )
                / spot
                * 100.0
            )

            moneyness_counts = (
                _value_counts_dict(
                    signed.map(
                        lambda value:
                            _moneyness_bucket(
                                float(
                                    value
                                )
                            )
                            if pd.notna(
                                value
                            )
                            else None
                    )
                )
            )

        zero_bid = (
            _numeric(
                subset,
                "bid",
            )
            == 0
        )

        zero_volume = (
            _numeric(
                subset,
                "totalVolume",
            )
            == 0
        )

        zero_open_interest = (
            _numeric(
                subset,
                "openInterest",
            )
            == 0
        )

        symbol_spread = (
            spread_pct.loc[
                subset.index
            ]
        )

        symbol_stale = (
            stale_mask.loc[
                subset.index
            ]
        )

        result[
            symbol
        ] = {
            "contracts":
                row_count,

            "spot":
                spot,

            "contracts_by_side":
                _value_counts_dict(
                    subset[
                        "put_call"
                    ]
                    if "put_call"
                    in subset.columns
                    else pd.Series(
                        dtype="object"
                    )
                ),

            "expiration_count":
                int(
                    subset[
                        "expiration_date"
                    ].nunique(
                        dropna=True
                    )
                )
                if "expiration_date"
                in subset.columns
                else 0,

            "dte_min":
                float(
                    dte.min()
                )
                if dte.notna().any()
                else None,

            "dte_max":
                float(
                    dte.max()
                )
                if dte.notna().any()
                else None,

            "contracts_by_dte_bucket":
                _value_counts_dict(
                    dte_buckets
                ),

            "contracts_by_moneyness":
                moneyness_counts,

            "zero_bid_count":
                int(
                    zero_bid.sum()
                ),

            "zero_bid_pct":
                _percentage(
                    int(
                        zero_bid.sum()
                    ),
                    row_count,
                ),

            "zero_volume_count":
                int(
                    zero_volume.sum()
                ),

            "zero_volume_pct":
                _percentage(
                    int(
                        zero_volume.sum()
                    ),
                    row_count,
                ),

            "zero_open_interest_count":
                int(
                    zero_open_interest.sum()
                ),

            "zero_open_interest_pct":
                _percentage(
                    int(
                        zero_open_interest.sum()
                    ),
                    row_count,
                ),

            "wide_spread_count":
                int(
                    symbol_spread.notna()
                    .sum()
                ),

            "stale_quote_count":
                int(
                    symbol_stale.sum()
                ),
        }

    return result


def _validate_options(
    options: pd.DataFrame,
    *,
    quotes: pd.DataFrame,
    issues: list[ValidationIssue],
    max_quote_age_minutes: float,
    wide_spread_pct: float,
    extreme_percent_change: float,
) -> dict[str, Any]:
    dataset = "option_chains"

    if options.empty:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "OPTION_DATASET_EMPTY",
            dataset=
                dataset,
            message=
                "Option-chain dataset is empty.",
        )

        return {}

    row_count = int(
        len(
            options
        )
    )

    required = {
        "underlying_symbol",
        "put_call",
        "strike",
        "expiration_dte",
        "bid",
        "ask",
        "mark",
        "volatility",
        "delta",
        "gamma",
        "vega",
    }

    missing = sorted(
        required
        - set(
            options.columns
        )
    )

    if missing:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "OPTION_REQUIRED_COLUMNS_MISSING",
            dataset=
                dataset,
            message=(
                "Option feature inputs are missing "
                f"columns: {missing}"
            ),
        )

    bid = _numeric(
        options,
        "bid",
    )

    ask = _numeric(
        options,
        "ask",
    )

    mark = _numeric(
        options,
        "mark",
    )

    volatility = _numeric(
        options,
        "volatility",
    )

    delta = _numeric(
        options,
        "delta",
    )

    gamma = _numeric(
        options,
        "gamma",
    )

    vega = _numeric(
        options,
        "vega",
    )

    negative_bid_ask = (
        (
            bid < 0
        )
        | (
            ask < 0
        )
    )

    negative_bid_ask_count = int(
        negative_bid_ask.sum()
    )

    if negative_bid_ask_count:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "NEGATIVE_OPTION_BID_ASK",
            dataset=
                dataset,
            message=
                "Negative option bid/ask values detected.",
            count=
                negative_bid_ask_count,
            denominator=
                row_count,
        )

    crossed_market = (
        bid.notna()
        & ask.notna()
        & (
            bid
            > ask
        )
    )

    crossed_market_count = int(
        crossed_market.sum()
    )

    if crossed_market_count:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "CROSSED_OPTION_MARKET",
            dataset=
                dataset,
            message=
                "Option bid exceeds ask.",
            count=
                crossed_market_count,
            denominator=
                row_count,
        )

    mark_outside = (
        bid.notna()
        & ask.notna()
        & mark.notna()
        & (
            (
                mark
                < (
                    bid
                    - 1e-8
                )
            )
            | (
                mark
                > (
                    ask
                    + 1e-8
                )
            )
        )
    )

    mark_outside_count = int(
        mark_outside.sum()
    )

    if mark_outside_count:
        _append_issue(
            issues,
            level="WARNING",
            code=
                "OPTION_MARK_OUTSIDE_BID_ASK",
            dataset=
                dataset,
            message=(
                "Option mark falls outside the quoted "
                "bid/ask interval."
            ),
            count=
                mark_outside_count,
            denominator=
                row_count,
        )

    midpoint = (
        (
            bid
            + ask
        )
        / 2.0
    )

    spread_pct = pd.Series(
        index=options.index,
        dtype="float64",
    )

    positive_midpoint = (
        midpoint
        > 0
    )

    spread_pct.loc[
        positive_midpoint
    ] = (
        (
            ask.loc[
                positive_midpoint
            ]
            - bid.loc[
                positive_midpoint
            ]
        )
        / midpoint.loc[
            positive_midpoint
        ]
        * 100.0
    )

    wide_spread_mask = (
        spread_pct
        > wide_spread_pct
    )

    wide_spread_count = int(
        wide_spread_mask.sum()
    )

    if wide_spread_count:
        _append_issue(
            issues,
            level="WARNING",
            code=
                "WIDE_OPTION_SPREAD",
            dataset=
                dataset,
            message=(
                "Option relative bid/ask spread exceeds "
                f"{wide_spread_pct:.2f}%."
            ),
            count=
                wide_spread_count,
            denominator=
                row_count,
        )

    zero_bid = (
        bid
        == 0
    )

    zero_bid_count = int(
        zero_bid.sum()
    )

    if zero_bid_count:
        _append_issue(
            issues,
            level="WARNING",
            code=
                "ZERO_OPTION_BID",
            dataset=
                dataset,
            message=(
                "Zero-bid option contracts should not "
                "be treated as equally liquid inputs."
            ),
            count=
                zero_bid_count,
            denominator=
                row_count,
        )

    nonpositive_iv = (
        volatility.notna()
        & (
            volatility
            <= 0
        )
    )

    nonpositive_iv_count = int(
        nonpositive_iv.sum()
    )

    if nonpositive_iv_count:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "NONPOSITIVE_OPTION_IV",
            dataset=
                dataset,
            message=
                "Non-positive option volatility detected.",
            count=
                nonpositive_iv_count,
            denominator=
                row_count,
        )

    invalid_delta = (
        delta.notna()
        & (
            (
                delta
                < -1.000001
            )
            | (
                delta
                > 1.000001
            )
        )
    )

    invalid_delta_count = int(
        invalid_delta.sum()
    )

    if invalid_delta_count:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "OPTION_DELTA_OUT_OF_RANGE",
            dataset=
                dataset,
            message=
                "Option delta falls outside [-1, 1].",
            count=
                invalid_delta_count,
            denominator=
                row_count,
        )

    negative_gamma = (
        gamma.notna()
        & (
            gamma
            < 0
        )
    )

    negative_gamma_count = int(
        negative_gamma.sum()
    )

    if negative_gamma_count:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "NEGATIVE_OPTION_GAMMA",
            dataset=
                dataset,
            message=
                "Negative gamma detected for long vanilla options.",
            count=
                negative_gamma_count,
            denominator=
                row_count,
        )

    negative_vega = (
        vega.notna()
        & (
            vega
            < 0
        )
    )

    negative_vega_count = int(
        negative_vega.sum()
    )

    if negative_vega_count:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "NEGATIVE_OPTION_VEGA",
            dataset=
                dataset,
            message=
                "Negative vega detected for long vanilla options.",
            count=
                negative_vega_count,
            denominator=
                row_count,
        )

    dte = _numeric(
        options,
        "expiration_dte",
    )

    zero_dte = (
        dte
        == 0
    )

    zero_dte_count = int(
        zero_dte.sum()
    )

    if zero_dte_count:
        _append_issue(
            issues,
            level="WARNING",
            code=
                "ZERO_DTE_OPTIONS_PRESENT",
            dataset=
                dataset,
            message=(
                "0-DTE contracts are present and should "
                "be modeled separately or explicitly filtered."
            ),
            count=
                zero_dte_count,
            denominator=
                row_count,
        )

    for column in (
        "percentChange",
        "markPercentChange",
    ):
        if column not in options.columns:
            continue

        values = _numeric(
            options,
            column,
        )

        extreme = (
            values.notna()
            & (
                values.abs()
                > extreme_percent_change
            )
        )

        extreme_count = int(
            extreme.sum()
        )

        if extreme_count:
            _append_issue(
                issues,
                level="WARNING",
                code=
                    "EXTREME_OPTION_PERCENT_CHANGE",
                dataset=
                    dataset,
                message=(
                    f"{column} exceeds +/-"
                    f"{extreme_percent_change:.0f}% and "
                    "should be clipped, transformed, or excluded."
                ),
                count=
                    extreme_count,
                denominator=
                    row_count,
            )

    for column in (
        "intrinsicValue",
        "extrinsicValue",
    ):
        if column not in options.columns:
            continue

        values = _numeric(
            options,
            column,
        )

        negative = (
            values.notna()
            & (
                values
                < 0
            )
        )

        negative_count = int(
            negative.sum()
        )

        if negative_count:
            _append_issue(
                issues,
                level="WARNING",
                code=
                    "NEGATIVE_OPTION_VALUE_COMPONENT",
                dataset=
                    dataset,
                message=(
                    f"{column} contains negative values; "
                    "do not use it without semantic review."
                ),
                count=
                    negative_count,
                denominator=
                    row_count,
            )

    constant_fields: list[
        str
    ] = []

    for field in OPTION_RESEARCH_FIELDS:
        if field not in options.columns:
            continue

        values = options[
            field
        ].dropna()

        if (
            not values.empty
            and int(
                values.nunique(
                    dropna=True
                )
            )
            <= 1
        ):
            constant_fields.append(
                field
            )

    if constant_fields:
        _append_issue(
            issues,
            level="WARNING",
            code=
                "CONSTANT_OPTION_RESEARCH_FIELD",
            dataset=
                dataset,
            message=(
                "Constant option fields have no "
                "cross-sectional information in this capture: "
                f"{constant_fields}"
            ),
            count=
                len(
                    constant_fields
                ),
        )

    quote_age = (
        _quote_age_minutes(
            options,
            quote_time_column=
                "quoteTimeInLong",
        )
    )

    stale_mask = (
        quote_age.notna()
        & (
            quote_age
            > max_quote_age_minutes
        )
    )

    stale_count = int(
        stale_mask.sum()
    )

    if stale_count:
        _append_issue(
            issues,
            level="WARNING",
            code=
                "STALE_OPTION_QUOTE",
            dataset=
                dataset,
            message=(
                "Option quote age exceeds "
                f"{max_quote_age_minutes:.1f} minutes "
                "at capture time."
            ),
            count=
                stale_count,
            denominator=
                row_count,
        )

    future_quote = (
        quote_age.notna()
        & (
            quote_age
            < -5.0
        )
    )

    future_quote_count = int(
        future_quote.sum()
    )

    if future_quote_count:
        _append_issue(
            issues,
            level="WARNING",
            code=
                "OPTION_QUOTE_AFTER_CAPTURE",
            dataset=
                dataset,
            message=(
                "Option quote timestamp is more than "
                "five minutes after its capture timestamp."
            ),
            count=
                future_quote_count,
            denominator=
                row_count,
        )

    diagnostics: dict[
        str,
        Any,
    ] = {}

    for field in (
        "volatility",
        "delta",
        "gamma",
        "theta",
        "vega",
        "rho",
        "openInterest",
        "totalVolume",
        "intrinsicValue",
        "extrinsicValue",
        "theoreticalOptionValue",
        "theoreticalVolatility",
    ):
        if field in options.columns:
            diagnostics[
                field
            ] = _quantiles(
                options[
                    field
                ]
            )

    diagnostics[
        "relative_spread_pct"
    ] = _quantiles(
        spread_pct
    )

    diagnostics[
        "quote_age_minutes"
    ] = _quantiles(
        quote_age
    )

    spots = _spot_map(
        quotes
    )

    coverage = (
        _option_coverage(
            options,
            spots=
                spots,
            spread_pct=
                wide_spread_mask.astype(
                    float
                ),
            stale_mask=
                stale_mask,
        )
    )

    for symbol, row in coverage.items():
        mask = (
            options[
                "underlying_symbol"
            ]
            .astype(str)
            .str.upper()
            == symbol
        )

        symbol_spreads = (
            wide_spread_mask.loc[
                mask
            ]
        )

        row[
            "wide_spread_count"
        ] = int(
            symbol_spreads.sum()
        )

        row[
            "wide_spread_pct"
        ] = _percentage(
            int(
                symbol_spreads.sum()
            ),
            int(
                mask.sum()
            ),
        )

        symbol_stale = (
            stale_mask.loc[
                mask
            ]
        )

        row[
            "stale_quote_count"
        ] = int(
            symbol_stale.sum()
        )

        row[
            "stale_quote_pct"
        ] = _percentage(
            int(
                symbol_stale.sum()
            ),
            int(
                mask.sum()
            ),
        )

    return {
        "rows":
            row_count,

        "constant_research_fields":
            constant_fields,

        "zero_bid_count":
            zero_bid_count,

        "zero_bid_pct":
            _percentage(
                zero_bid_count,
                row_count,
            ),

        "wide_spread_count":
            wide_spread_count,

        "wide_spread_pct":
            _percentage(
                wide_spread_count,
                row_count,
            ),

        "stale_quote_count":
            stale_count,

        "stale_quote_pct":
            _percentage(
                stale_count,
                row_count,
            ),

        "zero_dte_count":
            zero_dte_count,

        "zero_dte_pct":
            _percentage(
                zero_dte_count,
                row_count,
            ),

        "diagnostics":
            diagnostics,

        "coverage_by_symbol":
            coverage,

        "spot_prices":
            spots,
    }


def _validate_quotes(
    quotes: pd.DataFrame,
    *,
    issues: list[ValidationIssue],
    max_quote_age_minutes: float,
) -> dict[str, Any]:
    dataset = "quotes"

    if quotes.empty:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "QUOTE_DATASET_EMPTY",
            dataset=
                dataset,
            message=
                "Quote dataset is empty.",
        )

        return {}

    row_count = int(
        len(
            quotes
        )
    )

    bid = _numeric(
        quotes,
        "quote__bidPrice",
    )

    ask = _numeric(
        quotes,
        "quote__askPrice",
    )

    mark = _numeric(
        quotes,
        "quote__mark",
    )

    crossed = (
        bid.notna()
        & ask.notna()
        & (
            bid
            > ask
        )
    )

    crossed_count = int(
        crossed.sum()
    )

    if crossed_count:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "CROSSED_UNDERLYING_QUOTE",
            dataset=
                dataset,
            message=
                "Underlying quote bid exceeds ask.",
            count=
                crossed_count,
            denominator=
                row_count,
        )

    nonpositive_mark = (
        mark.notna()
        & (
            mark
            <= 0
        )
    )

    nonpositive_mark_count = int(
        nonpositive_mark.sum()
    )

    if nonpositive_mark_count:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "NONPOSITIVE_UNDERLYING_MARK",
            dataset=
                dataset,
            message=
                "Underlying quote mark is non-positive.",
            count=
                nonpositive_mark_count,
            denominator=
                row_count,
        )

    quote_age = (
        _quote_age_minutes(
            quotes,
            quote_time_column=
                "quote__quoteTime",
        )
    )

    stale = (
        quote_age.notna()
        & (
            quote_age
            > max_quote_age_minutes
        )
    )

    stale_count = int(
        stale.sum()
    )

    if stale_count:
        _append_issue(
            issues,
            level="WARNING",
            code=
                "STALE_UNDERLYING_QUOTE",
            dataset=
                dataset,
            message=(
                "Underlying quote age exceeds "
                f"{max_quote_age_minutes:.1f} minutes "
                "at capture time."
            ),
            count=
                stale_count,
            denominator=
                row_count,
        )

    return {
        "rows":
            row_count,

        "symbols":
            sorted(
                quotes[
                    "symbol"
                ]
                .dropna()
                .astype(str)
                .str.upper()
                .unique()
                .tolist()
            )
            if "symbol"
            in quotes.columns
            else [],

        "spot_prices":
            _spot_map(
                quotes
            ),

        "quote_age_minutes":
            _quantiles(
                quote_age
            ),
    }


def _validate_price_history(
    price_history: pd.DataFrame,
    *,
    issues: list[ValidationIssue],
) -> dict[str, Any]:
    dataset = "price_history"

    if price_history.empty:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "PRICE_HISTORY_EMPTY",
            dataset=
                dataset,
            message=
                "Price-history dataset is empty.",
        )

        return {}

    row_count = int(
        len(
            price_history
        )
    )

    if (
        "symbol"
        not in price_history.columns
        or "datetime_utc"
        not in price_history.columns
    ):
        _append_issue(
            issues,
            level="ERROR",
            code=
                "PRICE_HISTORY_REQUIRED_COLUMNS_MISSING",
            dataset=
                dataset,
            message=(
                "Price history requires symbol and "
                "datetime_utc for feature coverage checks."
            ),
        )

        return {
            "rows":
                row_count,
        }

    timestamp = pd.to_datetime(
        price_history[
            "datetime_utc"
        ],
        errors="coerce",
        utc=True,
    )

    summary_by_symbol: dict[
        str,
        dict[
            str,
            Any,
        ]
    ] = {}

    for symbol in sorted(
        price_history[
            "symbol"
        ]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
    ):
        mask = (
            price_history[
                "symbol"
            ]
            .astype(str)
            .str.upper()
            == symbol
        )

        subset_time = (
            timestamp.loc[
                mask
            ]
            .dropna()
        )

        volume = _numeric(
            price_history.loc[
                mask
            ],
            "volume",
        )

        rows = int(
            mask.sum()
        )

        zero_volume = int(
            (
                volume
                == 0
            ).sum()
        )

        summary_by_symbol[
            symbol
        ] = {
            "rows":
                rows,

            "start_utc":
                subset_time.min().isoformat()
                if not subset_time.empty
                else None,

            "end_utc":
                subset_time.max().isoformat()
                if not subset_time.empty
                else None,

            "zero_volume_count":
                zero_volume,

            "zero_volume_pct":
                _percentage(
                    zero_volume,
                    rows,
                ),
        }

    return {
        "rows":
            row_count,

        "symbols":
            summary_by_symbol,

        "start_utc":
            timestamp.min().isoformat()
            if timestamp.notna().any()
            else None,

        "end_utc":
            timestamp.max().isoformat()
            if timestamp.notna().any()
            else None,
    }


def validate_schwab_feature_inputs(
    *,
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
    run_summary: Path | None = None,
    max_quote_age_minutes: float = DEFAULT_MAX_QUOTE_AGE_MINUTES,
    wide_spread_pct: float = DEFAULT_WIDE_SPREAD_PCT,
    extreme_percent_change: float = DEFAULT_EXTREME_PERCENT_CHANGE,
) -> dict[str, Any]:
    """
    Perform semantic-quality checks on normalized Schwab
    market data before feature engineering.

    The validation is read-only. It operates only on the
    immutable artifacts referenced by one capture-run
    summary and does not call Schwab APIs.
    """
    if max_quote_age_minutes < 0:
        raise ValueError(
            "max_quote_age_minutes must be non-negative."
        )

    if wide_spread_pct < 0:
        raise ValueError(
            "wide_spread_pct must be non-negative."
        )

    if extreme_percent_change < 0:
        raise ValueError(
            "extreme_percent_change must be non-negative."
        )

    processed_root = Path(
        processed_root
    )

    (
        summary_path,
        summary,
    ) = _load_capture_summary(
        processed_root=
            processed_root,

        run_summary=
            run_summary,
    )

    artifacts = (
        _artifact_paths_by_dataset(
            processed_root=
                processed_root,

            summary=
                summary,
        )
    )

    frames = {
        dataset:
            _load_dataset(
                dataset_artifacts
            )
        for dataset, dataset_artifacts
        in artifacts.items()
    }

    issues: list[
        ValidationIssue
    ] = []

    quotes = frames.get(
        "quotes",
        pd.DataFrame(),
    )

    options = frames.get(
        "option_chains",
        pd.DataFrame(),
    )

    price_history = frames.get(
        "price_history",
        pd.DataFrame(),
    )

    quote_result = (
        _validate_quotes(
            quotes,
            issues=
                issues,
            max_quote_age_minutes=
                max_quote_age_minutes,
        )
    )

    option_result = (
        _validate_options(
            options,
            quotes=
                quotes,
            issues=
                issues,
            max_quote_age_minutes=
                max_quote_age_minutes,
            wide_spread_pct=
                wide_spread_pct,
            extreme_percent_change=
                extreme_percent_change,
        )
    )

    price_result = (
        _validate_price_history(
            price_history,
            issues=
                issues,
        )
    )

    error_count = sum(
        issue.level
        == "ERROR"
        for issue in issues
    )

    warning_count = sum(
        issue.level
        == "WARNING"
        for issue in issues
    )

    if error_count:
        status = "FAIL"

    elif warning_count:
        status = "PASS_WITH_WARNINGS"

    else:
        status = "PASS"

    identity = {
        "validation_schema_version":
            VALIDATION_SCHEMA_VERSION,

        "run_id":
            str(
                summary.get(
                    "run_id",
                    "",
                )
            ),

        "thresholds":
            {
                "max_quote_age_minutes":
                    max_quote_age_minutes,

                "wide_spread_pct":
                    wide_spread_pct,

                "extreme_percent_change":
                    extreme_percent_change,
            },

        "issue_codes":
            [
                {
                    "level":
                        issue.level,

                    "code":
                        issue.code,

                    "dataset":
                        issue.dataset,

                    "count":
                        issue.count,
                }
                for issue in issues
            ],
    }

    return {
        "validation_schema_version":
            VALIDATION_SCHEMA_VERSION,

        "validation_id":
            _sha256_json(
                identity
            ),

        "run_id":
            str(
                summary.get(
                    "run_id",
                    "",
                )
            ),

        "run_summary_path":
            str(
                summary_path
            ),

        "run_started_at_utc":
            str(
                summary.get(
                    "started_at_utc",
                    "",
                )
            ),

        "symbols":
            [
                str(
                    symbol
                )
                for symbol
                in summary.get(
                    "symbols",
                    [],
                )
            ],

        "thresholds":
            {
                "max_quote_age_minutes":
                    float(
                        max_quote_age_minutes
                    ),

                "wide_spread_pct":
                    float(
                        wide_spread_pct
                    ),

                "extreme_percent_change":
                    float(
                        extreme_percent_change
                    ),
            },

        "status":
            status,

        "error_count":
            int(
                error_count
            ),

        "warning_count":
            int(
                warning_count
            ),

        "issues":
            [
                asdict(
                    issue
                )
                for issue
                in issues
            ],

        "quotes":
            quote_result,

        "option_chains":
            option_result,

        "price_history":
            price_result,
    }


def _print_quantiles(
    name: str,
    values: Mapping[
        str,
        Any,
    ] | None,
) -> None:
    if not isinstance(
        values,
        Mapping,
    ):
        return

    print(
        f"  {name:<28} "
        f"p05={values.get('p05')!s:<12} "
        f"median={values.get('median')!s:<12} "
        f"p95={values.get('p95')!s:<12} "
        f"min={values.get('min')!s:<12} "
        f"max={values.get('max')!s}"
    )


def _print_result(
    result: Mapping[
        str,
        Any,
    ],
) -> None:
    print()
    print(
        "=" * 88
    )
    print(
        "SCHWAB FEATURE-INPUT VALIDATION"
    )
    print(
        "=" * 88
    )

    print(
        "Run ID:",
        result[
            "run_id"
        ],
    )

    print(
        "Validation ID:",
        result[
            "validation_id"
        ],
    )

    print(
        "Symbols:",
        ", ".join(
            result[
                "symbols"
            ]
        ),
    )

    thresholds = result[
        "thresholds"
    ]

    print(
        "Max quote age:",
        f"{thresholds['max_quote_age_minutes']:.1f} min",
    )

    print(
        "Wide spread threshold:",
        f"{thresholds['wide_spread_pct']:.1f}%",
    )

    print(
        "Extreme percent-change threshold:",
        f"{thresholds['extreme_percent_change']:.0f}%",
    )

    options = result.get(
        "option_chains",
        {},
    )

    if isinstance(
        options,
        Mapping,
    ) and options:
        print()
        print(
            "OPTION QUALITY"
        )
        print(
            "-" * 88
        )

        print(
            "Rows:",
            options.get(
                "rows"
            ),
        )

        print(
            "Zero bid:",
            (
                f"{options.get('zero_bid_count')} "
                f"({options.get('zero_bid_pct')}%)"
            ),
        )

        print(
            "Wide spread:",
            (
                f"{options.get('wide_spread_count')} "
                f"({options.get('wide_spread_pct')}%)"
            ),
        )

        print(
            "Stale quote:",
            (
                f"{options.get('stale_quote_count')} "
                f"({options.get('stale_quote_pct')}%)"
            ),
        )

        print(
            "0-DTE:",
            (
                f"{options.get('zero_dte_count')} "
                f"({options.get('zero_dte_pct')}%)"
            ),
        )

        constant = options.get(
            "constant_research_fields",
            [],
        )

        print(
            "Constant research fields:",
            (
                ", ".join(
                    str(
                        field
                    )
                    for field in constant
                )
                if constant
                else "none"
            ),
        )

        diagnostics = options.get(
            "diagnostics",
            {},
        )

        if isinstance(
            diagnostics,
            Mapping,
        ):
            print()
            print(
                "OPTION DISTRIBUTIONS"
            )

            for field in (
                "volatility",
                "delta",
                "gamma",
                "theta",
                "vega",
                "rho",
                "openInterest",
                "totalVolume",
                "intrinsicValue",
                "extrinsicValue",
                "relative_spread_pct",
                "quote_age_minutes",
            ):
                _print_quantiles(
                    field,
                    diagnostics.get(
                        field
                    ),
                )

        coverage = options.get(
            "coverage_by_symbol",
            {},
        )

        if isinstance(
            coverage,
            Mapping,
        ) and coverage:
            print()
            print(
                "OPTION COVERAGE BY SYMBOL"
            )
            print(
                "-" * 88
            )

            for symbol, row in (
                coverage.items()
            ):
                print()
                print(
                    symbol
                )

                print(
                    "  contracts:",
                    row.get(
                        "contracts"
                    ),
                )

                print(
                    "  spot:",
                    row.get(
                        "spot"
                    ),
                )

                print(
                    "  side:",
                    row.get(
                        "contracts_by_side"
                    ),
                )

                print(
                    "  DTE:",
                    (
                        f"{row.get('dte_min')} "
                        f"-> {row.get('dte_max')}"
                    ),
                )

                print(
                    "  DTE buckets:",
                    row.get(
                        "contracts_by_dte_bucket"
                    ),
                )

                print(
                    "  moneyness:",
                    row.get(
                        "contracts_by_moneyness"
                    ),
                )

                print(
                    "  zero bid:",
                    (
                        f"{row.get('zero_bid_count')} "
                        f"({row.get('zero_bid_pct')}%)"
                    ),
                )

                print(
                    "  wide spread:",
                    (
                        f"{row.get('wide_spread_count')} "
                        f"({row.get('wide_spread_pct')}%)"
                    ),
                )

                print(
                    "  stale quote:",
                    (
                        f"{row.get('stale_quote_count')} "
                        f"({row.get('stale_quote_pct')}%)"
                    ),
                )

    price = result.get(
        "price_history",
        {},
    )

    if isinstance(
        price,
        Mapping,
    ) and price:
        print()
        print(
            "PRICE-HISTORY COVERAGE"
        )
        print(
            "-" * 88
        )

        print(
            "Rows:",
            price.get(
                "rows"
            ),
        )

        print(
            "Range:",
            (
                f"{price.get('start_utc')} "
                f"-> {price.get('end_utc')}"
            ),
        )

        for symbol, row in (
            price.get(
                "symbols",
                {}
            ).items()
        ):
            print(
                f"  {symbol:<6} "
                f"rows={row.get('rows'):<7} "
                f"zero_volume={row.get('zero_volume_pct')}% "
                f"{row.get('start_utc')} -> "
                f"{row.get('end_utc')}"
            )

    print()
    print(
        "ISSUES"
    )
    print(
        "-" * 88
    )

    issues = result.get(
        "issues",
        [],
    )

    if not issues:
        print(
            "None"
        )

    else:
        for issue in issues:
            count_text = ""

            if issue.get(
                "count"
            ) is not None:
                count_text = (
                    f" count={issue['count']}"
                )

                if issue.get(
                    "percentage"
                ) is not None:
                    count_text += (
                        f" ({issue['percentage']}%)"
                    )

            print(
                f"[{issue['level']}] "
                f"{issue['dataset']} / "
                f"{issue['code']}: "
                f"{issue['message']}"
                f"{count_text}"
            )

    print()
    print(
        "ERRORS:",
        result[
            "error_count"
        ],
    )

    print(
        "WARNINGS:",
        result[
            "warning_count"
        ],
    )

    print(
        "FEATURE-INPUT GATE:",
        result[
            "status"
        ],
    )

    print(
        "=" * 88
    )


def _write_json(
    *,
    output_path: Path,
    result: Mapping[
        str,
        Any,
    ],
) -> None:
    path = Path(
        output_path
    )

    if not path.is_absolute():
        path = (
            PROJECT_ROOT
            / path
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    encoded = (
        json.dumps(
            result,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode(
        "utf-8"
    )

    if path.exists():
        existing = (
            path.read_bytes()
        )

        if existing != encoded:
            raise RuntimeError(
                "Validation output already exists with "
                "different content: "
                f"{path}"
            )

        return

    with path.open(
        "xb"
    ) as handle:
        handle.write(
            encoded
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate normalized Schwab market-data "
            "inputs before feature engineering."
        )
    )

    parser.add_argument(
        "--processed-root",
        type=Path,
        default=
            DEFAULT_PROCESSED_ROOT,
        help=
            "Normalized Schwab archive root.",
    )

    parser.add_argument(
        "--run-summary",
        type=Path,
        default=None,
        help=(
            "Specific capture-run summary JSON. "
            "Defaults to the newest summary."
        ),
    )

    parser.add_argument(
        "--max-quote-age-minutes",
        type=float,
        default=
            DEFAULT_MAX_QUOTE_AGE_MINUTES,
        help=(
            "Warn when a quote is older than this "
            "at capture time. Default: 30."
        ),
    )

    parser.add_argument(
        "--wide-spread-pct",
        type=float,
        default=
            DEFAULT_WIDE_SPREAD_PCT,
        help=(
            "Warn when option bid/ask spread as a "
            "percentage of midpoint exceeds this value. "
            "Default: 25."
        ),
    )

    parser.add_argument(
        "--extreme-percent-change",
        type=float,
        default=
            DEFAULT_EXTREME_PERCENT_CHANGE,
        help=(
            "Warn when absolute option percent-change "
            "fields exceed this value. Default: 1000."
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help=
            "Print the complete validation result as JSON.",
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Optionally persist the validation result "
            "with create-once semantics."
        ),
    )

    return parser


def main() -> None:
    args = (
        build_parser()
        .parse_args()
    )

    result = (
        validate_schwab_feature_inputs(
            processed_root=
                args.processed_root,

            run_summary=
                args.run_summary,

            max_quote_age_minutes=
                args.max_quote_age_minutes,

            wide_spread_pct=
                args.wide_spread_pct,

            extreme_percent_change=
                args.extreme_percent_change,
        )
    )

    if args.output_json is not None:
        _write_json(
            output_path=
                args.output_json,

            result=
                result,
        )

    if args.json:
        print(
            json.dumps(
                result,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
        )

    else:
        _print_result(
            result
        )

    if (
        result[
            "status"
        ]
        == "FAIL"
    ):
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
