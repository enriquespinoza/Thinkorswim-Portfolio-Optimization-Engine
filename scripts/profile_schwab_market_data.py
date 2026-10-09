from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
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
PROFILE_SCHEMA_VERSION = "1.0.0"

PROVENANCE_COLUMNS = {
    "raw_record_id",
    "raw_dataset",
    "raw_captured_at_utc",
    "raw_request_sha256",
    "raw_payload_sha256",
    "normalized_schema_version",
}

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
    "theoreticalOptionValue",
    "theoreticalVolatility",
)


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
            candidate = (
                PROJECT_ROOT
                / summary_path
            )

            if candidate.is_file():
                summary_path = candidate

    summary = _read_json(
        summary_path,
        context=
            "Schwab capture-run summary",
    )

    run_id = str(
        summary.get(
            "run_id",
            "",
        )
    ).strip()

    if not run_id:
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


def _read_dataset(
    artifacts: Iterable[
        tuple[
            str,
            Path,
        ]
    ],
) -> tuple[
    pd.DataFrame,
    list[
        dict[
            str,
            Any,
        ]
    ],
]:
    frames: list[
        pd.DataFrame
    ] = []

    artifact_profiles: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for scope, path in artifacts:
        frame = pd.read_csv(
            path,
            low_memory=False,
        )

        frames.append(
            frame
        )

        artifact_profiles.append(
            {
                "scope":
                    scope,

                "path":
                    str(
                        path
                    ),

                "rows":
                    int(
                        len(
                            frame
                        )
                    ),

                "columns":
                    int(
                        len(
                            frame.columns
                        )
                    ),
            }
        )

    if not frames:
        return (
            pd.DataFrame(),
            artifact_profiles,
        )

    combined = pd.concat(
        frames,
        axis=0,
        ignore_index=True,
        sort=False,
    )

    return (
        combined,
        artifact_profiles,
    )


def _json_scalar(
    value: Any,
) -> Any:
    if pd.isna(
        value
    ):
        return None

    if hasattr(
        value,
        "item",
    ):
        try:
            return value.item()
        except Exception:
            pass

    if isinstance(
        value,
        (
            datetime,
            pd.Timestamp,
        ),
    ):
        return value.isoformat()

    return value


def _numeric_stats(
    series: pd.Series,
) -> dict[str, Any] | None:
    if not pd.api.types.is_numeric_dtype(
        series.dtype
    ):
        return None

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return None

    return {
        "min":
            _json_scalar(
                numeric.min()
            ),

        "max":
            _json_scalar(
                numeric.max()
            ),

        "mean":
            _json_scalar(
                numeric.mean()
            ),

        "median":
            _json_scalar(
                numeric.median()
            ),
    }


def _looks_datetime_like(
    column: str,
) -> bool:
    lowered = str(
        column
    ).lower()

    tokens = (
        "date",
        "time",
        "datetime",
        "expiration",
    )

    return any(
        token in lowered
        for token in tokens
    )


def _datetime_stats(
    series: pd.Series,
    *,
    column: str,
) -> dict[str, Any] | None:
    if not _looks_datetime_like(
        column
    ):
        return None

    non_null = series.dropna()

    if non_null.empty:
        return None

    parsed: pd.Series

    if pd.api.types.is_numeric_dtype(
        non_null.dtype
    ):
        numeric = pd.to_numeric(
            non_null,
            errors="coerce",
        )

        finite = numeric.dropna()

        if finite.empty:
            return None

        median_abs = float(
            finite.abs().median()
        )

        if median_abs >= 1e11:
            unit = "ms"

        elif median_abs >= 1e8:
            unit = "s"

        else:
            return None

        parsed = pd.to_datetime(
            numeric,
            unit=unit,
            errors="coerce",
            utc=True,
        )

    else:
        parsed = pd.to_datetime(
            non_null,
            errors="coerce",
            utc=True,
        )

    valid = parsed.dropna()

    parse_rate = (
        len(
            valid
        )
        / len(
            non_null
        )
    )

    if (
        valid.empty
        or parse_rate < 0.80
    ):
        return None

    return {
        "parse_rate_pct":
            round(
                100.0
                * parse_rate,
                4,
            ),

        "min_utc":
            valid.min().isoformat(),

        "max_utc":
            valid.max().isoformat(),
    }


def _column_profile(
    frame: pd.DataFrame,
    column: str,
) -> dict[str, Any]:
    series = frame[
        column
    ]

    total = int(
        len(
            series
        )
    )

    non_null_count = int(
        series.notna().sum()
    )

    null_count = (
        total
        - non_null_count
    )

    if total:
        non_null_pct = (
            100.0
            * non_null_count
            / total
        )

    else:
        non_null_pct = 0.0

    non_null = series.dropna()

    if non_null.empty:
        unique_count = 0

    else:
        unique_count = int(
            non_null.nunique(
                dropna=True
            )
        )

    profile = {
        "column":
            column,

        "dtype":
            str(
                series.dtype
            ),

        "rows":
            total,

        "non_null_count":
            non_null_count,

        "non_null_pct":
            round(
                non_null_pct,
                4,
            ),

        "null_count":
            null_count,

        "unique_count":
            unique_count,

        "is_provenance":
            column
            in PROVENANCE_COLUMNS,
    }

    numeric = _numeric_stats(
        series
    )

    if numeric is not None:
        profile[
            "numeric"
        ] = numeric

    datetime_stats = (
        _datetime_stats(
            series,
            column=
                column,
        )
    )

    if datetime_stats is not None:
        profile[
            "datetime"
        ] = datetime_stats

    return profile


def _ordered_unique_strings(
    series: pd.Series,
) -> list[str]:
    values = (
        series
        .dropna()
        .astype(str)
        .str.strip()
    )

    return sorted(
        {
            value
            for value in values
            if value
        }
    )


def _symbol_column(
    frame: pd.DataFrame,
) -> str | None:
    preferred = (
        "symbol",
        "underlying_symbol",
    )

    for column in preferred:
        if column in frame.columns:
            return column

    candidates = sorted(
        column
        for column in frame.columns
        if column.endswith(
            "__symbol"
        )
    )

    if candidates:
        return candidates[0]

    return None


def _dataset_summary(
    dataset: str,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    summary: dict[
        str,
        Any,
    ] = {}

    symbol_column = (
        _symbol_column(
            frame
        )
    )

    if symbol_column is not None:
        summary[
            "symbol_column"
        ] = symbol_column

        summary[
            "symbols"
        ] = _ordered_unique_strings(
            frame[
                symbol_column
            ]
        )

        counts = (
            frame[
                symbol_column
            ]
            .dropna()
            .astype(str)
            .value_counts()
        )

        summary[
            "rows_by_symbol"
        ] = {
            str(
                symbol
            ):
                int(
                    count
                )
            for symbol, count
            in counts.items()
        }

    if dataset == "price_history":
        timestamp_column = (
            "datetime_utc"
            if "datetime_utc"
            in frame.columns
            else (
                "datetime"
                if "datetime"
                in frame.columns
                else None
            )
        )

        if timestamp_column:
            stats = _datetime_stats(
                frame[
                    timestamp_column
                ],
                column=
                    timestamp_column,
            )

            if stats is not None:
                summary[
                    "time_range"
                ] = {
                    "column":
                        timestamp_column,

                    **stats,
                }

    if dataset == "option_chains":
        if "put_call" in frame.columns:
            side_counts = (
                frame[
                    "put_call"
                ]
                .dropna()
                .astype(str)
                .str.upper()
                .value_counts()
            )

            summary[
                "contracts_by_side"
            ] = {
                str(
                    side
                ):
                    int(
                        count
                    )
                for side, count
                in side_counts.items()
            }

        if "expiration_date" in frame.columns:
            expiration = pd.to_datetime(
                frame[
                    "expiration_date"
                ],
                errors="coerce",
            ).dropna()

            if not expiration.empty:
                summary[
                    "expiration_count"
                ] = int(
                    expiration.nunique()
                )

                summary[
                    "expiration_min"
                ] = (
                    expiration.min()
                    .date()
                    .isoformat()
                )

                summary[
                    "expiration_max"
                ] = (
                    expiration.max()
                    .date()
                    .isoformat()
                )

        if "expiration_dte" in frame.columns:
            dte = pd.to_numeric(
                frame[
                    "expiration_dte"
                ],
                errors="coerce",
            ).dropna()

            if not dte.empty:
                summary[
                    "dte_min"
                ] = _json_scalar(
                    dte.min()
                )

                summary[
                    "dte_max"
                ] = _json_scalar(
                    dte.max()
                )

        if "strike" in frame.columns:
            strike = pd.to_numeric(
                frame[
                    "strike"
                ],
                errors="coerce",
            ).dropna()

            if not strike.empty:
                summary[
                    "strike_min"
                ] = _json_scalar(
                    strike.min()
                )

                summary[
                    "strike_max"
                ] = _json_scalar(
                    strike.max()
                )

        option_fields: dict[
            str,
            dict[
                str,
                Any,
            ]
        ] = {}

        for field in OPTION_RESEARCH_FIELDS:
            if field not in frame.columns:
                continue

            series = frame[
                field
            ]

            non_null_count = int(
                series.notna().sum()
            )

            if len(
                series
            ):
                non_null_pct = (
                    100.0
                    * non_null_count
                    / len(
                        series
                    )
                )

            else:
                non_null_pct = 0.0

            option_fields[
                field
            ] = {
                "non_null_count":
                    non_null_count,

                "non_null_pct":
                    round(
                        non_null_pct,
                        4,
                    ),
            }

        summary[
            "option_research_field_coverage"
        ] = option_fields

    if dataset == "expiration_chains":
        date_column = next(
            (
                column
                for column in (
                    "expirationDate",
                    "expiration_date",
                )
                if column
                in frame.columns
            ),
            None,
        )

        if date_column:
            expiration = pd.to_datetime(
                frame[
                    date_column
                ],
                errors="coerce",
            ).dropna()

            if not expiration.empty:
                summary[
                    "expiration_count"
                ] = int(
                    expiration.nunique()
                )

                summary[
                    "expiration_min"
                ] = (
                    expiration.min()
                    .date()
                    .isoformat()
                )

                summary[
                    "expiration_max"
                ] = (
                    expiration.max()
                    .date()
                    .isoformat()
                )

    return summary


def _sparse_columns(
    columns: list[
        dict[
            str,
            Any,
        ]
    ],
    *,
    threshold_pct: float,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    return sorted(
        (
            {
                "column":
                    str(
                        row[
                            "column"
                        ]
                    ),

                "non_null_pct":
                    float(
                        row[
                            "non_null_pct"
                        ]
                    ),
            }
            for row in columns
            if (
                not bool(
                    row[
                        "is_provenance"
                    ]
                )
                and float(
                    row[
                        "non_null_pct"
                    ]
                )
                < threshold_pct
            )
        ),
        key=lambda row:
            (
                row[
                    "non_null_pct"
                ],
                row[
                    "column"
                ],
            ),
    )


def profile_schwab_market_data(
    *,
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
    run_summary: Path | None = None,
    sparse_threshold_pct: float = 95.0,
) -> dict[str, Any]:
    """
    Profile the normalized tables referenced by one
    immutable Schwab capture-run summary.

    This is a read-only descriptive step. It does not
    call Schwab, alter normalized data, or calculate
    predictive features.
    """
    processed_root = Path(
        processed_root
    )

    if not (
        0.0
        <= sparse_threshold_pct
        <= 100.0
    ):
        raise ValueError(
            "sparse_threshold_pct must be between "
            "0 and 100."
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

    grouped = (
        _artifact_paths_by_dataset(
            processed_root=
                processed_root,

            summary=
                summary,
        )
    )

    dataset_profiles: dict[
        str,
        dict[
            str,
            Any,
        ]
    ] = {}

    for dataset in sorted(
        grouped
    ):
        (
            frame,
            artifact_profiles,
        ) = _read_dataset(
            grouped[
                dataset
            ]
        )

        columns = [
            _column_profile(
                frame,
                column,
            )
            for column
            in frame.columns
        ]

        dataset_profiles[
            dataset
        ] = {
            "artifact_count":
                len(
                    artifact_profiles
                ),

            "rows":
                int(
                    len(
                        frame
                    )
                ),

            "columns":
                int(
                    len(
                        frame.columns
                    )
                ),

            "artifacts":
                artifact_profiles,

            "dataset_summary":
                _dataset_summary(
                    dataset,
                    frame,
                ),

            "column_profile":
                columns,

            "sparse_columns":
                _sparse_columns(
                    columns,
                    threshold_pct=
                        sparse_threshold_pct,
                ),
        }

    profile_identity = {
        "profile_schema_version":
            PROFILE_SCHEMA_VERSION,

        "run_id":
            str(
                summary.get(
                    "run_id",
                    "",
                )
            ),

        "datasets":
            {
                dataset:
                    {
                        "artifact_count":
                            row[
                                "artifact_count"
                            ],

                        "rows":
                            row[
                                "rows"
                            ],

                        "columns":
                            row[
                                "columns"
                            ],
                    }
                for dataset, row
                in dataset_profiles.items()
            },
    }

    return {
        "profile_schema_version":
            PROFILE_SCHEMA_VERSION,

        "profile_id":
            _sha256_json(
                profile_identity
            ),

        "run_id":
            str(
                summary.get(
                    "run_id",
                    "",
                )
            ),

        "run_started_at_utc":
            str(
                summary.get(
                    "started_at_utc",
                    "",
                )
            ),

        "run_completed_at_utc":
            str(
                summary.get(
                    "completed_at_utc",
                    "",
                )
            ),

        "run_summary_path":
            str(
                summary_path
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

        "sparse_threshold_pct":
            float(
                sparse_threshold_pct
            ),

        "datasets":
            dataset_profiles,
    }


def _format_scalar(
    value: Any,
) -> str:
    if value is None:
        return "-"

    if isinstance(
        value,
        float,
    ):
        return f"{value:.6g}"

    return str(
        value
    )


def _print_column_profile(
    rows: Iterable[
        Mapping[
            str,
            Any,
        ]
    ],
) -> None:
    print(
        f"{'COLUMN':<42} "
        f"{'DTYPE':<12} "
        f"{'NON-NULL':>9} "
        f"{'UNIQUE':>9} "
        f"{'MIN':>14} "
        f"{'MAX':>14}"
    )

    print(
        "-" * 106
    )

    for row in rows:
        numeric = row.get(
            "numeric"
        )

        if isinstance(
            numeric,
            Mapping,
        ):
            minimum = _format_scalar(
                numeric.get(
                    "min"
                )
            )

            maximum = _format_scalar(
                numeric.get(
                    "max"
                )
            )

        else:
            minimum = "-"
            maximum = "-"

        print(
            f"{str(row['column'])[:42]:<42} "
            f"{str(row['dtype'])[:12]:<12} "
            f"{float(row['non_null_pct']):>8.2f}% "
            f"{int(row['unique_count']):>9} "
            f"{minimum[:14]:>14} "
            f"{maximum[:14]:>14}"
        )


def _print_dataset_summary(
    dataset: str,
    profile: Mapping[
        str,
        Any,
    ],
) -> None:
    print()
    print(
        dataset.upper()
    )
    print(
        "-" * 88
    )
    print(
        "Artifacts:",
        profile[
            "artifact_count"
        ],
    )
    print(
        "Rows:",
        profile[
            "rows"
        ],
    )
    print(
        "Columns:",
        profile[
            "columns"
        ],
    )

    summary = profile.get(
        "dataset_summary",
        {},
    )

    if isinstance(
        summary,
        Mapping,
    ):
        symbols = summary.get(
            "symbols"
        )

        if isinstance(
            symbols,
            list,
        ) and symbols:
            print(
                "Symbols:",
                ", ".join(
                    str(
                        symbol
                    )
                    for symbol in symbols
                ),
            )

        time_range = summary.get(
            "time_range"
        )

        if isinstance(
            time_range,
            Mapping,
        ):
            print(
                "Time range:",
                time_range.get(
                    "min_utc"
                ),
                "->",
                time_range.get(
                    "max_utc"
                ),
            )

        if (
            "contracts_by_side"
            in summary
        ):
            print(
                "Contracts by side:",
                summary[
                    "contracts_by_side"
                ],
            )

        if (
            "expiration_count"
            in summary
        ):
            print(
                "Expirations:",
                summary.get(
                    "expiration_count"
                ),
                (
                    f"({summary.get('expiration_min')} "
                    f"-> {summary.get('expiration_max')})"
                ),
            )

        if (
            "dte_min"
            in summary
            or "dte_max"
            in summary
        ):
            print(
                "DTE range:",
                summary.get(
                    "dte_min"
                ),
                "->",
                summary.get(
                    "dte_max"
                ),
            )

        if (
            "strike_min"
            in summary
            or "strike_max"
            in summary
        ):
            print(
                "Strike range:",
                summary.get(
                    "strike_min"
                ),
                "->",
                summary.get(
                    "strike_max"
                ),
            )

        option_coverage = summary.get(
            "option_research_field_coverage"
        )

        if isinstance(
            option_coverage,
            Mapping,
        ) and option_coverage:
            print()
            print(
                "Option research-field coverage:"
            )

            for field, row in (
                option_coverage.items()
            ):
                print(
                    f"  {field:<28} "
                    f"{float(row['non_null_pct']):>8.2f}% "
                    f"({int(row['non_null_count'])} rows)"
                )

    sparse = profile.get(
        "sparse_columns",
        [],
    )

    if sparse:
        print()
        print(
            "Sparse non-provenance columns:"
        )

        for row in sparse:
            print(
                f"  {str(row['column']):<42} "
                f"{float(row['non_null_pct']):>8.2f}%"
            )

    print()
    _print_column_profile(
        profile[
            "column_profile"
        ]
    )


def _print_profile(
    profile: Mapping[
        str,
        Any,
    ],
) -> None:
    print()
    print(
        "=" * 88
    )
    print(
        "SCHWAB MARKET DATA PROFILE"
    )
    print(
        "=" * 88
    )
    print(
        "Run ID:",
        profile[
            "run_id"
        ],
    )
    print(
        "Profile ID:",
        profile[
            "profile_id"
        ],
    )
    print(
        "Started UTC:",
        profile[
            "run_started_at_utc"
        ],
    )
    print(
        "Symbols:",
        ", ".join(
            profile[
                "symbols"
            ]
        ),
    )
    print(
        "Sparse threshold:",
        f"{profile['sparse_threshold_pct']:.2f}%",
    )

    datasets = profile.get(
        "datasets",
        {},
    )

    for dataset, dataset_profile in (
        datasets.items()
    ):
        _print_dataset_summary(
            dataset,
            dataset_profile,
        )

    print()
    print(
        "=" * 88
    )
    print(
        "PROFILE COMPLETE"
    )
    print(
        "=" * 88
    )


def _write_json(
    *,
    output_path: Path,
    profile: Mapping[
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
            profile,
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
                "Profile output already exists with "
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
            "Profile the normalized Schwab market-data "
            "tables referenced by one capture run."
        )
    )

    parser.add_argument(
        "--processed-root",
        type=Path,
        default=
            DEFAULT_PROCESSED_ROOT,
        help=(
            "Normalized Schwab archive root."
        ),
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
        "--sparse-threshold-pct",
        type=float,
        default=95.0,
        help=(
            "Report non-provenance columns below this "
            "non-null percentage as sparse. Default: 95."
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Print the complete profile as JSON."
        ),
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Optionally write the complete profile JSON "
            "to a file using create-once semantics."
        ),
    )

    return parser


def main() -> None:
    args = (
        build_parser()
        .parse_args()
    )

    profile = (
        profile_schwab_market_data(
            processed_root=
                args.processed_root,

            run_summary=
                args.run_summary,

            sparse_threshold_pct=
                args.sparse_threshold_pct,
        )
    )

    if args.output_json is not None:
        _write_json(
            output_path=
                args.output_json,

            profile=
                profile,
        )

    if args.json:
        print(
            json.dumps(
                profile,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
        )

    else:
        _print_profile(
            profile
        )


if __name__ == "__main__":
    main()
