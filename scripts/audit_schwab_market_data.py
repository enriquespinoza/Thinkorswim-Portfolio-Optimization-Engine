from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
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
    RAW_DATA_DIR,
)


DEFAULT_RAW_ROOT = (
    RAW_DATA_DIR
    / "schwab"
)

DEFAULT_PROCESSED_ROOT = (
    PROCESSED_DATA_DIR
    / "schwab"
)

RAW_MANIFEST_FILENAME = "manifest.jsonl"
NORMALIZED_MANIFEST_FILENAME = "manifest.jsonl"
CAPTURE_RUNS_DIRECTORY = "capture_runs"


@dataclass(frozen=True)
class AuditIssue:
    level: str
    code: str
    message: str


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


def _sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda:
                handle.read(
                    1024 * 1024
                ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


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


def _read_jsonl(
    path: Path,
    *,
    context: str,
) -> list[dict[str, Any]]:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{context} not found: {path}"
        )

    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                value = json.loads(
                    stripped
                )
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{context} contains invalid JSON "
                    f"on line {line_number}."
                ) from exc

            if not isinstance(
                value,
                Mapping,
            ):
                raise RuntimeError(
                    f"{context} line {line_number} "
                    "must contain a JSON object."
                )

            rows.append(
                dict(
                    value
                )
            )

    return rows


def _record_id_counts(
    rows: Iterable[
        Mapping[
            str,
            Any,
        ]
    ],
    key: str,
) -> Counter:
    return Counter(
        str(
            row.get(
                key,
                "",
            )
        )
        for row in rows
        if str(
            row.get(
                key,
                "",
            )
        )
    )


def _append_issue(
    issues: list[AuditIssue],
    *,
    level: str,
    code: str,
    message: str,
) -> None:
    issues.append(
        AuditIssue(
            level=
                level,

            code=
                code,

            message=
                message,
        )
    )


def _latest_capture_summary(
    processed_root: Path,
) -> Path | None:
    run_dir = (
        Path(
            processed_root
        )
        / CAPTURE_RUNS_DIRECTORY
    )

    if not run_dir.exists():
        return None

    candidates = sorted(
        run_dir.glob(
            "*.json"
        )
    )

    if not candidates:
        return None

    return candidates[-1]


def _load_processed_frame(
    processed_root: Path,
    relative_path: str,
) -> pd.DataFrame:
    path = (
        Path(
            processed_root
        )
        / str(
            relative_path
        )
    )

    return pd.read_csv(
        path
    )


def _audit_manifests(
    *,
    raw_root: Path,
    processed_root: Path,
    issues: list[AuditIssue],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    raw_manifest = (
        Path(
            raw_root
        )
        / RAW_MANIFEST_FILENAME
    )

    normalized_manifest = (
        Path(
            processed_root
        )
        / NORMALIZED_MANIFEST_FILENAME
    )

    raw_rows = _read_jsonl(
        raw_manifest,
        context=
            "Raw Schwab manifest",
    )

    normalized_rows = _read_jsonl(
        normalized_manifest,
        context=
            "Normalized Schwab manifest",
    )

    raw_counts = (
        _record_id_counts(
            raw_rows,
            "record_id",
        )
    )

    normalized_counts = (
        _record_id_counts(
            normalized_rows,
            "raw_record_id",
        )
    )

    for record_id, count in (
        raw_counts.items()
    ):
        if count > 1:
            _append_issue(
                issues,
                level="ERROR",
                code=
                    "DUPLICATE_RAW_RECORD_ID",
                message=(
                    f"Raw manifest contains record_id "
                    f"{record_id} {count} times."
                ),
            )

    for record_id, count in (
        normalized_counts.items()
    ):
        if count > 1:
            _append_issue(
                issues,
                level="ERROR",
                code=
                    "DUPLICATE_NORMALIZED_RECORD_ID",
                message=(
                    "Normalized manifest contains "
                    f"raw_record_id {record_id} "
                    f"{count} times."
                ),
            )

    raw_by_id = {
        str(
            row.get(
                "record_id"
            )
        ):
            row
        for row in raw_rows
        if row.get(
            "record_id"
        )
    }

    normalized_by_id = {
        str(
            row.get(
                "raw_record_id"
            )
        ):
            row
        for row in normalized_rows
        if row.get(
            "raw_record_id"
        )
    }

    for record_id, row in (
        raw_by_id.items()
    ):
        relative_path = str(
            row.get(
                "relative_path",
                "",
            )
        )

        if not relative_path:
            _append_issue(
                issues,
                level="ERROR",
                code=
                    "RAW_PATH_MISSING",
                message=(
                    f"Raw manifest record {record_id} "
                    "has no relative_path."
                ),
            )
            continue

        raw_path = (
            Path(
                raw_root
            )
            / relative_path
        )

        if not raw_path.is_file():
            _append_issue(
                issues,
                level="ERROR",
                code=
                    "RAW_FILE_MISSING",
                message=(
                    f"Raw artifact is missing: "
                    f"{relative_path}"
                ),
            )
            continue

        try:
            envelope = _read_json(
                raw_path,
                context=
                    "Raw Schwab artifact",
            )
        except Exception as exc:
            _append_issue(
                issues,
                level="ERROR",
                code=
                    "RAW_FILE_INVALID",
                message=(
                    f"{relative_path}: "
                    f"{type(exc).__name__}"
                ),
            )
            continue

        envelope_record_id = str(
            envelope.get(
                "record_id",
                "",
            )
        )

        if (
            envelope_record_id
            != record_id
        ):
            _append_issue(
                issues,
                level="ERROR",
                code=
                    "RAW_RECORD_ID_MISMATCH",
                message=(
                    f"{relative_path} record_id "
                    "does not match the manifest."
                ),
            )

        payload = envelope.get(
            "payload"
        )

        if not isinstance(
            payload,
            Mapping,
        ):
            _append_issue(
                issues,
                level="ERROR",
                code=
                    "RAW_PAYLOAD_INVALID",
                message=(
                    f"{relative_path} payload is "
                    "not a JSON object."
                ),
            )
            continue

        actual_payload_hash = (
            _sha256_json(
                payload
            )
        )

        envelope_hash = str(
            envelope.get(
                "payload_sha256",
                "",
            )
        )

        manifest_hash = str(
            row.get(
                "payload_sha256",
                "",
            )
        )

        if (
            actual_payload_hash
            != envelope_hash
            or actual_payload_hash
            != manifest_hash
        ):
            _append_issue(
                issues,
                level="ERROR",
                code=
                    "RAW_PAYLOAD_HASH_MISMATCH",
                message=(
                    f"{relative_path} payload SHA-256 "
                    "does not match recorded provenance."
                ),
            )

    for record_id, row in (
        normalized_by_id.items()
    ):
        relative_path = str(
            row.get(
                "relative_path",
                "",
            )
        )

        if record_id not in raw_by_id:
            _append_issue(
                issues,
                level="ERROR",
                code=
                    "NORMALIZED_WITHOUT_RAW",
                message=(
                    "Normalized record references unknown "
                    f"raw_record_id {record_id}."
                ),
            )

        if not relative_path:
            _append_issue(
                issues,
                level="ERROR",
                code=
                    "NORMALIZED_PATH_MISSING",
                message=(
                    f"Normalized record {record_id} "
                    "has no relative_path."
                ),
            )
            continue

        processed_path = (
            Path(
                processed_root
            )
            / relative_path
        )

        if not processed_path.is_file():
            _append_issue(
                issues,
                level="ERROR",
                code=
                    "NORMALIZED_FILE_MISSING",
                message=(
                    "Normalized artifact is missing: "
                    f"{relative_path}"
                ),
            )
            continue

        expected_hash = str(
            row.get(
                "normalized_sha256",
                "",
            )
        )

        actual_hash = (
            _sha256_file(
                processed_path
            )
        )

        if (
            expected_hash
            and actual_hash
            != expected_hash
        ):
            _append_issue(
                issues,
                level="ERROR",
                code=
                    "NORMALIZED_HASH_MISMATCH",
                message=(
                    f"{relative_path} SHA-256 does not "
                    "match the normalization manifest."
                ),
            )

        raw_row = raw_by_id.get(
            record_id
        )

        if raw_row is not None:
            expected_source_hash = str(
                row.get(
                    "source_payload_sha256",
                    "",
                )
            )

            raw_payload_hash = str(
                raw_row.get(
                    "payload_sha256",
                    "",
                )
            )

            if (
                expected_source_hash
                != raw_payload_hash
            ):
                _append_issue(
                    issues,
                    level="ERROR",
                    code=
                        "RAW_NORMALIZED_LINK_MISMATCH",
                    message=(
                        f"Normalized record {record_id} "
                        "does not link to the raw "
                        "payload SHA-256."
                    ),
                )

    missing_normalized = (
        set(
            raw_by_id
        )
        - set(
            normalized_by_id
        )
    )

    for record_id in sorted(
        missing_normalized
    ):
        _append_issue(
            issues,
            level="WARNING",
            code=
                "RAW_NOT_NORMALIZED",
            message=(
                f"Raw record {record_id} has no "
                "normalized manifest entry."
            ),
        )

    return (
        raw_rows,
        normalized_rows,
    )


def _required_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    dataset: str,
    scope: str,
    issues: list[AuditIssue],
) -> bool:
    missing = (
        required
        - set(
            frame.columns
        )
    )

    if missing:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "MISSING_COLUMNS",
            message=(
                f"{dataset} {scope} is missing "
                f"columns: {sorted(missing)}"
            ),
        )
        return False

    return True


def _audit_quotes(
    frame: pd.DataFrame,
    *,
    expected_symbols: set[str],
    scope: str,
    issues: list[AuditIssue],
) -> None:
    if frame.empty:
        _append_issue(
            issues,
            level="ERROR",
            code="EMPTY_QUOTES",
            message=(
                f"Quote table {scope} is empty."
            ),
        )
        return

    if not _required_columns(
        frame,
        {
            "symbol",
            "raw_record_id",
        },
        dataset="quotes",
        scope=scope,
        issues=issues,
    ):
        return

    symbols = set(
        frame[
            "symbol"
        ]
        .dropna()
        .astype(str)
        .str.upper()
    )

    missing = (
        expected_symbols
        - symbols
    )

    if missing:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "QUOTE_SYMBOL_COVERAGE",
            message=(
                f"Quote table {scope} is missing "
                f"symbols: {sorted(missing)}"
            ),
        )

    duplicates = int(
        frame[
            "symbol"
        ]
        .astype(str)
        .str.upper()
        .duplicated()
        .sum()
    )

    if duplicates:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "DUPLICATE_QUOTE_SYMBOL",
            message=(
                f"Quote table {scope} contains "
                f"{duplicates} duplicate symbol rows."
            ),
        )


def _audit_price_history(
    frame: pd.DataFrame,
    *,
    scope: str,
    issues: list[AuditIssue],
) -> None:
    required = {
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "datetime",
    }

    if frame.empty:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "EMPTY_PRICE_HISTORY",
            message=(
                f"Price-history table {scope} is empty."
            ),
        )
        return

    if not _required_columns(
        frame,
        required,
        dataset=
            "price_history",
        scope=
            scope,
        issues=
            issues,
    ):
        return

    duplicate_timestamps = int(
        frame[
            [
                "symbol",
                "datetime",
            ]
        ]
        .duplicated()
        .sum()
    )

    if duplicate_timestamps:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "DUPLICATE_CANDLE_TIMESTAMP",
            message=(
                f"Price-history table {scope} contains "
                f"{duplicate_timestamps} duplicate "
                "symbol/datetime rows."
            ),
        )

    numeric = frame[
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    null_numeric = int(
        numeric.isna().any(
            axis=1
        ).sum()
    )

    if null_numeric:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "INVALID_CANDLE_NUMERIC",
            message=(
                f"Price-history table {scope} contains "
                f"{null_numeric} rows with invalid "
                "OHLCV values."
            ),
        )

    finite = numeric.dropna()

    ohlc_violation = (
        (finite["low"] > finite["high"])
        | (finite["open"] < finite["low"])
        | (finite["open"] > finite["high"])
        | (finite["close"] < finite["low"])
        | (finite["close"] > finite["high"])
    )

    violation_count = int(
        ohlc_violation.sum()
    )

    if violation_count:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "OHLC_INVARIANT_VIOLATION",
            message=(
                f"Price-history table {scope} contains "
                f"{violation_count} OHLC invariant "
                "violations."
            ),
        )

    negative_volume = int(
        (
            finite[
                "volume"
            ]
            < 0
        ).sum()
    )

    if negative_volume:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "NEGATIVE_VOLUME",
            message=(
                f"Price-history table {scope} contains "
                f"{negative_volume} negative-volume rows."
            ),
        )

    datetime_values = pd.to_numeric(
        frame[
            "datetime"
        ],
        errors="coerce",
    )

    if datetime_values.isna().any():
        _append_issue(
            issues,
            level="ERROR",
            code=
                "INVALID_CANDLE_TIMESTAMP",
            message=(
                f"Price-history table {scope} contains "
                "non-numeric candle timestamps."
            ),
        )

    elif not datetime_values.is_monotonic_increasing:
        _append_issue(
            issues,
            level="WARNING",
            code=
                "CANDLES_NOT_CHRONOLOGICAL",
            message=(
                f"Price-history table {scope} is not "
                "ordered chronologically."
            ),
        )


def _audit_option_chain(
    frame: pd.DataFrame,
    *,
    scope: str,
    issues: list[AuditIssue],
) -> None:
    required = {
        "underlying_symbol",
        "put_call",
        "expiration_date",
        "strike",
    }

    if frame.empty:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "EMPTY_OPTION_CHAIN",
            message=(
                f"Option-chain table {scope} is empty."
            ),
        )
        return

    if not _required_columns(
        frame,
        required,
        dataset=
            "option_chains",
        scope=
            scope,
        issues=
            issues,
    ):
        return

    put_call = set(
        frame[
            "put_call"
        ]
        .dropna()
        .astype(str)
        .str.upper()
    )

    invalid_sides = (
        put_call
        - {
            "CALL",
            "PUT",
        }
    )

    if invalid_sides:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "INVALID_OPTION_SIDE",
            message=(
                f"Option-chain table {scope} contains "
                f"invalid sides: {sorted(invalid_sides)}"
            ),
        )

    missing_sides = (
        {
            "CALL",
            "PUT",
        }
        - put_call
    )

    if missing_sides:
        _append_issue(
            issues,
            level="WARNING",
            code=
                "OPTION_SIDE_COVERAGE",
            message=(
                f"Option-chain table {scope} has no "
                f"{sorted(missing_sides)} contracts."
            ),
        )

    strike = pd.to_numeric(
        frame[
            "strike"
        ],
        errors="coerce",
    )

    invalid_strike = int(
        (
            strike.isna()
            | (
                strike
                <= 0
            )
        ).sum()
    )

    if invalid_strike:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "INVALID_OPTION_STRIKE",
            message=(
                f"Option-chain table {scope} contains "
                f"{invalid_strike} invalid strikes."
            ),
        )

    expiration = pd.to_datetime(
        frame[
            "expiration_date"
        ],
        errors="coerce",
    )

    invalid_expiration = int(
        expiration.isna().sum()
    )

    if invalid_expiration:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "INVALID_OPTION_EXPIRATION",
            message=(
                f"Option-chain table {scope} contains "
                f"{invalid_expiration} invalid "
                "expiration dates."
            ),
        )

    if (
        "expiration_dte"
        in frame.columns
    ):
        dte = pd.to_numeric(
            frame[
                "expiration_dte"
            ],
            errors="coerce",
        )

        invalid_dte = int(
            (
                dte.notna()
                & (
                    dte
                    < 0
                )
            ).sum()
        )

        if invalid_dte:
            _append_issue(
                issues,
                level="ERROR",
                code=
                    "NEGATIVE_OPTION_DTE",
                message=(
                    f"Option-chain table {scope} contains "
                    f"{invalid_dte} negative DTE values."
                ),
            )

    if "symbol" in frame.columns:
        duplicate_contracts = int(
            frame[
                "symbol"
            ]
            .dropna()
            .astype(str)
            .duplicated()
            .sum()
        )
    else:
        key_columns = [
            column
            for column in (
                "underlying_symbol",
                "put_call",
                "expiration_date",
                "strike",
                "contract_index",
            )
            if column
            in frame.columns
        ]

        duplicate_contracts = int(
            frame[
                key_columns
            ]
            .duplicated()
            .sum()
        )

    if duplicate_contracts:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "DUPLICATE_OPTION_CONTRACT",
            message=(
                f"Option-chain table {scope} contains "
                f"{duplicate_contracts} duplicate "
                "contract rows."
            ),
        )


def _audit_expiration_chain(
    frame: pd.DataFrame,
    *,
    scope: str,
    issues: list[AuditIssue],
) -> None:
    if frame.empty:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "EMPTY_EXPIRATION_CHAIN",
            message=(
                f"Expiration-chain table {scope} is empty."
            ),
        )
        return

    if not _required_columns(
        frame,
        {
            "expirationDate",
        },
        dataset=
            "expiration_chains",
        scope=
            scope,
        issues=
            issues,
    ):
        return

    values = pd.to_datetime(
        frame[
            "expirationDate"
        ],
        errors="coerce",
    )

    invalid = int(
        values.isna().sum()
    )

    if invalid:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "INVALID_EXPIRATION_DATE",
            message=(
                f"Expiration-chain table {scope} contains "
                f"{invalid} invalid dates."
            ),
        )

    duplicate = int(
        frame[
            "expirationDate"
        ]
        .astype(str)
        .duplicated()
        .sum()
    )

    if duplicate:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "DUPLICATE_EXPIRATION",
            message=(
                f"Expiration-chain table {scope} contains "
                f"{duplicate} duplicate expirations."
            ),
        )


def _audit_instruments(
    frame: pd.DataFrame,
    *,
    expected_symbols: set[str],
    scope: str,
    issues: list[AuditIssue],
) -> None:
    if frame.empty:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "EMPTY_INSTRUMENTS",
            message=(
                f"Instrument table {scope} is empty."
            ),
        )
        return

    symbol_column = None

    if "symbol" in frame.columns:
        symbol_column = "symbol"

    else:
        candidates = [
            column
            for column in frame.columns
            if column.endswith(
                "__symbol"
            )
        ]

        if candidates:
            symbol_column = (
                candidates[0]
            )

    if symbol_column is None:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "INSTRUMENT_SYMBOL_COLUMN_MISSING",
            message=(
                f"Instrument table {scope} has no "
                "symbol column."
            ),
        )
        return

    symbols = set(
        frame[
            symbol_column
        ]
        .dropna()
        .astype(str)
        .str.upper()
    )

    missing = (
        expected_symbols
        - symbols
    )

    if missing:
        _append_issue(
            issues,
            level="WARNING",
            code=
                "INSTRUMENT_SYMBOL_COVERAGE",
            message=(
                f"Instrument table {scope} is missing "
                f"symbols: {sorted(missing)}"
            ),
        )


def _audit_market_hours(
    frame: pd.DataFrame,
    *,
    scope: str,
    issues: list[AuditIssue],
) -> None:
    if frame.empty:
        _append_issue(
            issues,
            level="ERROR",
            code=
                "EMPTY_MARKET_HOURS",
            message=(
                f"Market-hours table {scope} is empty."
            ),
        )
        return

    _required_columns(
        frame,
        {
            "market_group",
        },
        dataset=
            "market_hours",
        scope=
            scope,
        issues=
            issues,
    )


def _audit_movers(
    frame: pd.DataFrame,
    *,
    scope: str,
    issues: list[AuditIssue],
) -> None:
    if frame.empty:
        _append_issue(
            issues,
            level="WARNING",
            code=
                "EMPTY_MOVERS",
            message=(
                f"Movers table {scope} is empty."
            ),
        )
        return

    _required_columns(
        frame,
        {
            "mover_index",
        },
        dataset=
            "movers",
        scope=
            scope,
        issues=
            issues,
    )


def _audit_capture_run(
    *,
    summary_path: Path,
    raw_rows: list[
        dict[
            str,
            Any,
        ]
    ],
    normalized_rows: list[
        dict[
            str,
            Any,
        ]
    ],
    processed_root: Path,
    issues: list[AuditIssue],
) -> dict[str, Any]:
    summary = _read_json(
        summary_path,
        context=
            "Schwab capture-run summary",
    )

    artifacts = summary.get(
        "artifacts",
        [],
    )

    if not isinstance(
        artifacts,
        list,
    ):
        raise RuntimeError(
            "Capture-run artifacts must be a list."
        )

    declared_count = summary.get(
        "artifact_count"
    )

    if (
        declared_count is not None
        and int(
            declared_count
        )
        != len(
            artifacts
        )
    ):
        _append_issue(
            issues,
            level="ERROR",
            code=
                "RUN_ARTIFACT_COUNT_MISMATCH",
            message=(
                "Capture-run artifact_count does not "
                "match the artifacts list."
            ),
        )

    raw_ids = {
        str(
            row.get(
                "record_id"
            )
        )
        for row in raw_rows
    }

    normalized_ids = {
        str(
            row.get(
                "raw_record_id"
            )
        )
        for row in normalized_rows
    }

    expected_symbols = {
        str(
            symbol
        ).upper()
        for symbol in summary.get(
            "symbols",
            [],
        )
    }

    dataset_rows: dict[
        str,
        int,
    ] = defaultdict(
        int
    )

    dataset_artifacts: dict[
        str,
        int,
    ] = defaultdict(
        int
    )

    for artifact in artifacts:
        if not isinstance(
            artifact,
            Mapping,
        ):
            _append_issue(
                issues,
                level="ERROR",
                code=
                    "RUN_ARTIFACT_INVALID",
                message=(
                    "Capture-run artifact entry is "
                    "not an object."
                ),
            )
            continue

        dataset = str(
            artifact.get(
                "dataset",
                "",
            )
        )

        scope = str(
            artifact.get(
                "scope",
                "",
            )
        )

        raw = artifact.get(
            "raw",
            {},
        )

        normalized = artifact.get(
            "normalized",
            {},
        )

        if not isinstance(
            raw,
            Mapping,
        ) or not isinstance(
            normalized,
            Mapping,
        ):
            _append_issue(
                issues,
                level="ERROR",
                code=
                    "RUN_ARTIFACT_PROVENANCE_INVALID",
                message=(
                    f"Capture-run artifact {dataset} "
                    f"{scope} lacks raw/normalized "
                    "provenance."
                ),
            )
            continue

        raw_record_id = str(
            raw.get(
                "record_id",
                "",
            )
        )

        normalized_record_id = str(
            normalized.get(
                "raw_record_id",
                "",
            )
        )

        if raw_record_id not in raw_ids:
            _append_issue(
                issues,
                level="ERROR",
                code=
                    "RUN_RAW_RECORD_MISSING",
                message=(
                    f"Capture-run raw record "
                    f"{raw_record_id} is absent from "
                    "the raw manifest."
                ),
            )

        if (
            normalized_record_id
            not in normalized_ids
        ):
            _append_issue(
                issues,
                level="ERROR",
                code=
                    "RUN_NORMALIZED_RECORD_MISSING",
                message=(
                    f"Capture-run normalized record "
                    f"{normalized_record_id} is absent "
                    "from the normalized manifest."
                ),
            )

        if (
            raw_record_id
            != normalized_record_id
        ):
            _append_issue(
                issues,
                level="ERROR",
                code=
                    "RUN_RAW_NORMALIZED_ID_MISMATCH",
                message=(
                    f"Capture-run artifact {dataset} "
                    f"{scope} links different raw and "
                    "normalized record IDs."
                ),
            )

        relative_path = str(
            normalized.get(
                "relative_path",
                "",
            )
        )

        if not relative_path:
            _append_issue(
                issues,
                level="ERROR",
                code=
                    "RUN_NORMALIZED_PATH_MISSING",
                message=(
                    f"Capture-run artifact {dataset} "
                    f"{scope} lacks normalized path."
                ),
            )
            continue

        processed_path = (
            Path(
                processed_root
            )
            / relative_path
        )

        if not processed_path.is_file():
            _append_issue(
                issues,
                level="ERROR",
                code=
                    "RUN_NORMALIZED_FILE_MISSING",
                message=(
                    f"Capture-run normalized file "
                    f"is missing: {relative_path}"
                ),
            )
            continue

        frame = (
            _load_processed_frame(
                processed_root,
                relative_path,
            )
        )

        dataset_rows[
            dataset
        ] += int(
            len(
                frame
            )
        )

        dataset_artifacts[
            dataset
        ] += 1

        if dataset == "quotes":
            _audit_quotes(
                frame,
                expected_symbols=
                    expected_symbols,
                scope=
                    scope,
                issues=
                    issues,
            )

        elif dataset == "price_history":
            _audit_price_history(
                frame,
                scope=
                    scope,
                issues=
                    issues,
            )

        elif dataset == "option_chains":
            _audit_option_chain(
                frame,
                scope=
                    scope,
                issues=
                    issues,
            )

        elif dataset == "expiration_chains":
            _audit_expiration_chain(
                frame,
                scope=
                    scope,
                issues=
                    issues,
            )

        elif dataset == "instruments":
            _audit_instruments(
                frame,
                expected_symbols=
                    expected_symbols,
                scope=
                    scope,
                issues=
                    issues,
            )

        elif dataset == "market_hours":
            _audit_market_hours(
                frame,
                scope=
                    scope,
                issues=
                    issues,
            )

        elif dataset == "movers":
            _audit_movers(
                frame,
                scope=
                    scope,
                issues=
                    issues,
            )

        else:
            _append_issue(
                issues,
                level="WARNING",
                code=
                    "UNKNOWN_RUN_DATASET",
                message=(
                    "Capture-run contains unrecognized "
                    f"dataset {dataset!r}."
                ),
            )

    return {
        "run_id":
            str(
                summary.get(
                    "run_id",
                    "",
                )
            ),

        "started_at_utc":
            str(
                summary.get(
                    "started_at_utc",
                    "",
                )
            ),

        "completed_at_utc":
            str(
                summary.get(
                    "completed_at_utc",
                    "",
                )
            ),

        "symbols":
            sorted(
                expected_symbols
            ),

        "artifact_count":
            len(
                artifacts
            ),

        "dataset_artifacts":
            dict(
                sorted(
                    dataset_artifacts.items()
                )
            ),

        "dataset_rows":
            dict(
                sorted(
                    dataset_rows.items()
                )
            ),

        "summary_path":
            str(
                summary_path
            ),
    }


def audit_schwab_market_data(
    *,
    raw_root: Path = DEFAULT_RAW_ROOT,
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
    run_summary: Path | None = None,
) -> dict[str, Any]:
    """
    Audit Schwab raw/normalized provenance and validate
    the most recent capture-run research tables.
    """
    raw_root = Path(
        raw_root
    )

    processed_root = Path(
        processed_root
    )

    issues: list[
        AuditIssue
    ] = []

    (
        raw_rows,
        normalized_rows,
    ) = _audit_manifests(
        raw_root=
            raw_root,

        processed_root=
            processed_root,

        issues=
            issues,
    )

    if run_summary is None:
        resolved_summary = (
            _latest_capture_summary(
                processed_root
            )
        )

    else:
        resolved_summary = Path(
            run_summary
        )

    run_result = None

    if resolved_summary is None:
        _append_issue(
            issues,
            level="WARNING",
            code=
                "NO_CAPTURE_RUN_SUMMARY",
            message=(
                "No capture-run summary was found. "
                "Manifest integrity was audited only."
            ),
        )

    else:
        run_result = (
            _audit_capture_run(
                summary_path=
                    resolved_summary,

                raw_rows=
                    raw_rows,

                normalized_rows=
                    normalized_rows,

                processed_root=
                    processed_root,

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

    result = {
        "status":
            (
                "PASS"
                if error_count == 0
                else "FAIL"
            ),

        "raw_manifest_records":
            len(
                raw_rows
            ),

        "normalized_manifest_records":
            len(
                normalized_rows
            ),

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
                {
                    "level":
                        issue.level,

                    "code":
                        issue.code,

                    "message":
                        issue.message,
                }
                for issue in issues
            ],

        "run":
            run_result,
    }

    return result


def _print_result(
    result: Mapping[
        str,
        Any,
    ],
) -> None:
    print()
    print("=" * 88)
    print(
        "SCHWAB MARKET DATA AUDIT"
    )
    print("=" * 88)

    print(
        "Raw manifest records:",
        result[
            "raw_manifest_records"
        ],
    )

    print(
        "Normalized manifest records:",
        result[
            "normalized_manifest_records"
        ],
    )

    run = result.get(
        "run"
    )

    if isinstance(
        run,
        Mapping,
    ):
        print()
        print(
            "LATEST CAPTURE"
        )
        print(
            "Run ID:",
            run.get(
                "run_id"
            ),
        )
        print(
            "Started UTC:",
            run.get(
                "started_at_utc"
            ),
        )
        print(
            "Symbols:",
            ", ".join(
                run.get(
                    "symbols",
                    [],
                )
            ),
        )
        print(
            "Artifacts:",
            run.get(
                "artifact_count"
            ),
        )

        print()
        print(
            "DATASET COVERAGE"
        )

        dataset_artifacts = (
            run.get(
                "dataset_artifacts",
                {},
            )
        )

        dataset_rows = (
            run.get(
                "dataset_rows",
                {},
            )
        )

        all_datasets = sorted(
            set(
                dataset_artifacts
            )
            | set(
                dataset_rows
            )
        )

        for dataset in all_datasets:
            print(
                f"{dataset:<22} "
                f"artifacts="
                f"{dataset_artifacts.get(dataset, 0):<3} "
                f"rows="
                f"{dataset_rows.get(dataset, 0)}"
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

    issues = result.get(
        "issues",
        [],
    )

    if issues:
        print()
        print(
            "ISSUES"
        )

        for issue in issues:
            print(
                f"[{issue['level']}] "
                f"{issue['code']}: "
                f"{issue['message']}"
            )

    print()
    print(
        "AUDIT RESULT:",
        result[
            "status"
        ],
    )
    print("=" * 88)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit permanent Schwab raw and normalized "
            "market-data research archives."
        )
    )

    parser.add_argument(
        "--raw-root",
        type=Path,
        default=
            DEFAULT_RAW_ROOT,
        help=(
            "Raw Schwab archive root."
        ),
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
        "--json",
        action="store_true",
        help=(
            "Print the audit result as JSON instead "
            "of the human-readable report."
        ),
    )

    return parser


def main() -> None:
    args = (
        build_parser()
        .parse_args()
    )

    result = (
        audit_schwab_market_data(
            raw_root=
                args.raw_root,

            processed_root=
                args.processed_root,

            run_summary=
                args.run_summary,
        )
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
        != "PASS"
    ):
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
