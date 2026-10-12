from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterator, Mapping

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


from config.scanner_handoff import (
    CANDIDATE_COLUMNS,
    EXPECTED_SCANNER_ALPHA_SPEC_VERSION,
    EXPECTED_SCANNER_PROVIDER,
    HANDOFF_CONTRACT_VERSION,
    MANIFEST_REQUIRED_FIELDS,
    PRIMARY_SIGNAL_NAME,
    SELECTION_POLICY,
)
from config.settings import (
    DATA_DIR,
)


DEFAULT_HISTORY_PATH = (
    DATA_DIR
    / "history"
    / "scanner"
    / "scanner_candidate_history_v1.csv"
)

CANDIDATE_HISTORY_SCHEMA_VERSION = "1.0.0"
CANDIDATE_HISTORY_TABLE_NAME = "scanner_candidate_history_v1"

PRIMARY_KEY_COLUMNS = (
    "scanner_snapshot_sha256",
    "symbol",
    "handoff_contract_version",
)

HISTORY_PROVENANCE_COLUMNS = (
    "candidate_history_schema_version",
    "source_candidate_csv_sha256",
    "source_candidate_manifest_sha256",
)

_SHA256_PATTERN = re.compile(
    r"^[0-9a-f]{64}$"
)


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


def _sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda:
                handle.read(
                    1024
                    * 1024
                ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def _require_sha256(
    value: Any,
    *,
    field_name: str,
) -> str:
    normalized = str(
        value
    ).strip().lower()

    if not _SHA256_PATTERN.fullmatch(
        normalized
    ):
        raise RuntimeError(
            f"{field_name} must be a lowercase 64-character "
            "SHA-256 hex digest."
        )

    return normalized


def _parse_utc_timestamp(
    value: Any,
    *,
    field_name: str,
) -> pd.Timestamp:
    timestamp = pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )

    if pd.isna(
        timestamp
    ):
        raise RuntimeError(
            f"{field_name} contains an invalid timestamp: "
            f"{value!r}"
        )

    return timestamp


def _parse_session_date(
    value: Any,
) -> str:
    text = str(
        value
    ).strip()

    try:
        parsed = pd.Timestamp(
            text
        )

    except Exception as exc:
        raise RuntimeError(
            f"as_of_session is invalid: {value!r}"
        ) from exc

    if parsed.strftime(
        "%Y-%m-%d"
    ) != text:
        raise RuntimeError(
            "as_of_session must use YYYY-MM-DD format."
        )

    return text


def _coerce_bool_series(
    series: pd.Series,
    *,
    field_name: str,
) -> pd.Series:
    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        "true": True,
        "false": False,
        "True": True,
        "False": False,
        "1": True,
        "0": False,
    }

    normalized = series.map(
        mapping
    )

    if normalized.isna().any():
        bad = (
            series.loc[
                normalized.isna()
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise RuntimeError(
            f"{field_name} contains invalid boolean values: "
            f"{bad}"
        )

    return normalized.astype(
        bool
    )


def _validate_manifest(
    manifest: Mapping[
        str,
        Any,
    ],
    *,
    candidate_csv_sha256: str,
) -> dict[str, Any]:
    missing = [
        field
        for field
        in MANIFEST_REQUIRED_FIELDS
        if field
        not in manifest
    ]

    if missing:
        raise RuntimeError(
            "Scanner candidate manifest is missing required "
            f"fields: {missing}"
        )

    handoff_version = str(
        manifest[
            "handoff_contract_version"
        ]
    ).strip()

    alpha_spec = str(
        manifest[
            "scanner_alpha_spec_version"
        ]
    ).strip()

    provider = str(
        manifest[
            "scanner_provider"
        ]
    ).strip().lower()

    selection_policy = str(
        manifest[
            "selection_policy"
        ]
    ).strip()

    if (
        handoff_version
        != HANDOFF_CONTRACT_VERSION
    ):
        raise RuntimeError(
            "Unsupported scanner handoff contract. "
            f"Expected {HANDOFF_CONTRACT_VERSION!r}; "
            f"received {handoff_version!r}."
        )

    if (
        alpha_spec
        != EXPECTED_SCANNER_ALPHA_SPEC_VERSION
    ):
        raise RuntimeError(
            "Unsupported scanner alpha specification. "
            "Expected "
            f"{EXPECTED_SCANNER_ALPHA_SPEC_VERSION!r}; "
            f"received {alpha_spec!r}."
        )

    if (
        provider
        != EXPECTED_SCANNER_PROVIDER
    ):
        raise RuntimeError(
            "Unsupported scanner provider. "
            f"Expected {EXPECTED_SCANNER_PROVIDER!r}; "
            f"received {provider!r}."
        )

    if (
        selection_policy
        != SELECTION_POLICY
    ):
        raise RuntimeError(
            "Unsupported scanner selection policy. "
            f"Expected {SELECTION_POLICY!r}; "
            f"received {selection_policy!r}."
        )

    immutable = manifest[
        "immutable"
    ]

    if immutable is not True:
        raise RuntimeError(
            "Scanner candidate manifest must declare "
            "immutable=true."
        )

    try:
        candidate_count = int(
            manifest[
                "candidate_count"
            ]
        )

        candidate_limit = int(
            manifest[
                "candidate_limit"
            ]
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise RuntimeError(
            "candidate_count and candidate_limit must be "
            "integers."
        ) from exc

    if candidate_count <= 0:
        raise RuntimeError(
            "candidate_count must be positive."
        )

    if candidate_limit <= 0:
        raise RuntimeError(
            "candidate_limit must be positive."
        )

    if candidate_count > candidate_limit:
        raise RuntimeError(
            "candidate_count cannot exceed candidate_limit."
        )

    as_of_session = _parse_session_date(
        manifest[
            "as_of_session"
        ]
    )

    generated_at = _parse_utc_timestamp(
        manifest[
            "generated_at_utc"
        ],
        field_name=
            "generated_at_utc",
    )

    scanner_snapshot_sha256 = (
        _require_sha256(
            manifest[
                "scanner_snapshot_sha256"
            ],
            field_name=
                "scanner_snapshot_sha256",
        )
    )

    source_alpha_manifest_sha256 = (
        _require_sha256(
            manifest[
                "source_alpha_manifest_sha256"
            ],
            field_name=
                "source_alpha_manifest_sha256",
        )
    )

    expected_candidate_sha = (
        _require_sha256(
            manifest[
                "candidate_csv_sha256"
            ],
            field_name=
                "candidate_csv_sha256",
        )
    )

    if (
        expected_candidate_sha
        != candidate_csv_sha256
    ):
        raise RuntimeError(
            "Candidate CSV SHA-256 does not match the "
            "scanner candidate manifest."
        )

    return {
        "handoff_contract_version":
            handoff_version,

        "scanner_alpha_spec_version":
            alpha_spec,

        "scanner_provider":
            provider,

        "as_of_session":
            as_of_session,

        "generated_at_utc":
            generated_at.isoformat(),

        "scanner_snapshot_sha256":
            scanner_snapshot_sha256,

        "source_alpha_manifest_sha256":
            source_alpha_manifest_sha256,

        "candidate_count":
            candidate_count,

        "candidate_limit":
            candidate_limit,

        "selection_policy":
            selection_policy,
    }


def _validate_candidate_frame(
    frame: pd.DataFrame,
    *,
    manifest: Mapping[
        str,
        Any,
    ],
) -> pd.DataFrame:
    actual_columns = tuple(
        str(
            column
        )
        for column
        in frame.columns
    )

    if (
        actual_columns
        != CANDIDATE_COLUMNS
    ):
        missing = [
            column
            for column
            in CANDIDATE_COLUMNS
            if column
            not in actual_columns
        ]

        extra = [
            column
            for column
            in actual_columns
            if column
            not in CANDIDATE_COLUMNS
        ]

        raise RuntimeError(
            "Scanner candidate CSV does not match the "
            "frozen handoff-v1 column contract. "
            f"Missing={missing}; extra={extra}; "
            "column order must also match the contract."
        )

    if frame.empty:
        raise RuntimeError(
            "Scanner candidate CSV is empty."
        )

    if (
        len(
            frame
        )
        != int(
            manifest[
                "candidate_count"
            ]
        )
    ):
        raise RuntimeError(
            "Candidate CSV row count does not match "
            "manifest candidate_count."
        )

    result = frame.copy()

    constant_fields = (
        "handoff_contract_version",
        "scanner_alpha_spec_version",
        "scanner_provider",
        "as_of_session",
        "generated_at_utc",
        "scanner_snapshot_sha256",
        "source_alpha_manifest_sha256",
    )

    for field in constant_fields:
        if field not in manifest:
            raise RuntimeError(
                f"Validated manifest has no {field!r} value."
            )

        expected = str(
            manifest[
                field
            ]
        ).strip()

        if field == "scanner_provider":
            actual = (
                result[
                    field
                ]
                .astype(str)
                .str.strip()
                .str.lower()
            )

        elif field == "generated_at_utc":
            parsed = pd.to_datetime(
                result[
                    field
                ],
                errors="coerce",
                utc=True,
            )

            if parsed.isna().any():
                raise RuntimeError(
                    "Candidate rows contain invalid "
                    "generated_at_utc values."
                )

            actual = parsed.map(
                lambda value:
                    value.isoformat()
            )

        else:
            actual = (
                result[
                    field
                ]
                .astype(str)
                .str.strip()
            )

        if not actual.eq(
            expected
        ).all():
            raise RuntimeError(
                f"Candidate rows do not all match manifest "
                f"field {field!r}."
            )

    result[
        "symbol"
    ] = (
        result[
            "symbol"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if (
        result[
            "symbol"
        ]
        .eq("")
        .any()
    ):
        raise RuntimeError(
            "Candidate CSV contains an empty symbol."
        )

    if (
        result[
            "symbol"
        ]
        .duplicated()
        .any()
    ):
        duplicates = sorted(
            result.loc[
                result[
                    "symbol"
                ].duplicated(
                    keep=False
                ),
                "symbol",
            ].unique()
        )

        raise RuntimeError(
            "Candidate CSV contains duplicate symbols: "
            f"{duplicates}"
        )

    result[
        "scanner_eligible"
    ] = _coerce_bool_series(
        result[
            "scanner_eligible"
        ],
        field_name=
            "scanner_eligible",
    )

    if not result[
        "scanner_eligible"
    ].all():
        raise RuntimeError(
            "Handoff v1 may contain only scanner-eligible "
            "candidates."
        )

    primary_signal_names = (
        result[
            "primary_signal_name"
        ]
        .astype(str)
        .str.strip()
    )

    if not primary_signal_names.eq(
        PRIMARY_SIGNAL_NAME
    ).all():
        raise RuntimeError(
            "Candidate rows must use primary signal "
            f"{PRIMARY_SIGNAL_NAME!r}."
        )

    ranks = pd.to_numeric(
        result[
            "scanner_rank"
        ],
        errors="coerce",
    )

    if (
        ranks.isna().any()
        or (
            ranks
            <= 0
        ).any()
        or not (
            ranks
            .map(
                lambda value:
                    float(
                        value
                    ).is_integer()
            )
            .all()
        )
    ):
        raise RuntimeError(
            "scanner_rank must contain positive integers."
        )

    ranks = ranks.astype(
        int
    )

    if ranks.duplicated().any():
        raise RuntimeError(
            "scanner_rank must be unique within a handoff."
        )

    expected_ranks = list(
        range(
            1,
            len(
                result
            )
            + 1,
        )
    )

    if ranks.tolist() != expected_ranks:
        raise RuntimeError(
            "Handoff candidates must be ordered as the "
            "contiguous top-ranked alpha-complete set "
            f"1..{len(result)}."
        )

    result[
        "scanner_rank"
    ] = ranks

    numeric_required = (
        "scanner_score",
        "scanner_percentile",
        "close",
        "primary_signal_value",
    )

    for field in numeric_required:
        numeric = pd.to_numeric(
            result[
                field
            ],
            errors="coerce",
        )

        finite = numeric.map(
            lambda value:
                (
                    pd.notna(
                        value
                    )
                    and math.isfinite(
                        float(
                            value
                        )
                    )
                )
        )

        if not finite.all():
            raise RuntimeError(
                f"{field} must contain only finite numeric "
                "values."
            )

        result[
            field
        ] = numeric.astype(
            float
        )

    if (
        result[
            "close"
        ]
        <= 0
    ).any():
        raise RuntimeError(
            "Candidate close values must be positive."
        )

    if (
        (
            result[
                "scanner_percentile"
            ]
            < 0
        )
        | (
            result[
                "scanner_percentile"
            ]
            > 100
        )
    ).any():
        raise RuntimeError(
            "scanner_percentile must be within [0, 100]."
        )

    context_numeric = pd.to_numeric(
        result[
            "ema_50_to_200"
        ],
        errors="coerce",
    )

    invalid_context = (
        result[
            "ema_50_to_200"
        ].notna()
        & context_numeric.isna()
    )

    if invalid_context.any():
        raise RuntimeError(
            "ema_50_to_200 must be numeric or missing."
        )

    result[
        "ema_50_to_200"
    ] = context_numeric

    source_timestamps = pd.to_datetime(
        result[
            "source_timestamp_utc"
        ],
        errors="coerce",
        utc=True,
    )

    if source_timestamps.isna().any():
        raise RuntimeError(
            "source_timestamp_utc contains invalid "
            "timestamps."
        )

    source_sessions = set(
        source_timestamps.dt.strftime(
            "%Y-%m-%d"
        )
    )

    if source_sessions != {
        manifest[
            "as_of_session"
        ]
    }:
        raise RuntimeError(
            "Candidate source timestamps do not match "
            "manifest as_of_session."
        )

    generated_at = _parse_utc_timestamp(
        manifest[
            "generated_at_utc"
        ],
        field_name=
            "generated_at_utc",
    )

    if (
        source_timestamps
        > generated_at
    ).any():
        raise RuntimeError(
            "Candidate source timestamp cannot be later "
            "than generated_at_utc."
        )

    result[
        "source_timestamp_utc"
    ] = source_timestamps.map(
        lambda value:
            value.isoformat()
    )

    result[
        "generated_at_utc"
    ] = generated_at.isoformat()

    return result


def _sort_history(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    return frame.sort_values(
        [
            "as_of_session",
            "scanner_snapshot_sha256",
            "scanner_rank",
            "symbol",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )


def _key_index(
    frame: pd.DataFrame,
) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(
        frame[
            list(
                PRIMARY_KEY_COLUMNS
            )
        ].astype(
            str
        )
    )


def _assert_matching_rows(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
) -> None:
    left = _sort_history(
        existing.copy()
    )

    right = _sort_history(
        incoming.copy()
    )

    try:
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_like=False,
            rtol=1e-12,
            atol=1e-12,
        )

    except AssertionError as exc:
        raise RuntimeError(
            "Candidate history already contains one or more "
            "incoming primary keys with different content."
        ) from exc


@contextmanager
def _history_lock(
    history_path: Path,
) -> Iterator[None]:
    lock_path = Path(
        str(
            history_path
        )
        + ".lock"
    )

    lock_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        with lock_path.open(
            "x",
            encoding="utf-8",
        ) as handle:
            handle.write(
                f"pid={os.getpid()}\n"
            )

    except FileExistsError as exc:
        raise RuntimeError(
            "Scanner candidate history lock already exists: "
            f"{lock_path}. Another ingest may be active, "
            "or a prior process may have exited unexpectedly."
        ) from exc

    try:
        yield

    finally:
        try:
            lock_path.unlink()

        except FileNotFoundError:
            pass


def _atomic_write_csv(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, temp_name = tempfile.mkstemp(
        prefix=
            path.name
            + ".",

        suffix=
            ".tmp",

        dir=
            str(
                path.parent
            ),
    )

    temp_path = Path(
        temp_name
    )

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            frame.to_csv(
                handle,
                index=False,
                lineterminator="\n",
            )

            handle.flush()
            os.fsync(
                handle.fileno()
            )

        os.replace(
            temp_path,
            path,
        )

    except Exception:
        try:
            temp_path.unlink()

        except FileNotFoundError:
            pass

        raise


def ingest_scanner_candidates(
    *,
    candidate_csv: Path,
    candidate_manifest: Path,
    history_path: Path = DEFAULT_HISTORY_PATH,
    dry_run: bool = False,
) -> dict[str, Any]:
    candidate_csv = Path(
        candidate_csv
    )

    candidate_manifest = Path(
        candidate_manifest
    )

    history_path = Path(
        history_path
    )

    if not candidate_csv.is_file():
        raise FileNotFoundError(
            f"Scanner candidate CSV is missing: {candidate_csv}"
        )

    if not candidate_manifest.is_file():
        raise FileNotFoundError(
            "Scanner candidate manifest is missing: "
            f"{candidate_manifest}"
        )

    candidate_csv_sha256 = (
        _sha256_file(
            candidate_csv
        )
    )

    candidate_manifest_sha256 = (
        _sha256_file(
            candidate_manifest
        )
    )

    raw_manifest = _read_json(
        candidate_manifest,
        context=
            "Scanner candidate manifest",
    )

    manifest = _validate_manifest(
        raw_manifest,
        candidate_csv_sha256=
            candidate_csv_sha256,
    )

    frame = pd.read_csv(
        candidate_csv,
        low_memory=False,
    )

    incoming = _validate_candidate_frame(
        frame,
        manifest=
            manifest,
    )

    for column in reversed(
        HISTORY_PROVENANCE_COLUMNS
    ):
        value = {
            "candidate_history_schema_version":
                CANDIDATE_HISTORY_SCHEMA_VERSION,

            "source_candidate_csv_sha256":
                candidate_csv_sha256,

            "source_candidate_manifest_sha256":
                candidate_manifest_sha256,
        }[
            column
        ]

        incoming.insert(
            0,
            column,
            value,
        )

    incoming = _sort_history(
        incoming
    )

    if incoming.duplicated(
        subset=list(
            PRIMARY_KEY_COLUMNS
        ),
        keep=False,
    ).any():
        raise RuntimeError(
            "Incoming scanner candidate batch contains "
            "duplicate history keys."
        )

    if dry_run:
        existing_rows = 0

        if history_path.is_file():
            existing_rows = int(
                len(
                    pd.read_csv(
                        history_path,
                        low_memory=False,
                    )
                )
            )

        return {
            **manifest,

            "candidate_history_table_name":
                CANDIDATE_HISTORY_TABLE_NAME,

            "candidate_history_schema_version":
                CANDIDATE_HISTORY_SCHEMA_VERSION,

            "candidate_csv":
                str(
                    candidate_csv
                ),

            "candidate_manifest":
                str(
                    candidate_manifest
                ),

            "history_path":
                str(
                    history_path
                ),

            "status":
                "DRY_RUN",

            "incoming_rows":
                int(
                    len(
                        incoming
                    )
                ),

            "appended_rows":
                0,

            "existing_rows_before":
                existing_rows,

            "history_rows_after":
                existing_rows,

            "history_sha256":
                (
                    _sha256_file(
                        history_path
                    )
                    if history_path.is_file()
                    else None
                ),
        }

    with _history_lock(
        history_path
    ):
        if history_path.is_file():
            existing = pd.read_csv(
                history_path,
                low_memory=False,
            )

            if list(
                existing.columns
            ) != list(
                incoming.columns
            ):
                raise RuntimeError(
                    "Scanner candidate history columns do "
                    "not match the frozen history-v1 schema. "
                    "Refusing to ingest across schema drift."
                )

            if existing.duplicated(
                subset=list(
                    PRIMARY_KEY_COLUMNS
                ),
                keep=False,
            ).any():
                raise RuntimeError(
                    "Existing scanner candidate history "
                    "contains duplicate primary keys."
                )

        else:
            existing = pd.DataFrame(
                columns=
                    incoming.columns
            )

        rows_before = int(
            len(
                existing
            )
        )

        incoming_index = _key_index(
            incoming
        )

        if existing.empty:
            overlap_mask = pd.Series(
                False,
                index=incoming.index,
            )

        else:
            existing_index = _key_index(
                existing
            )

            overlap_mask = pd.Series(
                incoming_index.isin(
                    existing_index
                ),
                index=incoming.index,
            )

        if overlap_mask.any():
            overlap_incoming = (
                incoming.loc[
                    overlap_mask
                ].copy()
            )

            overlap_keys = _key_index(
                overlap_incoming
            )

            overlap_existing = existing.loc[
                _key_index(
                    existing
                ).isin(
                    overlap_keys
                )
            ].copy()

            _assert_matching_rows(
                overlap_existing,
                overlap_incoming,
            )

        new_rows = incoming.loc[
            ~overlap_mask
        ].copy()

        if new_rows.empty:
            status = "EXISTS"
            combined = existing.copy()

        else:
            status = "APPENDED"

            combined = pd.concat(
                [
                    existing,
                    new_rows,
                ],
                ignore_index=True,
                sort=False,
            )

            combined = _sort_history(
                combined
            )

            if combined.duplicated(
                subset=list(
                    PRIMARY_KEY_COLUMNS
                ),
                keep=False,
            ).any():
                raise RuntimeError(
                    "Scanner candidate ingest would create "
                    "duplicate history keys."
                )

            _atomic_write_csv(
                combined,
                history_path,
            )

        rows_after = int(
            len(
                combined
            )
        )

    history_sha256 = (
        _sha256_file(
            history_path
        )
        if history_path.is_file()
        else None
    )

    return {
        **manifest,

        "candidate_history_table_name":
            CANDIDATE_HISTORY_TABLE_NAME,

        "candidate_history_schema_version":
            CANDIDATE_HISTORY_SCHEMA_VERSION,

        "candidate_csv":
            str(
                candidate_csv
            ),

        "candidate_manifest":
            str(
                candidate_manifest
            ),

        "history_path":
            str(
                history_path
            ),

        "status":
            status,

        "incoming_rows":
            int(
                len(
                    incoming
                )
            ),

        "appended_rows":
            int(
                len(
                    new_rows
                )
            ),

        "existing_rows_before":
            rows_before,

        "history_rows_after":
            rows_after,

        "history_sha256":
            history_sha256,
    }


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
        "SCANNER CANDIDATE HANDOFF INGEST"
    )
    print(
        "=" * 88
    )

    print(
        "Handoff contract:",
        result[
            "handoff_contract_version"
        ],
    )

    print(
        "Scanner alpha spec:",
        result[
            "scanner_alpha_spec_version"
        ],
    )

    print(
        "Provider:",
        result[
            "scanner_provider"
        ],
    )

    print(
        "As-of session:",
        result[
            "as_of_session"
        ],
    )

    print(
        "Scanner snapshot SHA-256:",
        result[
            "scanner_snapshot_sha256"
        ],
    )

    print(
        "History schema:",
        result[
            "candidate_history_schema_version"
        ],
    )

    print(
        "Status:",
        result[
            "status"
        ],
    )

    print(
        "Incoming rows:",
        result[
            "incoming_rows"
        ],
    )

    print(
        "Appended rows:",
        result[
            "appended_rows"
        ],
    )

    print(
        "History rows:",
        (
            f"{result['existing_rows_before']} -> "
            f"{result['history_rows_after']}"
        ),
    )

    print(
        "History SHA-256:",
        result[
            "history_sha256"
        ],
    )

    print()
    print(
        "History path:",
        result[
            "history_path"
        ],
    )

    print(
        "=" * 88
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and append an immutable scanner "
            "candidate handoff to optimizer candidate "
            "history."
        )
    )

    parser.add_argument(
        "--candidate-csv",
        type=Path,
        required=True,
        help=
            "Scanner candidate handoff CSV.",
    )

    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        required=True,
        help=
            "Scanner candidate handoff manifest JSON.",
    )

    parser.add_argument(
        "--history-path",
        type=Path,
        default=
            DEFAULT_HISTORY_PATH,
        help=
            "Canonical scanner candidate-history CSV.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=
            "Validate the handoff without writing history.",
    )

    return parser


def main() -> None:
    args = (
        build_parser()
        .parse_args()
    )

    result = ingest_scanner_candidates(
        candidate_csv=
            args.candidate_csv,

        candidate_manifest=
            args.candidate_manifest,

        history_path=
            args.history_path,

        dry_run=
            args.dry_run,
    )

    _print_result(
        result
    )


if __name__ == "__main__":
    main()
