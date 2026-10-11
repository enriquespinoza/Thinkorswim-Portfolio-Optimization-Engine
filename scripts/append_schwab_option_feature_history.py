from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
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


from config.settings import (
    FEATURE_DATA_DIR,
)


DEFAULT_OPTION_FEATURE_ROOT = (
    FEATURE_DATA_DIR
    / "schwab"
    / "options"
)

DEFAULT_HISTORY_PATH = (
    DEFAULT_OPTION_FEATURE_ROOT
    / "history"
    / "option_feature_history_v1.csv"
)

HISTORY_SCHEMA_VERSION = "1.0.0"
HISTORY_TABLE_NAME = "option_feature_history_v1"
EXPECTED_FEATURE_SCHEMA_VERSION = "1.2.0"
EXPECTED_QUALITY_POLICY_VERSION = "1.0.0"

PRIMARY_KEY_COLUMNS = (
    "capture_timestamp_utc",
    "run_id",
    "underlying_symbol",
    "feature_schema_version",
    "quality_policy_version",
)

HISTORY_PROVENANCE_COLUMNS = (
    "history_schema_version",
    "capture_timestamp_utc",
    "run_id",
    "input_validation_id",
    "source_validated_features_sha256",
    "source_validation_metadata_sha256",
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


def _latest_validation_metadata(
    feature_root: Path,
) -> Path:
    candidates = sorted(
        Path(
            feature_root
        ).glob(
            "*__option_feature_validation_metadata.json"
        )
    )

    if not candidates:
        raise FileNotFoundError(
            "No Schwab option-feature validation metadata "
            f"files found in {feature_root}"
        )

    return candidates[-1]


def _resolve_path(
    value: Path,
    *,
    fallback_parent: Path | None = None,
) -> Path:
    path = Path(
        value
    )

    if path.is_file():
        return path

    if not path.is_absolute():
        project_candidate = (
            PROJECT_ROOT
            / path
        )

        if project_candidate.is_file():
            return project_candidate

    if fallback_parent is not None:
        fallback = (
            Path(
                fallback_parent
            )
            / path.name
        )

        if fallback.is_file():
            return fallback

    return path


def _resolve_validation_metadata(
    *,
    feature_root: Path,
    validation_metadata: Path | None,
) -> Path:
    if validation_metadata is None:
        return _latest_validation_metadata(
            feature_root
        )

    return _resolve_path(
        validation_metadata,
        fallback_parent=
            feature_root,
    )


def _metadata_output(
    metadata: Mapping[
        str,
        Any,
    ],
    *,
    name: str,
) -> dict[str, Any]:
    outputs = metadata.get(
        "outputs"
    )

    if not isinstance(
        outputs,
        Mapping,
    ):
        raise RuntimeError(
            "Validation metadata has no outputs object."
        )

    output = outputs.get(
        name
    )

    if not isinstance(
        output,
        Mapping,
    ):
        raise RuntimeError(
            "Validation metadata has no "
            f"{name!r} output."
        )

    return dict(
        output
    )


def _parse_capture_timestamp(
    value: Any,
) -> str:
    timestamp = pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )

    if pd.isna(
        timestamp
    ):
        raise RuntimeError(
            "Source feature metadata contains an invalid "
            f"run_started_at_utc value: {value!r}"
        )

    return timestamp.isoformat()


def _assert_version_contract(
    *,
    feature_schema_version: str,
    quality_policy_version: str,
) -> None:
    if (
        feature_schema_version
        != EXPECTED_FEATURE_SCHEMA_VERSION
    ):
        raise RuntimeError(
            "History v1 accepts only Schwab option "
            f"feature schema {EXPECTED_FEATURE_SCHEMA_VERSION}; "
            f"received {feature_schema_version!r}."
        )

    if (
        quality_policy_version
        != EXPECTED_QUALITY_POLICY_VERSION
    ):
        raise RuntimeError(
            "History v1 accepts only Schwab option "
            f"quality policy {EXPECTED_QUALITY_POLICY_VERSION}; "
            f"received {quality_policy_version!r}."
        )


def _load_validated_batch(
    *,
    feature_root: Path,
    validation_metadata_path: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    validation_metadata = _read_json(
        validation_metadata_path,
        context=
            "Schwab option-feature validation metadata",
    )

    run_id = str(
        validation_metadata.get(
            "run_id",
            "",
        )
    ).strip()

    feature_schema_version = str(
        validation_metadata.get(
            "feature_schema_version",
            "",
        )
    ).strip()

    quality_policy_version = str(
        validation_metadata.get(
            "quality_policy_version",
            "",
        )
    ).strip()

    if not run_id:
        raise RuntimeError(
            "Validation metadata has no run_id."
        )

    _assert_version_contract(
        feature_schema_version=
            feature_schema_version,

        quality_policy_version=
            quality_policy_version,
    )

    validated_output = _metadata_output(
        validation_metadata,
        name=
            "validated_option_features",
    )

    relative_path = str(
        validated_output.get(
            "relative_path",
            "",
        )
    ).strip()

    expected_sha = str(
        validated_output.get(
            "sha256",
            "",
        )
    ).strip()

    if not relative_path:
        raise RuntimeError(
            "Validation metadata contains no validated "
            "option-feature path."
        )

    validated_path = _resolve_path(
        Path(
            relative_path
        ),
        fallback_parent=
            validation_metadata_path.parent,
    )

    if not validated_path.is_file():
        candidate = (
            Path(
                feature_root
            )
            / relative_path
        )

        if candidate.is_file():
            validated_path = candidate

    if not validated_path.is_file():
        raise FileNotFoundError(
            "Validated option-feature CSV is missing: "
            f"{validated_path}"
        )

    actual_validated_sha = (
        _sha256_file(
            validated_path
        )
    )

    if (
        expected_sha
        and actual_validated_sha
        != expected_sha
    ):
        raise RuntimeError(
            "Validated option-feature CSV hash does not "
            "match validation metadata."
        )

    source_feature_reference = str(
        validation_metadata.get(
            "source_feature_metadata",
            "",
        )
    ).strip()

    if not source_feature_reference:
        raise RuntimeError(
            "Validation metadata contains no "
            "source_feature_metadata reference."
        )

    source_feature_path = _resolve_path(
        Path(
            source_feature_reference
        ),
        fallback_parent=
            validation_metadata_path.parent,
    )

    if not source_feature_path.is_file():
        raise FileNotFoundError(
            "Source option-feature metadata is missing: "
            f"{source_feature_path}"
        )

    source_feature_metadata = _read_json(
        source_feature_path,
        context=
            "Schwab option-feature metadata",
    )

    source_run_id = str(
        source_feature_metadata.get(
            "run_id",
            "",
        )
    ).strip()

    source_feature_schema = str(
        source_feature_metadata.get(
            "feature_schema_version",
            "",
        )
    ).strip()

    if source_run_id != run_id:
        raise RuntimeError(
            "Run ID mismatch between feature and validation "
            "metadata."
        )

    if (
        source_feature_schema
        != feature_schema_version
    ):
        raise RuntimeError(
            "Feature-schema mismatch between feature and "
            "validation metadata."
        )

    capture_timestamp_utc = (
        _parse_capture_timestamp(
            source_feature_metadata.get(
                "run_started_at_utc"
            )
        )
    )

    source_validation = (
        source_feature_metadata.get(
            "validation"
        )
    )

    input_validation_id = ""

    if isinstance(
        source_validation,
        Mapping,
    ):
        input_validation_id = str(
            source_validation.get(
                "validation_id",
                "",
            )
        ).strip()

    frame = pd.read_csv(
        validated_path,
        low_memory=False,
    )

    required = {
        "feature_schema_version",
        "underlying_symbol",
        "quality_policy_version",
        "quality_symbol_status",
    }

    missing = sorted(
        required
        - set(
            frame.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Validated option-feature table is missing "
            f"required columns: {missing}"
        )

    if frame.empty:
        raise RuntimeError(
            "Validated option-feature table is empty."
        )

    row_feature_schemas = {
        str(
            value
        )
        for value
        in frame[
            "feature_schema_version"
        ].dropna().unique()
    }

    row_quality_policies = {
        str(
            value
        )
        for value
        in frame[
            "quality_policy_version"
        ].dropna().unique()
    }

    if row_feature_schemas != {
        feature_schema_version
    }:
        raise RuntimeError(
            "Validated rows do not all match the metadata "
            "feature schema."
        )

    if row_quality_policies != {
        quality_policy_version
    }:
        raise RuntimeError(
            "Validated rows do not all match the metadata "
            "quality policy."
        )

    frame[
        "underlying_symbol"
    ] = (
        frame[
            "underlying_symbol"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if (
        frame[
            "underlying_symbol"
        ]
        .eq("")
        .any()
    ):
        raise RuntimeError(
            "Validated option-feature table contains an "
            "empty underlying symbol."
        )

    metadata_sha = _sha256_file(
        validation_metadata_path
    )

    provenance = {
        "history_schema_version":
            HISTORY_SCHEMA_VERSION,

        "capture_timestamp_utc":
            capture_timestamp_utc,

        "run_id":
            run_id,

        "input_validation_id":
            input_validation_id,

        "source_validated_features_sha256":
            actual_validated_sha,

        "source_validation_metadata_sha256":
            metadata_sha,
    }

    for column in reversed(
        HISTORY_PROVENANCE_COLUMNS
    ):
        frame.insert(
            0,
            column,
            provenance[
                column
            ],
        )

    duplicated = frame.duplicated(
        subset=list(
            PRIMARY_KEY_COLUMNS
        ),
        keep=False,
    )

    if duplicated.any():
        values = (
            frame.loc[
                duplicated,
                list(
                    PRIMARY_KEY_COLUMNS
                ),
            ]
            .to_dict(
                orient="records"
            )
        )

        raise RuntimeError(
            "Incoming validated feature batch contains "
            f"duplicate history keys: {values}"
        )

    batch_info = {
        "run_id":
            run_id,

        "capture_timestamp_utc":
            capture_timestamp_utc,

        "feature_schema_version":
            feature_schema_version,

        "quality_policy_version":
            quality_policy_version,

        "validated_feature_path":
            str(
                validated_path
            ),

        "validated_feature_sha256":
            actual_validated_sha,

        "validation_metadata_path":
            str(
                validation_metadata_path
            ),

        "validation_metadata_sha256":
            metadata_sha,

        "source_feature_metadata_path":
            str(
                source_feature_path
            ),
    }

    return (
        frame,
        batch_info,
    )


def _sort_history(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    return frame.sort_values(
        [
            "capture_timestamp_utc",
            "run_id",
            "underlying_symbol",
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
            "History already contains one or more incoming "
            "primary keys with different row content."
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
            "History append lock already exists: "
            f"{lock_path}. Another append may be active, "
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


def append_option_feature_history(
    *,
    feature_root: Path = DEFAULT_OPTION_FEATURE_ROOT,
    validation_metadata: Path | None = None,
    history_path: Path = DEFAULT_HISTORY_PATH,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Append one validated Schwab option-feature snapshot to
    the frozen history-v1 table.

    The operation is idempotent by the composite key:
      capture_timestamp_utc
      run_id
      underlying_symbol
      feature_schema_version
      quality_policy_version

    Existing keys with different content are treated as a
    provenance conflict and fail closed.
    """
    feature_root = Path(
        feature_root
    )

    history_path = Path(
        history_path
    )

    validation_metadata_path = (
        _resolve_validation_metadata(
            feature_root=
                feature_root,

            validation_metadata=
                validation_metadata,
        )
    )

    if not validation_metadata_path.is_file():
        raise FileNotFoundError(
            "Option-feature validation metadata is missing: "
            f"{validation_metadata_path}"
        )

    (
        incoming,
        batch_info,
    ) = _load_validated_batch(
        feature_root=
            feature_root,

        validation_metadata_path=
            validation_metadata_path,
    )

    incoming = _sort_history(
        incoming
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
            **batch_info,

            "history_table_name":
                HISTORY_TABLE_NAME,

            "history_schema_version":
                HISTORY_SCHEMA_VERSION,

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
                    "History table columns do not match the "
                    "frozen history-v1 schema. Refusing to "
                    "append across schema drift."
                )

            duplicated_existing = (
                existing.duplicated(
                    subset=list(
                        PRIMARY_KEY_COLUMNS
                    ),
                    keep=False,
                )
            )

            if duplicated_existing.any():
                raise RuntimeError(
                    "Existing history contains duplicate "
                    "primary keys."
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
                    "History append would create duplicate "
                    "primary keys."
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
        **batch_info,

        "history_table_name":
            HISTORY_TABLE_NAME,

        "history_schema_version":
            HISTORY_SCHEMA_VERSION,

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
        "SCHWAB OPTION FEATURE HISTORY APPEND"
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
        "Capture timestamp:",
        result[
            "capture_timestamp_utc"
        ],
    )

    print(
        "Feature schema:",
        result[
            "feature_schema_version"
        ],
    )

    print(
        "Quality policy:",
        result[
            "quality_policy_version"
        ],
    )

    print(
        "History schema:",
        result[
            "history_schema_version"
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
            "Append one validated Schwab option-feature "
            "snapshot to the frozen history-v1 table."
        )
    )

    parser.add_argument(
        "--feature-root",
        type=Path,
        default=
            DEFAULT_OPTION_FEATURE_ROOT,
        help=
            "Schwab option-feature artifact directory.",
    )

    parser.add_argument(
        "--validation-metadata",
        type=Path,
        default=None,
        help=(
            "Specific option-feature validation metadata "
            "JSON. Defaults to the newest validation "
            "metadata artifact."
        ),
    )

    parser.add_argument(
        "--history-path",
        type=Path,
        default=
            DEFAULT_HISTORY_PATH,
        help=
            "Canonical option feature-history CSV.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=
            "Validate and report the append without writing.",
    )

    return parser


def main() -> None:
    args = (
        build_parser()
        .parse_args()
    )

    result = append_option_feature_history(
        feature_root=
            args.feature_root,

        validation_metadata=
            args.validation_metadata,

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
