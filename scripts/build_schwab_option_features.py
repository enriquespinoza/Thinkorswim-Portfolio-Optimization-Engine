from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

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
    PROCESSED_DATA_DIR,
)
from scripts.validate_schwab_feature_inputs import (
    DEFAULT_EXTREME_PERCENT_CHANGE,
    validate_schwab_feature_inputs,
)
from src.features.schwab_option_features import (
    FEATURE_SCHEMA_VERSION,
    OptionFeatureConfig,
    build_schwab_option_features,
    eligibility_summary,
    prepare_schwab_option_contracts,
)


DEFAULT_PROCESSED_ROOT = (
    PROCESSED_DATA_DIR
    / "schwab"
)

DEFAULT_OUTPUT_ROOT = (
    FEATURE_DATA_DIR
    / "schwab"
    / "options"
)

CAPTURE_RUNS_DIRECTORY = "capture_runs"
METADATA_SCHEMA_VERSION = "1.0.0"


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


def _sha256_bytes(
    value: bytes,
) -> str:
    return hashlib.sha256(
        value
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


def _resolve_summary_path(
    *,
    processed_root: Path,
    run_summary: Path | None,
) -> Path:
    if run_summary is None:
        return _latest_capture_summary(
            processed_root
        )

    path = Path(
        run_summary
    )

    if path.is_absolute():
        return path

    project_candidate = (
        PROJECT_ROOT
        / path
    )

    if project_candidate.is_file():
        return project_candidate

    return path


def _normalized_artifacts(
    summary: Mapping[
        str,
        Any,
    ],
    *,
    dataset: str,
    processed_root: Path,
) -> list[
    tuple[
        str,
        Path,
        dict[str, Any],
        dict[str, Any],
    ]
]:
    result: list[
        tuple[
            str,
            Path,
            dict[str, Any],
            dict[str, Any],
        ]
    ] = []

    artifacts = summary.get(
        "artifacts",
        [],
    )

    if not isinstance(
        artifacts,
        list,
    ):
        raise RuntimeError(
            "Capture-run summary artifacts must be a list."
        )

    for index, artifact in enumerate(
        artifacts
    ):
        if not isinstance(
            artifact,
            Mapping,
        ):
            raise RuntimeError(
                f"Capture-run artifact {index} is not an object."
            )

        if str(
            artifact.get(
                "dataset",
                "",
            )
        ) != dataset:
            continue

        normalized = artifact.get(
            "normalized"
        )

        raw = artifact.get(
            "raw"
        )

        if not isinstance(
            normalized,
            Mapping,
        ):
            raise RuntimeError(
                f"{dataset} artifact {index} has no "
                "normalized metadata."
            )

        if not isinstance(
            raw,
            Mapping,
        ):
            raise RuntimeError(
                f"{dataset} artifact {index} has no "
                "raw metadata."
            )

        relative_path = str(
            normalized.get(
                "relative_path",
                "",
            )
        ).strip()

        if not relative_path:
            raise RuntimeError(
                f"{dataset} artifact {index} has no "
                "normalized relative_path."
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

        result.append(
            (
                str(
                    artifact.get(
                        "scope",
                        "",
                    )
                ),
                path,
                dict(
                    raw
                ),
                dict(
                    normalized
                ),
            )
        )

    if not result:
        raise RuntimeError(
            "Capture run contains no "
            f"{dataset!r} artifacts."
        )

    return result


def _load_dataset(
    artifacts: list[
        tuple[
            str,
            Path,
            dict[str, Any],
            dict[str, Any],
        ]
    ],
) -> pd.DataFrame:
    frames: list[
        pd.DataFrame
    ] = []

    for scope, path, _, _ in artifacts:
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

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


def _csv_bytes(
    frame: pd.DataFrame,
) -> bytes:
    return (
        frame
        .to_csv(
            index=False,
            lineterminator="\n",
        )
        .encode(
            "utf-8"
        )
    )


def _write_create_once(
    path: Path,
    content: bytes,
) -> str:
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        with path.open(
            "xb"
        ) as handle:
            handle.write(
                content
            )

    except FileExistsError:
        existing = (
            path.read_bytes()
        )

        if existing != content:
            raise RuntimeError(
                "Output path already exists with "
                "different content: "
                f"{path}"
            )

        return "EXISTS"

    return "WRITTEN"


def _artifact_source_metadata(
    artifacts: list[
        tuple[
            str,
            Path,
            dict[str, Any],
            dict[str, Any],
        ]
    ],
) -> list[
    dict[
        str,
        Any,
    ]
]:
    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for (
        scope,
        path,
        raw,
        normalized,
    ) in artifacts:
        rows.append(
            {
                "scope":
                    scope,

                "raw_record_id":
                    str(
                        raw.get(
                            "record_id",
                            "",
                        )
                    ),

                "raw_payload_sha256":
                    str(
                        raw.get(
                            "payload_sha256",
                            "",
                        )
                    ),

                "normalized_relative_path":
                    str(
                        normalized.get(
                            "relative_path",
                            "",
                        )
                    ),

                "normalized_sha256":
                    str(
                        normalized.get(
                            "normalized_sha256",
                            "",
                        )
                    ),

                "resolved_path":
                    str(
                        path
                    ),
            }
        )

    return rows


def build_option_feature_artifacts(
    *,
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_summary: Path | None = None,
    config: OptionFeatureConfig | None = None,
    extreme_percent_change: float = (
        DEFAULT_EXTREME_PERCENT_CHANGE
    ),
) -> dict[str, Any]:
    """
    Build immutable V2 Schwab option-feature artifacts for
    one validated capture run.

    The runner:
      1. resolves one capture-run summary,
      2. re-runs semantic input validation,
      3. blocks on validation FAIL,
      4. loads only that run's quote/option artifacts,
      5. derives eligible contracts and symbol features,
      6. writes deterministic create-once outputs.
    """
    processed_root = Path(
        processed_root
    )

    output_root = Path(
        output_root
    )

    if config is None:
        config = (
            OptionFeatureConfig()
        )

    summary_path = (
        _resolve_summary_path(
            processed_root=
                processed_root,

            run_summary=
                run_summary,
        )
    )

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

    validation = (
        validate_schwab_feature_inputs(
            processed_root=
                processed_root,

            run_summary=
                summary_path,

            max_quote_age_minutes=
                config.max_quote_age_minutes,

            wide_spread_pct=
                config.max_relative_spread_pct,

            extreme_percent_change=
                extreme_percent_change,
        )
    )

    if (
        validation[
            "status"
        ]
        == "FAIL"
    ):
        raise RuntimeError(
            "Schwab feature-input validation failed "
            f"for run {run_id}; feature artifacts were "
            "not written."
        )

    option_artifacts = (
        _normalized_artifacts(
            summary,
            dataset=
                "option_chains",
            processed_root=
                processed_root,
        )
    )

    quote_artifacts = (
        _normalized_artifacts(
            summary,
            dataset=
                "quotes",
            processed_root=
                processed_root,
        )
    )

    options = _load_dataset(
        option_artifacts
    )

    quotes = _load_dataset(
        quote_artifacts
    )

    prepared = (
        prepare_schwab_option_contracts(
            options,
            quotes,
            config=
                config,
        )
    )

    features = (
        build_schwab_option_features(
            options,
            quotes,
            config=
                config,
        )
    )

    eligibility = (
        eligibility_summary(
            prepared
        )
    )

    features = features.sort_values(
        "underlying_symbol",
        kind="stable",
    ).reset_index(
        drop=True
    )

    eligibility = eligibility.sort_values(
        "underlying_symbol",
        kind="stable",
    ).reset_index(
        drop=True
    )

    feature_bytes = _csv_bytes(
        features
    )

    eligibility_bytes = _csv_bytes(
        eligibility
    )

    feature_filename = (
        f"{run_id}__option_features.csv"
    )

    eligibility_filename = (
        f"{run_id}__option_eligibility.csv"
    )

    metadata_filename = (
        f"{run_id}__option_feature_metadata.json"
    )

    feature_path = (
        output_root
        / feature_filename
    )

    eligibility_path = (
        output_root
        / eligibility_filename
    )

    metadata_path = (
        output_root
        / metadata_filename
    )

    metadata: dict[
        str,
        Any,
    ] = {
        "metadata_schema_version":
            METADATA_SCHEMA_VERSION,

        "feature_schema_version":
            FEATURE_SCHEMA_VERSION,

        "run_id":
            run_id,

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

        "validation":
            {
                "validation_id":
                    str(
                        validation.get(
                            "validation_id",
                            "",
                        )
                    ),

                "status":
                    str(
                        validation.get(
                            "status",
                            "",
                        )
                    ),

                "error_count":
                    int(
                        validation.get(
                            "error_count",
                            0,
                        )
                    ),

                "warning_count":
                    int(
                        validation.get(
                            "warning_count",
                            0,
                        )
                    ),
            },

        "config":
            asdict(
                config
            ),

        "source_artifacts":
            {
                "option_chains":
                    _artifact_source_metadata(
                        option_artifacts
                    ),

                "quotes":
                    _artifact_source_metadata(
                        quote_artifacts
                    ),
            },

        "outputs":
            {
                "option_features":
                    {
                        "relative_path":
                            feature_filename,

                        "rows":
                            int(
                                len(
                                    features
                                )
                            ),

                        "columns":
                            int(
                                len(
                                    features.columns
                                )
                            ),

                        "sha256":
                            _sha256_bytes(
                                feature_bytes
                            ),

                        "column_names":
                            [
                                str(
                                    column
                                )
                                for column
                                in features.columns
                            ],
                    },

                "option_eligibility":
                    {
                        "relative_path":
                            eligibility_filename,

                        "rows":
                            int(
                                len(
                                    eligibility
                                )
                            ),

                        "columns":
                            int(
                                len(
                                    eligibility.columns
                                )
                            ),

                        "sha256":
                            _sha256_bytes(
                                eligibility_bytes
                            ),

                        "column_names":
                            [
                                str(
                                    column
                                )
                                for column
                                in eligibility.columns
                            ],
                    },
            },

        "excluded_provider_fields":
            [
                "percentChange",
                "markPercentChange",
                "intrinsicValue",
                "extrinsicValue",
                "theoreticalVolatility",
            ],
    }

    metadata_bytes = (
        json.dumps(
            metadata,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode(
        "utf-8"
    )

    statuses = {
        "option_features":
            _write_create_once(
                feature_path,
                feature_bytes,
            ),

        "option_eligibility":
            _write_create_once(
                eligibility_path,
                eligibility_bytes,
            ),

        "metadata":
            _write_create_once(
                metadata_path,
                metadata_bytes,
            ),
    }

    return {
        "run_id":
            run_id,

        "validation_status":
            validation[
                "status"
            ],

        "validation_id":
            validation[
                "validation_id"
            ],

        "feature_rows":
            int(
                len(
                    features
                )
            ),

        "eligibility_rows":
            int(
                len(
                    eligibility
                )
            ),

        "eligible_contracts":
            int(
                prepared[
                    "eligible_contract"
                ].sum()
            ),

        "total_contracts":
            int(
                len(
                    prepared
                )
            ),

        "paths":
            {
                "option_features":
                    str(
                        feature_path
                    ),

                "option_eligibility":
                    str(
                        eligibility_path
                    ),

                "metadata":
                    str(
                        metadata_path
                    ),
            },

        "statuses":
            statuses,
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
        "SCHWAB OPTION FEATURE BUILD"
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
        "Validation:",
        result[
            "validation_status"
        ],
    )

    print(
        "Validation ID:",
        result[
            "validation_id"
        ],
    )

    print(
        "Contracts:",
        (
            f"{result['eligible_contracts']} eligible / "
            f"{result['total_contracts']} total"
        ),
    )

    print(
        "Feature rows:",
        result[
            "feature_rows"
        ],
    )

    print(
        "Eligibility rows:",
        result[
            "eligibility_rows"
        ],
    )

    print()
    print(
        "OUTPUTS"
    )
    print(
        "-" * 88
    )

    for key, path in (
        result[
            "paths"
        ].items()
    ):
        print(
            f"{key:<20} "
            f"{result['statuses'][key]:<8} "
            f"{path}"
        )

    print()
    print(
        "No raw Schwab payloads were printed."
    )

    print(
        "=" * 88
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic V2 option-feature "
            "artifacts from one validated Schwab "
            "capture run."
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
        "--output-root",
        type=Path,
        default=
            DEFAULT_OUTPUT_ROOT,
        help=
            "Schwab option feature output directory.",
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
        "--max-relative-spread-pct",
        type=float,
        default=25.0,
        help=(
            "Maximum eligible option bid/ask spread "
            "as percentage of midpoint. Default: 25."
        ),
    )

    parser.add_argument(
        "--max-quote-age-minutes",
        type=float,
        default=30.0,
        help=(
            "Maximum eligible option quote age at "
            "capture. Default: 30 minutes."
        ),
    )

    parser.add_argument(
        "--atm-moneyness-pct",
        type=float,
        default=2.0,
        help=(
            "Absolute moneyness threshold used for ATM "
            "aggregates. Default: 2 percent."
        ),
    )

    parser.add_argument(
        "--allow-zero-bid",
        action="store_true",
        help=(
            "Allow bid=0 contracts through the bid gate. "
            "Disabled by default."
        ),
    )

    parser.add_argument(
        "--include-zero-dte",
        action="store_true",
        help=(
            "Include DTE=0 contracts in the standard "
            "feature aggregates. Disabled by default."
        ),
    )

    parser.add_argument(
        "--extreme-percent-change",
        type=float,
        default=
            DEFAULT_EXTREME_PERCENT_CHANGE,
        help=(
            "Semantic-validation warning threshold for "
            "provider percent-change fields. These fields "
            "remain excluded from generated features."
        ),
    )

    return parser


def main() -> None:
    args = (
        build_parser()
        .parse_args()
    )

    config = (
        OptionFeatureConfig(
            max_relative_spread_pct=
                args.max_relative_spread_pct,

            max_quote_age_minutes=
                args.max_quote_age_minutes,

            atm_moneyness_pct=
                args.atm_moneyness_pct,

            require_positive_bid=
                not args.allow_zero_bid,

            exclude_zero_dte=
                not args.include_zero_dte,
        )
    )

    result = (
        build_option_feature_artifacts(
            processed_root=
                args.processed_root,

            output_root=
                args.output_root,

            run_summary=
                args.run_summary,

            config=
                config,

            extreme_percent_change=
                args.extreme_percent_change,
        )
    )

    _print_result(
        result
    )


if __name__ == "__main__":
    main()
