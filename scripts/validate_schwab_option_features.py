from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
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
from src.features.schwab_option_features import (
    FEATURE_SCHEMA_VERSION,
)


DEFAULT_OPTION_FEATURE_ROOT = (
    FEATURE_DATA_DIR
    / "schwab"
    / "options"
)

QUALITY_POLICY_VERSION = "1.0.0"
VALIDATION_METADATA_SCHEMA_VERSION = "1.0.0"

PASS_MIN_ELIGIBLE_RATIO = 0.80
PASS_MIN_ELIGIBLE_CONTRACTS = 100
PASS_MIN_ELIGIBLE_EXPIRATIONS = 10

CAUTION_MIN_ELIGIBLE_RATIO = 0.50
CAUTION_MIN_ELIGIBLE_CONTRACTS = 50
CAUTION_MIN_ELIGIBLE_EXPIRATIONS = 5

MIN_OVERALL_MATCHED_PAIR_COUNT = 10
MIN_BUCKET_MATCHED_PAIR_COUNT = 5
MIN_ATM_BUCKET_CONTRACT_COUNT = 20

DTE_BUCKET_NAMES = (
    "dte_1_7",
    "dte_8_30",
    "dte_31_90",
    "dte_91_365",
    "dte_366_plus",
)

IDENTITY_AND_SUPPORT_COLUMNS = {
    "feature_schema_version",
    "underlying_symbol",
    "spot",
    "contract_count_total",
    "contract_count_eligible",
    "eligible_contract_ratio",
    "zero_dte_contract_count",
    "zero_dte_contract_ratio",
    "eligible_call_count",
    "eligible_put_count",
    "eligible_expiration_count",
    "eligible_dte_min",
    "eligible_dte_max",
    "atm_contract_count",
    "atm_expiration_count",
    "delta_25_expiration_pair_count",
    "atm_expiration_pair_count",
}

for _bucket in DTE_BUCKET_NAMES:
    IDENTITY_AND_SUPPORT_COLUMNS.update(
        {
            f"atm_contract_count_{_bucket}",
            f"atm_expiration_count_{_bucket}",
            f"delta_25_pair_count_{_bucket}",
            f"atm_skew_pair_count_{_bucket}",
        }
    )


@dataclass(frozen=True)
class FeatureRule:
    feature_name: str
    support_column: str
    minimum_support: int
    rule_code: str


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


def _latest_feature_metadata(
    feature_root: Path,
) -> Path:
    candidates = sorted(
        Path(
            feature_root
        ).glob(
            "*__option_feature_metadata.json"
        )
    )

    if not candidates:
        raise FileNotFoundError(
            "No Schwab option-feature metadata files found in "
            f"{feature_root}"
        )

    return candidates[-1]


def _resolve_metadata_path(
    *,
    feature_root: Path,
    feature_metadata: Path | None,
) -> Path:
    if feature_metadata is None:
        return _latest_feature_metadata(
            feature_root
        )

    path = Path(
        feature_metadata
    )

    if path.is_absolute():
        return path

    project_candidate = (
        PROJECT_ROOT
        / path
    )

    if project_candidate.is_file():
        return project_candidate

    feature_candidate = (
        Path(
            feature_root
        )
        / path
    )

    if feature_candidate.is_file():
        return feature_candidate

    return path


def _feature_rules() -> list[FeatureRule]:
    rules = [
        FeatureRule(
            feature_name=
                "delta_25_put_call_iv_skew_matched_median",
            support_column=
                "delta_25_expiration_pair_count",
            minimum_support=
                MIN_OVERALL_MATCHED_PAIR_COUNT,
            rule_code=
                "OVERALL_MATCHED_PAIR_SUPPORT",
        ),
        FeatureRule(
            feature_name=
                "atm_put_call_iv_skew_matched_median",
            support_column=
                "atm_expiration_pair_count",
            minimum_support=
                MIN_OVERALL_MATCHED_PAIR_COUNT,
            rule_code=
                "OVERALL_MATCHED_PAIR_SUPPORT",
        ),
    ]

    for bucket in DTE_BUCKET_NAMES:
        rules.extend(
            [
                FeatureRule(
                    feature_name=
                        f"delta_25_put_call_iv_skew_{bucket}",
                    support_column=
                        f"delta_25_pair_count_{bucket}",
                    minimum_support=
                        MIN_BUCKET_MATCHED_PAIR_COUNT,
                    rule_code=
                        "BUCKET_MATCHED_PAIR_SUPPORT",
                ),
                FeatureRule(
                    feature_name=
                        f"atm_put_call_iv_skew_{bucket}",
                    support_column=
                        f"atm_skew_pair_count_{bucket}",
                    minimum_support=
                        MIN_BUCKET_MATCHED_PAIR_COUNT,
                    rule_code=
                        "BUCKET_MATCHED_PAIR_SUPPORT",
                ),
                FeatureRule(
                    feature_name=
                        f"atm_iv_median_{bucket}",
                    support_column=
                        f"atm_contract_count_{bucket}",
                    minimum_support=
                        MIN_ATM_BUCKET_CONTRACT_COUNT,
                    rule_code=
                        "ATM_BUCKET_CONTRACT_SUPPORT",
                ),
            ]
        )

    return rules


def classify_symbol_quality(
    row: Mapping[
        str,
        Any,
    ],
) -> tuple[
    str,
    list[str],
]:
    ratio = float(
        row[
            "eligible_contract_ratio"
        ]
    )

    contracts = int(
        row[
            "contract_count_eligible"
        ]
    )

    expirations = int(
        row[
            "eligible_expiration_count"
        ]
    )

    if (
        ratio
        >= PASS_MIN_ELIGIBLE_RATIO
        and contracts
        >= PASS_MIN_ELIGIBLE_CONTRACTS
        and expirations
        >= PASS_MIN_ELIGIBLE_EXPIRATIONS
    ):
        return (
            "PASS",
            [],
        )

    if (
        ratio
        >= CAUTION_MIN_ELIGIBLE_RATIO
        and contracts
        >= CAUTION_MIN_ELIGIBLE_CONTRACTS
        and expirations
        >= CAUTION_MIN_ELIGIBLE_EXPIRATIONS
    ):
        reasons: list[
            str
        ] = []

        if ratio < PASS_MIN_ELIGIBLE_RATIO:
            reasons.append(
                "eligible_contract_ratio_below_pass"
            )

        if contracts < PASS_MIN_ELIGIBLE_CONTRACTS:
            reasons.append(
                "eligible_contract_count_below_pass"
            )

        if expirations < PASS_MIN_ELIGIBLE_EXPIRATIONS:
            reasons.append(
                "eligible_expiration_count_below_pass"
            )

        return (
            "CAUTION",
            reasons,
        )

    reasons = []

    if ratio < CAUTION_MIN_ELIGIBLE_RATIO:
        reasons.append(
            "eligible_contract_ratio_below_caution"
        )

    if contracts < CAUTION_MIN_ELIGIBLE_CONTRACTS:
        reasons.append(
            "eligible_contract_count_below_caution"
        )

    if expirations < CAUTION_MIN_ELIGIBLE_EXPIRATIONS:
        reasons.append(
            "eligible_expiration_count_below_caution"
        )

    return (
        "SUPPRESS",
        reasons,
    )


def _require_columns(
    frame: pd.DataFrame,
    columns: set[str],
    *,
    context: str,
) -> None:
    missing = sorted(
        columns
        - set(
            frame.columns
        )
    )

    if missing:
        raise ValueError(
            f"{context} is missing required columns: "
            f"{missing}"
        )


def _model_feature_columns(
    frame: pd.DataFrame,
) -> list[str]:
    return [
        str(
            column
        )
        for column
        in frame.columns
        if column
        not in IDENTITY_AND_SUPPORT_COLUMNS
        and not str(
            column
        ).startswith(
            "quality_"
        )
    ]


def validate_option_feature_frame(
    features: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Apply the frozen option-feature quality policy.

    The returned validated frame preserves every symbol row
    and all identity/support columns. Unsupported model
    features are set to NaN. The long-form quality table
    records the reason and support behind every decision.
    """
    required = {
        "feature_schema_version",
        "underlying_symbol",
        "eligible_contract_ratio",
        "contract_count_eligible",
        "eligible_expiration_count",
    }

    for rule in _feature_rules():
        required.add(
            rule.feature_name
        )
        required.add(
            rule.support_column
        )

    required.update(
        {
            "atm_contract_count_dte_8_30",
            "atm_contract_count_dte_91_365",
            "atm_iv_term_spread_91_365_minus_8_30",
        }
    )

    _require_columns(
        features,
        required,
        context=
            "Schwab option-feature frame",
    )

    validated = (
        features.copy()
    )

    quality_rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    statuses: list[
        str
    ] = []

    reasons_text: list[
        str
    ] = []

    model_columns = (
        _model_feature_columns(
            validated
        )
    )

    rule_by_feature = {
        rule.feature_name:
            rule
        for rule in _feature_rules()
    }

    for index, row in (
        validated.iterrows()
    ):
        symbol = str(
            row[
                "underlying_symbol"
            ]
        )

        (
            symbol_status,
            symbol_reasons,
        ) = classify_symbol_quality(
            row
        )

        statuses.append(
            symbol_status
        )

        reasons_text.append(
            "|".join(
                symbol_reasons
            )
        )

        if symbol_status == "SUPPRESS":
            for feature_name in model_columns:
                if feature_name in validated.columns:
                    validated.loc[
                        index,
                        feature_name,
                    ] = np.nan

                    quality_rows.append(
                        {
                            "underlying_symbol":
                                symbol,

                            "feature_name":
                                feature_name,

                            "usable":
                                False,

                            "quality_status":
                                "SUPPRESS",

                            "rule_code":
                                "SYMBOL_QUALITY",

                            "support_column":
                                None,

                            "support_count":
                                None,

                            "minimum_support":
                                None,

                            "reason":
                                "|".join(
                                    symbol_reasons
                                )
                                or "symbol_quality_suppressed",
                        }
                    )

            continue

        for feature_name in model_columns:
            rule = rule_by_feature.get(
                feature_name
            )

            if rule is None:
                quality_rows.append(
                    {
                        "underlying_symbol":
                            symbol,

                        "feature_name":
                            feature_name,

                        "usable":
                            True,

                        "quality_status":
                            symbol_status,

                        "rule_code":
                            "SYMBOL_QUALITY_ONLY",

                        "support_column":
                            None,

                        "support_count":
                            None,

                        "minimum_support":
                            None,

                        "reason":
                            "",
                    }
                )

                continue

            support_value = pd.to_numeric(
                pd.Series(
                    [
                        row[
                            rule.support_column
                        ]
                    ]
                ),
                errors="coerce",
            ).iloc[
                0
            ]

            supported = (
                pd.notna(
                    support_value
                )
                and float(
                    support_value
                )
                >= rule.minimum_support
            )

            if not supported:
                validated.loc[
                    index,
                    feature_name,
                ] = np.nan

            quality_rows.append(
                {
                    "underlying_symbol":
                        symbol,

                    "feature_name":
                        feature_name,

                    "usable":
                        bool(
                            supported
                        ),

                    "quality_status":
                        symbol_status
                        if supported
                        else "SUPPRESS",

                    "rule_code":
                        rule.rule_code,

                    "support_column":
                        rule.support_column,

                    "support_count":
                        (
                            float(
                                support_value
                            )
                            if pd.notna(
                                support_value
                            )
                            else None
                        ),

                    "minimum_support":
                        rule.minimum_support,

                    "reason":
                        (
                            ""
                            if supported
                            else "insufficient_feature_support"
                        ),
                }
            )

        near_support = float(
            row[
                "atm_contract_count_dte_8_30"
            ]
        )

        back_support = float(
            row[
                "atm_contract_count_dte_91_365"
            ]
        )

        term_feature = (
            "atm_iv_term_spread_91_365_minus_8_30"
        )

        term_supported = (
            near_support
            >= MIN_ATM_BUCKET_CONTRACT_COUNT
            and back_support
            >= MIN_ATM_BUCKET_CONTRACT_COUNT
        )

        if not term_supported:
            validated.loc[
                index,
                term_feature,
            ] = np.nan

        existing = [
            quality
            for quality in quality_rows
            if quality[
                "underlying_symbol"
            ]
            == symbol
            and quality[
                "feature_name"
            ]
            == term_feature
        ]

        if existing:
            quality_rows.remove(
                existing[
                    -1
                ]
            )

        quality_rows.append(
            {
                "underlying_symbol":
                    symbol,

                "feature_name":
                    term_feature,

                "usable":
                    bool(
                        term_supported
                    ),

                "quality_status":
                    symbol_status
                    if term_supported
                    else "SUPPRESS",

                "rule_code":
                    "TERM_STRUCTURE_COMPONENT_SUPPORT",

                "support_column":
                    (
                        "atm_contract_count_dte_8_30"
                        "|atm_contract_count_dte_91_365"
                    ),

                "support_count":
                    float(
                        min(
                            near_support,
                            back_support,
                        )
                    ),

                "minimum_support":
                    MIN_ATM_BUCKET_CONTRACT_COUNT,

                "reason":
                    (
                        ""
                        if term_supported
                        else "insufficient_term_structure_support"
                    ),
            }
        )

    validated[
        "quality_policy_version"
    ] = QUALITY_POLICY_VERSION

    validated[
        "quality_symbol_status"
    ] = statuses

    validated[
        "quality_symbol_reasons"
    ] = reasons_text

    quality = pd.DataFrame(
        quality_rows
    )

    if not quality.empty:
        quality = quality.sort_values(
            [
                "underlying_symbol",
                "feature_name",
            ],
            kind="stable",
        ).reset_index(
            drop=True
        )

    return (
        validated,
        quality,
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
                "Output path already exists with different "
                f"content: {path}"
            )

        return "EXISTS"

    return "WRITTEN"


def validate_option_feature_artifacts(
    *,
    feature_root: Path = DEFAULT_OPTION_FEATURE_ROOT,
    feature_metadata: Path | None = None,
) -> dict[str, Any]:
    feature_root = Path(
        feature_root
    )

    metadata_path = (
        _resolve_metadata_path(
            feature_root=
                feature_root,

            feature_metadata=
                feature_metadata,
        )
    )

    source_metadata = _read_json(
        metadata_path,
        context=
            "Schwab option-feature metadata",
    )

    source_schema = str(
        source_metadata.get(
            "feature_schema_version",
            "",
        )
    )

    if (
        source_schema
        != FEATURE_SCHEMA_VERSION
    ):
        raise RuntimeError(
            "Option-feature schema mismatch. "
            f"Source={source_schema!r}, "
            f"code={FEATURE_SCHEMA_VERSION!r}."
        )

    run_id = str(
        source_metadata.get(
            "run_id",
            "",
        )
    ).strip()

    if not run_id:
        raise RuntimeError(
            "Option-feature metadata has no run_id."
        )

    outputs = source_metadata.get(
        "outputs"
    )

    if not isinstance(
        outputs,
        Mapping,
    ):
        raise RuntimeError(
            "Option-feature metadata has no outputs object."
        )

    feature_output = outputs.get(
        "option_features"
    )

    if not isinstance(
        feature_output,
        Mapping,
    ):
        raise RuntimeError(
            "Option-feature metadata has no option_features output."
        )

    relative_path = str(
        feature_output.get(
            "relative_path",
            "",
        )
    ).strip()

    expected_sha = str(
        feature_output.get(
            "sha256",
            "",
        )
    ).strip()

    if not relative_path:
        raise RuntimeError(
            "Option-feature metadata contains no feature path."
        )

    feature_path = (
        metadata_path.parent
        / relative_path
    )

    if not feature_path.is_file():
        raise FileNotFoundError(
            "Option-feature CSV is missing: "
            f"{feature_path}"
        )

    actual_sha = (
        _sha256_file(
            feature_path
        )
    )

    if (
        expected_sha
        and actual_sha
        != expected_sha
    ):
        raise RuntimeError(
            "Option-feature CSV hash does not match metadata."
        )

    features = pd.read_csv(
        feature_path,
        low_memory=False,
    )

    (
        validated,
        quality,
    ) = validate_option_feature_frame(
        features
    )

    validated = validated.sort_values(
        "underlying_symbol",
        kind="stable",
    ).reset_index(
        drop=True
    )

    validated_bytes = _csv_bytes(
        validated
    )

    quality_bytes = _csv_bytes(
        quality
    )

    version_token = (
        "v"
        + FEATURE_SCHEMA_VERSION
    )

    validated_filename = (
        f"{run_id}__{version_token}"
        "__validated_option_features.csv"
    )

    quality_filename = (
        f"{run_id}__{version_token}"
        "__option_feature_quality.csv"
    )

    validation_metadata_filename = (
        f"{run_id}__{version_token}"
        "__option_feature_validation_metadata.json"
    )

    validated_path = (
        feature_root
        / validated_filename
    )

    quality_path = (
        feature_root
        / quality_filename
    )

    validation_metadata_path = (
        feature_root
        / validation_metadata_filename
    )

    symbol_status_counts = {
        str(
            key
        ):
            int(
                value
            )
        for key, value
        in validated[
            "quality_symbol_status"
        ].value_counts().items()
    }

    feature_suppression_count = int(
        (
            quality[
                "usable"
            ]
            == False
        ).sum()
    )

    policy = {
        "quality_policy_version":
            QUALITY_POLICY_VERSION,

        "symbol_quality":
            {
                "pass":
                    {
                        "eligible_contract_ratio_min":
                            PASS_MIN_ELIGIBLE_RATIO,

                        "eligible_contract_count_min":
                            PASS_MIN_ELIGIBLE_CONTRACTS,

                        "eligible_expiration_count_min":
                            PASS_MIN_ELIGIBLE_EXPIRATIONS,
                    },

                "caution":
                    {
                        "eligible_contract_ratio_min":
                            CAUTION_MIN_ELIGIBLE_RATIO,

                        "eligible_contract_count_min":
                            CAUTION_MIN_ELIGIBLE_CONTRACTS,

                        "eligible_expiration_count_min":
                            CAUTION_MIN_ELIGIBLE_EXPIRATIONS,
                    },

                "otherwise":
                    "SUPPRESS",
            },

        "feature_support":
            {
                "overall_matched_expiration_skew_min_pairs":
                    MIN_OVERALL_MATCHED_PAIR_COUNT,

                "dte_bucket_matched_skew_min_pairs":
                    MIN_BUCKET_MATCHED_PAIR_COUNT,

                "atm_iv_bucket_min_contracts":
                    MIN_ATM_BUCKET_CONTRACT_COUNT,

                "term_structure":
                    (
                        "both component ATM-IV buckets must "
                        "meet atm_iv_bucket_min_contracts"
                    ),
            },
    }

    validation_metadata = {
        "validation_metadata_schema_version":
            VALIDATION_METADATA_SCHEMA_VERSION,

        "quality_policy_version":
            QUALITY_POLICY_VERSION,

        "feature_schema_version":
            FEATURE_SCHEMA_VERSION,

        "run_id":
            run_id,

        "source_feature_metadata":
            str(
                metadata_path
            ),

        "source_option_features":
            {
                "path":
                    str(
                        feature_path
                    ),

                "sha256":
                    actual_sha,
            },

        "policy":
            policy,

        "symbol_status_counts":
            symbol_status_counts,

        "feature_suppression_count":
            feature_suppression_count,

        "outputs":
            {
                "validated_option_features":
                    {
                        "relative_path":
                            validated_filename,

                        "rows":
                            int(
                                len(
                                    validated
                                )
                            ),

                        "columns":
                            int(
                                len(
                                    validated.columns
                                )
                            ),

                        "sha256":
                            _sha256_bytes(
                                validated_bytes
                            ),
                    },

                "option_feature_quality":
                    {
                        "relative_path":
                            quality_filename,

                        "rows":
                            int(
                                len(
                                    quality
                                )
                            ),

                        "columns":
                            int(
                                len(
                                    quality.columns
                                )
                            ),

                        "sha256":
                            _sha256_bytes(
                                quality_bytes
                            ),
                    },
            },
    }

    metadata_bytes = (
        json.dumps(
            validation_metadata,
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
        "validated_option_features":
            _write_create_once(
                validated_path,
                validated_bytes,
            ),

        "option_feature_quality":
            _write_create_once(
                quality_path,
                quality_bytes,
            ),

        "validation_metadata":
            _write_create_once(
                validation_metadata_path,
                metadata_bytes,
            ),
    }

    return {
        "run_id":
            run_id,

        "feature_schema_version":
            FEATURE_SCHEMA_VERSION,

        "quality_policy_version":
            QUALITY_POLICY_VERSION,

        "symbol_status_counts":
            symbol_status_counts,

        "feature_suppression_count":
            feature_suppression_count,

        "paths":
            {
                "validated_option_features":
                    str(
                        validated_path
                    ),

                "option_feature_quality":
                    str(
                        quality_path
                    ),

                "validation_metadata":
                    str(
                        validation_metadata_path
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
        "SCHWAB OPTION FEATURE QUALITY GATE"
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
        "Symbol statuses:",
        result[
            "symbol_status_counts"
        ],
    )

    print(
        "Suppressed feature values:",
        result[
            "feature_suppression_count"
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
            f"{key:<30} "
            f"{result['statuses'][key]:<8} "
            f"{path}"
        )

    print(
        "=" * 88
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the frozen Schwab V2 option-feature "
            "quality policy and suppress unsupported "
            "model inputs without dropping symbol rows."
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
        "--feature-metadata",
        type=Path,
        default=None,
        help=(
            "Specific option-feature metadata JSON. "
            "Defaults to the newest metadata file."
        ),
    )

    return parser


def main() -> None:
    args = (
        build_parser()
        .parse_args()
    )

    result = (
        validate_option_feature_artifacts(
            feature_root=
                args.feature_root,

            feature_metadata=
                args.feature_metadata,
        )
    )

    _print_result(
        result
    )


if __name__ == "__main__":
    main()
