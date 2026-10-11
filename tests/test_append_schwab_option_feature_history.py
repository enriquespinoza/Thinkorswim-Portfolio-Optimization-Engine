from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.append_schwab_option_feature_history import (
    append_option_feature_history,
)


RUN_ID = "run123"
FEATURE_SCHEMA = "1.2.0"
QUALITY_POLICY = "1.0.0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _write_fixture(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
]:
    feature_root = (
        tmp_path
        / "features"
    )
    feature_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    validated_name = (
        f"{RUN_ID}__v{FEATURE_SCHEMA}"
        "__validated_option_features.csv"
    )

    validated_path = (
        feature_root
        / validated_name
    )

    frame = pd.DataFrame(
        [
            {
                "feature_schema_version":
                    FEATURE_SCHEMA,
                "underlying_symbol":
                    "SPY",
                "spot":
                    500.0,
                "contract_count_total":
                    200,
                "contract_count_eligible":
                    190,
                "eligible_contract_ratio":
                    0.95,
                "eligible_expiration_count":
                    12,
                "quality_policy_version":
                    QUALITY_POLICY,
                "quality_symbol_status":
                    "PASS",
                "quality_symbol_reasons":
                    "",
                "iv_median":
                    20.0,
            },
            {
                "feature_schema_version":
                    FEATURE_SCHEMA,
                "underlying_symbol":
                    "TLT",
                "spot":
                    100.0,
                "contract_count_total":
                    180,
                "contract_count_eligible":
                    170,
                "eligible_contract_ratio":
                    0.9444,
                "eligible_expiration_count":
                    11,
                "quality_policy_version":
                    QUALITY_POLICY,
                "quality_symbol_status":
                    "PASS",
                "quality_symbol_reasons":
                    "",
                "iv_median":
                    15.0,
            },
        ]
    )

    frame.to_csv(
        validated_path,
        index=False,
        lineterminator="\n",
    )

    feature_metadata_path = (
        feature_root
        / f"{RUN_ID}__v{FEATURE_SCHEMA}"
        "__option_feature_metadata.json"
    )

    feature_metadata = {
        "feature_schema_version":
            FEATURE_SCHEMA,
        "run_id":
            RUN_ID,
        "run_started_at_utc":
            "2026-10-10T13:30:00+00:00",
        "validation":
            {
                "validation_id":
                    "input-validation-123",
            },
    }

    feature_metadata_path.write_text(
        json.dumps(
            feature_metadata,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    validation_metadata_path = (
        feature_root
        / f"{RUN_ID}__v{FEATURE_SCHEMA}"
        "__option_feature_validation_metadata.json"
    )

    validation_metadata = {
        "feature_schema_version":
            FEATURE_SCHEMA,
        "quality_policy_version":
            QUALITY_POLICY,
        "run_id":
            RUN_ID,
        "source_feature_metadata":
            str(
                feature_metadata_path
            ),
        "outputs":
            {
                "validated_option_features":
                    {
                        "relative_path":
                            validated_name,
                        "sha256":
                            _sha256(
                                validated_path
                            ),
                    },
            },
    }

    validation_metadata_path.write_text(
        json.dumps(
            validation_metadata,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return (
        feature_root,
        validation_metadata_path,
    )


def test_append_creates_history_with_provenance(
    tmp_path: Path,
):
    (
        feature_root,
        validation_metadata,
    ) = _write_fixture(
        tmp_path
    )

    history_path = (
        tmp_path
        / "history"
        / "option_feature_history_v1.csv"
    )

    result = append_option_feature_history(
        feature_root=
            feature_root,
        validation_metadata=
            validation_metadata,
        history_path=
            history_path,
    )

    assert (
        result[
            "status"
        ]
        == "APPENDED"
    )
    assert (
        result[
            "appended_rows"
        ]
        == 2
    )
    assert history_path.is_file()

    history = pd.read_csv(
        history_path,
        low_memory=False,
    )

    assert len(
        history
    ) == 2

    assert set(
        history[
            "underlying_symbol"
        ]
    ) == {
        "SPY",
        "TLT",
    }

    assert set(
        history[
            "run_id"
        ]
    ) == {
        RUN_ID,
    }

    assert set(
        history[
            "capture_timestamp_utc"
        ]
    ) == {
        "2026-10-10T13:30:00+00:00",
    }

    assert set(
        history[
            "history_schema_version"
        ].astype(
            str
        )
    ) == {
        "1.0.0",
    }


def test_reappend_is_idempotent(
    tmp_path: Path,
):
    (
        feature_root,
        validation_metadata,
    ) = _write_fixture(
        tmp_path
    )

    history_path = (
        tmp_path
        / "history.csv"
    )

    first = append_option_feature_history(
        feature_root=
            feature_root,
        validation_metadata=
            validation_metadata,
        history_path=
            history_path,
    )

    second = append_option_feature_history(
        feature_root=
            feature_root,
        validation_metadata=
            validation_metadata,
        history_path=
            history_path,
    )

    assert first[
        "status"
    ] == "APPENDED"

    assert second[
        "status"
    ] == "EXISTS"

    assert second[
        "appended_rows"
    ] == 0

    assert second[
        "history_rows_after"
    ] == 2


def test_dry_run_does_not_write(
    tmp_path: Path,
):
    (
        feature_root,
        validation_metadata,
    ) = _write_fixture(
        tmp_path
    )

    history_path = (
        tmp_path
        / "history.csv"
    )

    result = append_option_feature_history(
        feature_root=
            feature_root,
        validation_metadata=
            validation_metadata,
        history_path=
            history_path,
        dry_run=True,
    )

    assert result[
        "status"
    ] == "DRY_RUN"

    assert not history_path.exists()


def test_history_schema_drift_fails_closed(
    tmp_path: Path,
):
    (
        feature_root,
        validation_metadata,
    ) = _write_fixture(
        tmp_path
    )

    history_path = (
        tmp_path
        / "history.csv"
    )

    append_option_feature_history(
        feature_root=
            feature_root,
        validation_metadata=
            validation_metadata,
        history_path=
            history_path,
    )

    history = pd.read_csv(
        history_path,
        low_memory=False,
    )

    history[
        "unexpected_column"
    ] = 1

    history.to_csv(
        history_path,
        index=False,
        lineterminator="\n",
    )

    with pytest.raises(
        RuntimeError,
        match="schema drift",
    ):
        append_option_feature_history(
            feature_root=
                feature_root,
            validation_metadata=
                validation_metadata,
            history_path=
                history_path,
        )
