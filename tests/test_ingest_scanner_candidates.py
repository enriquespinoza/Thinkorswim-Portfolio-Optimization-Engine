from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from config.scanner_handoff import (
    CANDIDATE_COLUMNS,
    EXPECTED_SCANNER_ALPHA_SPEC_VERSION,
    EXPECTED_SCANNER_PROVIDER,
    HANDOFF_CONTRACT_VERSION,
    PRIMARY_SIGNAL_NAME,
    SELECTION_POLICY,
)
from scripts.ingest_scanner_candidates import (
    ingest_scanner_candidates,
)


def _sha256(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _write_handoff(
    tmp_path: Path,
    *,
    bad_candidate_hash: bool = False,
    noncontiguous_rank: bool = False,
) -> tuple[
    Path,
    Path,
]:
    candidate_csv = (
        tmp_path
        / "scanner_candidates.csv"
    )

    candidate_manifest = (
        tmp_path
        / "scanner_candidates_manifest.json"
    )

    generated_at = (
        "2026-10-12T00:30:00+00:00"
    )

    scanner_snapshot_sha256 = (
        "a" * 64
    )

    source_alpha_manifest_sha256 = (
        "b" * 64
    )

    ranks = (
        [1, 3]
        if noncontiguous_rank
        else [1, 2]
    )

    rows = [
        {
            "handoff_contract_version":
                HANDOFF_CONTRACT_VERSION,
            "scanner_alpha_spec_version":
                EXPECTED_SCANNER_ALPHA_SPEC_VERSION,
            "scanner_provider":
                EXPECTED_SCANNER_PROVIDER,
            "as_of_session":
                "2026-10-11",
            "generated_at_utc":
                generated_at,
            "scanner_snapshot_sha256":
                scanner_snapshot_sha256,
            "source_alpha_manifest_sha256":
                source_alpha_manifest_sha256,
            "source_timestamp_utc":
                "2026-10-11T20:00:00+00:00",
            "symbol":
                "AAPL",
            "scanner_rank":
                ranks[0],
            "scanner_score":
                2.5,
            "scanner_percentile":
                100.0,
            "scanner_eligible":
                True,
            "close":
                250.0,
            "primary_signal_name":
                PRIMARY_SIGNAL_NAME,
            "primary_signal_value":
                0.24,
            "ema_50_to_200":
                1.08,
        },
        {
            "handoff_contract_version":
                HANDOFF_CONTRACT_VERSION,
            "scanner_alpha_spec_version":
                EXPECTED_SCANNER_ALPHA_SPEC_VERSION,
            "scanner_provider":
                EXPECTED_SCANNER_PROVIDER,
            "as_of_session":
                "2026-10-11",
            "generated_at_utc":
                generated_at,
            "scanner_snapshot_sha256":
                scanner_snapshot_sha256,
            "source_alpha_manifest_sha256":
                source_alpha_manifest_sha256,
            "source_timestamp_utc":
                "2026-10-11T20:00:00+00:00",
            "symbol":
                "MSFT",
            "scanner_rank":
                ranks[1],
            "scanner_score":
                2.1,
            "scanner_percentile":
                98.0,
            "scanner_eligible":
                True,
            "close":
                520.0,
            "primary_signal_name":
                PRIMARY_SIGNAL_NAME,
            "primary_signal_value":
                0.21,
            "ema_50_to_200":
                1.05,
        },
    ]

    frame = pd.DataFrame(
        rows,
        columns=
            CANDIDATE_COLUMNS,
    )

    frame.to_csv(
        candidate_csv,
        index=False,
        lineterminator="\n",
    )

    candidate_sha = (
        "c" * 64
        if bad_candidate_hash
        else _sha256(
            candidate_csv
        )
    )

    manifest = {
        "handoff_contract_version":
            HANDOFF_CONTRACT_VERSION,
        "scanner_alpha_spec_version":
            EXPECTED_SCANNER_ALPHA_SPEC_VERSION,
        "scanner_provider":
            EXPECTED_SCANNER_PROVIDER,
        "as_of_session":
            "2026-10-11",
        "generated_at_utc":
            generated_at,
        "scanner_snapshot_sha256":
            scanner_snapshot_sha256,
        "source_alpha_manifest_sha256":
            source_alpha_manifest_sha256,
        "candidate_count":
            2,
        "candidate_limit":
            20,
        "selection_policy":
            SELECTION_POLICY,
        "candidate_csv_sha256":
            candidate_sha,
        "immutable":
            True,
    }

    candidate_manifest.write_text(
        json.dumps(
            manifest,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return (
        candidate_csv,
        candidate_manifest,
    )


def test_dry_run_validates_without_writing(
    tmp_path: Path,
):
    (
        candidate_csv,
        candidate_manifest,
    ) = _write_handoff(
        tmp_path
    )

    history_path = (
        tmp_path
        / "history.csv"
    )

    result = ingest_scanner_candidates(
        candidate_csv=
            candidate_csv,
        candidate_manifest=
            candidate_manifest,
        history_path=
            history_path,
        dry_run=True,
    )

    assert (
        result[
            "status"
        ]
        == "DRY_RUN"
    )

    assert (
        result[
            "incoming_rows"
        ]
        == 2
    )

    assert not history_path.exists()


def test_ingest_appends_candidate_history(
    tmp_path: Path,
):
    (
        candidate_csv,
        candidate_manifest,
    ) = _write_handoff(
        tmp_path
    )

    history_path = (
        tmp_path
        / "history"
        / "scanner_candidate_history_v1.csv"
    )

    result = ingest_scanner_candidates(
        candidate_csv=
            candidate_csv,
        candidate_manifest=
            candidate_manifest,
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

    history = pd.read_csv(
        history_path,
        low_memory=False,
    )

    assert history[
        "symbol"
    ].tolist() == [
        "AAPL",
        "MSFT",
    ]

    assert history[
        "scanner_rank"
    ].tolist() == [
        1,
        2,
    ]

    assert set(
        history[
            "candidate_history_schema_version"
        ].astype(
            str
        )
    ) == {
        "1.0.0",
    }


def test_reingest_is_idempotent(
    tmp_path: Path,
):
    (
        candidate_csv,
        candidate_manifest,
    ) = _write_handoff(
        tmp_path
    )

    history_path = (
        tmp_path
        / "history.csv"
    )

    first = ingest_scanner_candidates(
        candidate_csv=
            candidate_csv,
        candidate_manifest=
            candidate_manifest,
        history_path=
            history_path,
    )

    second = ingest_scanner_candidates(
        candidate_csv=
            candidate_csv,
        candidate_manifest=
            candidate_manifest,
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


def test_bad_candidate_hash_fails_closed(
    tmp_path: Path,
):
    (
        candidate_csv,
        candidate_manifest,
    ) = _write_handoff(
        tmp_path,
        bad_candidate_hash=True,
    )

    with pytest.raises(
        RuntimeError,
        match="SHA-256",
    ):
        ingest_scanner_candidates(
            candidate_csv=
                candidate_csv,
            candidate_manifest=
                candidate_manifest,
            history_path=
                tmp_path
                / "history.csv",
        )


def test_noncontiguous_top_rank_set_fails_closed(
    tmp_path: Path,
):
    (
        candidate_csv,
        candidate_manifest,
    ) = _write_handoff(
        tmp_path,
        noncontiguous_rank=True,
    )

    with pytest.raises(
        RuntimeError,
        match="contiguous top-ranked",
    ):
        ingest_scanner_candidates(
            candidate_csv=
                candidate_csv,
            candidate_manifest=
                candidate_manifest,
            history_path=
                tmp_path
                / "history.csv",
        )
