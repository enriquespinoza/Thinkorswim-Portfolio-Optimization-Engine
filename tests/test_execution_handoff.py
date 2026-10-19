from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from config.execution_handoff import (
    PORTFOLIO_TARGET_HANDOFF_VERSION,
    TARGET_COLUMNS,
)
from src.allocations.execution_targets import (
    build_execution_safe_targets,
    sha256_file,
    write_execution_safe_targets,
)


def _allocation() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "allocation_contract_version": "scanner-candidate-allocation-v1",
                "as_of_session": "2026-10-16",
                "scanner_snapshot_sha256": "a" * 64,
                "symbol": "AAA",
                "scanner_rank": 1,
                "close": 100.0,
                "target_weight": 0.60,
            },
            {
                "allocation_contract_version": "scanner-candidate-allocation-v1",
                "as_of_session": "2026-10-16",
                "scanner_snapshot_sha256": "a" * 64,
                "symbol": "BBB",
                "scanner_rank": 2,
                "close": 200.0,
                "target_weight": 0.40,
            },
        ]
    )


def _manifest(allocation_sha: str) -> dict:
    return {
        "allocation_contract_version": "scanner-candidate-allocation-v1",
        "as_of_session": "2026-10-16",
        "scanner_snapshot_sha256": "a" * 64,
        "candidate_allocation_sha256": allocation_sha,
    }


def _positions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "AAA", "current_position": 40, "reference_price": 101.0},
            {"symbol": "CCC", "current_position": 5, "reference_price": 50.0},
        ]
    )


def test_build_execution_targets_includes_dropped_position() -> None:
    allocation = _allocation()
    targets, manifest = build_execution_safe_targets(
        allocation=allocation,
        allocation_manifest=_manifest("b" * 64),
        allocation_sha256="b" * 64,
        allocation_manifest_sha256="c" * 64,
        positions=_positions(),
        portfolio_capital=10_000.0,
        execution_state="DRY_RUN",
    )

    assert tuple(targets.columns) == TARGET_COLUMNS
    assert targets["symbol"].tolist() == ["AAA", "BBB", "CCC"]

    aaa = targets.loc[targets["symbol"].eq("AAA")].iloc[0]
    bbb = targets.loc[targets["symbol"].eq("BBB")].iloc[0]
    ccc = targets.loc[targets["symbol"].eq("CCC")].iloc[0]

    assert aaa["target_shares"] == 60
    assert aaa["target_share_notional"] == 6000.0
    assert aaa["realized_target_weight"] == 0.60
    assert aaa["weight_rounding_error"] == 0.0
    assert aaa["target_status"] == "TARGETED"
    assert aaa["current_position"] == 40
    assert aaa["share_delta"] == 20
    assert aaa["trade_side"] == "BUY"

    assert bbb["target_shares"] == 20
    assert bbb["target_share_notional"] == 4000.0
    assert bbb["realized_target_weight"] == 0.40
    assert bbb["weight_rounding_error"] == 0.0
    assert bbb["target_status"] == "TARGETED"
    assert bbb["current_position"] == 0
    assert bbb["share_delta"] == 20
    assert bbb["trade_side"] == "BUY"

    assert ccc["approved_target_weight"] == 0.0
    assert ccc["target_shares"] == 0
    assert ccc["target_share_notional"] == 0.0
    assert ccc["realized_target_weight"] == 0.0
    assert ccc["weight_rounding_error"] == 0.0
    assert ccc["target_status"] == "EXIT"
    assert ccc["share_delta"] == -5
    assert ccc["trade_side"] == "SELL"

    assert manifest["broker_submission_permitted"] is False
    assert manifest["execution_state"] == "DRY_RUN"
    assert manifest["targeted_count"] == 2
    assert manifest["exit_count"] == 1
    assert manifest["rounding_to_zero_count"] == 0
    assert manifest["realized_weight_sum"] == 1.0
    assert manifest["total_weight_rounding_error"] == 0.0




def test_rounding_to_zero_is_distinct_from_zero_weight() -> None:
    allocation = pd.DataFrame(
        [
            {
                "allocation_contract_version": "scanner-candidate-allocation-v1",
                "as_of_session": "2026-10-16",
                "scanner_snapshot_sha256": "a" * 64,
                "symbol": "HIGH",
                "scanner_rank": 1,
                "close": 1_000.0,
                "target_weight": 0.01,
            },
            {
                "allocation_contract_version": "scanner-candidate-allocation-v1",
                "as_of_session": "2026-10-16",
                "scanner_snapshot_sha256": "a" * 64,
                "symbol": "ZERO",
                "scanner_rank": 2,
                "close": 100.0,
                "target_weight": 0.0,
            },
            {
                "allocation_contract_version": "scanner-candidate-allocation-v1",
                "as_of_session": "2026-10-16",
                "scanner_snapshot_sha256": "a" * 64,
                "symbol": "CORE",
                "scanner_rank": 3,
                "close": 100.0,
                "target_weight": 0.99,
            },
        ]
    )

    targets, manifest = build_execution_safe_targets(
        allocation=allocation,
        allocation_manifest=_manifest("b" * 64),
        allocation_sha256="b" * 64,
        allocation_manifest_sha256="c" * 64,
        positions=pd.DataFrame(columns=["symbol", "current_position", "reference_price"]),
        portfolio_capital=10_000.0,
        execution_state="DRY_RUN",
    )

    high = targets.loc[targets["symbol"].eq("HIGH")].iloc[0]
    zero = targets.loc[targets["symbol"].eq("ZERO")].iloc[0]

    assert high["approved_target_weight"] == 0.01
    assert high["target_shares"] == 0
    assert high["target_share_notional"] == 0.0
    assert high["realized_target_weight"] == 0.0
    assert high["weight_rounding_error"] == 0.01
    assert high["target_status"] == "ROUNDING_TO_ZERO"
    assert high["trade_side"] == "HOLD"

    assert zero["approved_target_weight"] == 0.0
    assert zero["target_status"] == "ZERO_WEIGHT"
    assert zero["trade_side"] == "HOLD"

    assert manifest["rounding_to_zero_count"] == 1
    assert manifest["zero_weight_count"] == 1
    assert manifest["realized_weight_sum"] == 0.99
    assert manifest["total_weight_rounding_error"] == pytest.approx(0.01)


def test_reference_price_uses_allocation_for_target_names() -> None:
    allocation = _allocation()
    targets, _ = build_execution_safe_targets(
        allocation=allocation,
        allocation_manifest=_manifest("b" * 64),
        allocation_sha256="b" * 64,
        allocation_manifest_sha256="c" * 64,
        positions=_positions(),
        portfolio_capital=10_000.0,
        execution_state="PAPER",
    )
    aaa = targets.loc[targets["symbol"].eq("AAA")].iloc[0]
    assert aaa["reference_price"] == 100.0
    assert aaa["reference_price_source"] == "scanner_allocation_close"


def test_rejects_allocation_hash_mismatch() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        build_execution_safe_targets(
            allocation=_allocation(),
            allocation_manifest=_manifest("d" * 64),
            allocation_sha256="b" * 64,
            allocation_manifest_sha256="c" * 64,
            positions=_positions(),
            portfolio_capital=10_000.0,
            execution_state="DRY_RUN",
        )


def test_rejects_live_execution_state() -> None:
    with pytest.raises(ValueError, match="execution_state"):
        build_execution_safe_targets(
            allocation=_allocation(),
            allocation_manifest=_manifest("b" * 64),
            allocation_sha256="b" * 64,
            allocation_manifest_sha256="c" * 64,
            positions=_positions(),
            portfolio_capital=10_000.0,
            execution_state="LIVE",
        )


def test_write_execution_target_is_immutable(tmp_path: Path) -> None:
    allocation_path = tmp_path / "candidate_allocation.csv"
    manifest_path = tmp_path / "allocation_manifest.json"
    positions_path = tmp_path / "positions.csv"
    output_root = tmp_path / "out"

    _allocation().to_csv(allocation_path, index=False, lineterminator="\n")
    allocation_sha = sha256_file(allocation_path)
    manifest_path.write_text(
        json.dumps(_manifest(allocation_sha), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _positions().to_csv(positions_path, index=False, lineterminator="\n")

    target_path, output_manifest_path, output_manifest = write_execution_safe_targets(
        allocation_path=allocation_path,
        allocation_manifest_path=manifest_path,
        positions_path=positions_path,
        portfolio_capital=10_000.0,
        execution_state="DRY_RUN",
        output_root=output_root,
    )

    assert output_manifest["handoff_contract_version"] == PORTFOLIO_TARGET_HANDOFF_VERSION
    assert output_manifest["execution_safe_targets_sha256"] == sha256_file(target_path)
    assert output_manifest["positions_sha256"] == sha256_file(positions_path)
    assert output_manifest["target_builder_version"] == "1.2.0"
    assert target_path.parent.name == "1.2.0"

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_execution_safe_targets(
            allocation_path=allocation_path,
            allocation_manifest_path=manifest_path,
            positions_path=positions_path,
            portfolio_capital=10_000.0,
            execution_state="DRY_RUN",
            output_root=output_root,
        )



def test_write_execution_target_from_verified_paper_positions_manifest(
    tmp_path: Path,
) -> None:
    allocation_path = tmp_path / "candidate_allocation.csv"
    allocation_manifest_path = tmp_path / "allocation_manifest.json"
    positions_path = tmp_path / "managed_sleeve_positions.csv"
    positions_manifest_path = tmp_path / "positions_manifest.json"
    output_root = tmp_path / "out"

    _allocation().to_csv(
        allocation_path,
        index=False,
        lineterminator="\n",
    )
    allocation_sha = sha256_file(allocation_path)
    allocation_manifest_path.write_text(
        json.dumps(
            _manifest(allocation_sha),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    _positions().to_csv(
        positions_path,
        index=False,
        lineterminator="\n",
    )
    positions_sha = sha256_file(positions_path)

    positions_manifest = {
        "position_snapshot_contract":
            "paper-optimizer-position-snapshot-v1",
        "generated_at_utc":
            "2026-10-19T18:00:00+00:00",
        "immutable": True,
        "broker_submission_permitted": False,
        "paper_state_version": 2,
        "paper_state_sha256": "d" * 64,
        "paper_previous_state_sha256": "e" * 64,
        "source_optimizer_artifact_sha256": "f" * 64,
        "market_snapshot_sha256": "1" * 64,
        "cash": 1000.0,
        "position_market_value": 9000.0,
        "portfolio_capital": 10000.0,
        "position_count": 2,
        "positions_filename": positions_path.name,
        "positions_sha256": positions_sha,
    }
    positions_manifest_path.write_text(
        json.dumps(
            positions_manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    target_path, manifest_path, manifest = (
        write_execution_safe_targets(
            allocation_path=allocation_path,
            allocation_manifest_path=allocation_manifest_path,
            positions_path=positions_path,
            positions_manifest_path=positions_manifest_path,
            portfolio_capital=None,
            execution_state="PAPER",
            output_root=output_root,
        )
    )

    assert target_path.is_file()
    assert manifest_path.is_file()
    assert manifest["portfolio_capital"] == 10000.0
    assert manifest["paper_state_version"] == 2
    assert manifest["paper_state_sha256"] == "d" * 64
    assert (
        manifest["position_market_snapshot_sha256"]
        == "1" * 64
    )
    assert (
        manifest["positions_manifest_sha256"]
        == sha256_file(positions_manifest_path)
    )
    assert manifest["target_builder_version"] == "1.2.0"


def test_rejects_tampered_paper_positions_snapshot(
    tmp_path: Path,
) -> None:
    allocation_path = tmp_path / "candidate_allocation.csv"
    allocation_manifest_path = tmp_path / "allocation_manifest.json"
    positions_path = tmp_path / "managed_sleeve_positions.csv"
    positions_manifest_path = tmp_path / "positions_manifest.json"

    _allocation().to_csv(
        allocation_path,
        index=False,
        lineterminator="\n",
    )
    allocation_sha = sha256_file(allocation_path)
    allocation_manifest_path.write_text(
        json.dumps(_manifest(allocation_sha)),
        encoding="utf-8",
    )

    _positions().to_csv(
        positions_path,
        index=False,
        lineterminator="\n",
    )

    positions_manifest_path.write_text(
        json.dumps(
            {
                "position_snapshot_contract":
                    "paper-optimizer-position-snapshot-v1",
                "immutable": True,
                "broker_submission_permitted": False,
                "paper_state_version": 2,
                "paper_state_sha256": "d" * 64,
                "market_snapshot_sha256": "1" * 64,
                "portfolio_capital": 10000.0,
                "position_count": 2,
                "positions_filename": positions_path.name,
                "positions_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="CSV SHA-256"):
        write_execution_safe_targets(
            allocation_path=allocation_path,
            allocation_manifest_path=allocation_manifest_path,
            positions_path=positions_path,
            positions_manifest_path=positions_manifest_path,
            portfolio_capital=None,
            execution_state="PAPER",
            output_root=tmp_path / "out",
        )
