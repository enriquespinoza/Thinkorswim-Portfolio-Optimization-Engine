from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.scanner_allocation import (
    ALLOCATION_COLUMNS,
    ALLOCATION_CONTRACT_VERSION,
)
from src.allocations.scanner_candidates import (
    build_scanner_candidate_allocation,
    sha256_file,
    write_scanner_candidate_allocation,
)


def _candidate_history(rows: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_history_schema_version": "1.0.0",
            "source_candidate_csv_sha256": "a" * 64,
            "source_candidate_manifest_sha256": "b" * 64,
            "handoff_contract_version": "scanner-candidate-handoff-v1",
            "scanner_alpha_spec_version": "scanner-alpha-v1",
            "scanner_provider": "schwab",
            "as_of_session": "2026-10-12",
            "generated_at_utc": "2026-10-12T21:00:00+00:00",
            "scanner_snapshot_sha256": "c" * 64,
            "source_alpha_manifest_sha256": "d" * 64,
            "source_timestamp_utc": "2026-10-12T04:00:00+00:00",
            "symbol": [f"S{i:02d}" for i in range(rows)],
            "scanner_rank": np.arange(1, rows + 1),
            "scanner_score": np.linspace(5.0, 1.0, rows),
            "scanner_percentile": np.linspace(100.0, 96.0, rows),
            "scanner_eligible": True,
            "close": np.linspace(50.0, 100.0, rows),
            "primary_signal_name": "momentum_252d_ex_20d",
            "primary_signal_value": np.linspace(2.0, 0.5, rows),
            "ema_50_to_200": np.linspace(0.4, 0.1, rows),
        }
    )


def _prices(rows: int = 20) -> pd.DataFrame:
    dates = pd.bdate_range(end="2026-10-12", periods=300)
    x = np.arange(len(dates), dtype=float)
    data = {}
    for i in range(rows):
        drift = 0.0003 + i * 0.00001
        cycle = 0.002 * np.sin(x / (7.0 + i * 0.1) + i)
        daily = drift + cycle
        data[f"S{i:02d}"] = 100.0 * np.cumprod(1.0 + daily)
    return pd.DataFrame(data, index=dates)


def test_allocation_preserves_scanner_metadata_and_constraints() -> None:
    history = _candidate_history()
    prices = _prices()

    allocation, manifest = build_scanner_candidate_allocation(
        candidate_history=history,
        prices=prices,
        as_of_session="2026-10-12",
        source_candidate_history_sha256="e" * 64,
        source_prices_sha256="f" * 64,
    )

    assert tuple(allocation.columns) == ALLOCATION_COLUMNS
    assert allocation["symbol"].tolist() == history["symbol"].tolist()
    assert allocation["scanner_rank"].tolist() == list(range(1, 21))
    assert np.allclose(allocation["scanner_score"], history["scanner_score"])
    assert np.isclose(allocation["target_weight"].sum(), 1.0, atol=1e-9)
    assert (allocation["target_weight"] >= -1e-10).all()
    assert (allocation["target_weight"] <= 0.10 + 1e-10).all()
    assert allocation["allocation_contract_version"].eq(
        ALLOCATION_CONTRACT_VERSION
    ).all()
    assert manifest["candidate_count"] == 20
    assert manifest["scanner_snapshot_sha256"] == "c" * 64
    assert manifest["source_candidate_history_sha256"] == "e" * 64
    assert manifest["source_prices_sha256"] == "f" * 64
    assert manifest["immutable"] is True


def test_allocation_does_not_use_scanner_score_as_optimizer_input() -> None:
    history = _candidate_history()
    prices = _prices()

    baseline, _ = build_scanner_candidate_allocation(
        candidate_history=history,
        prices=prices,
        as_of_session="2026-10-12",
        source_candidate_history_sha256="e" * 64,
        source_prices_sha256="f" * 64,
    )

    changed = history.copy()
    changed["scanner_score"] = changed["scanner_score"].iloc[::-1].to_numpy()
    changed["scanner_percentile"] = changed["scanner_percentile"].iloc[::-1].to_numpy()

    rescored, _ = build_scanner_candidate_allocation(
        candidate_history=changed,
        prices=prices,
        as_of_session="2026-10-12",
        source_candidate_history_sha256="e" * 64,
        source_prices_sha256="f" * 64,
    )

    assert np.allclose(
        baseline["target_weight"],
        rescored["target_weight"],
        atol=1e-10,
    )


def test_write_allocation_is_immutable_and_hashes_file(tmp_path: Path) -> None:
    history_path = tmp_path / "scanner_candidate_history_v1.csv"
    prices_path = tmp_path / "prices.csv"
    output_root = tmp_path / "reports"

    _candidate_history().to_csv(history_path, index=False, lineterminator="\n")
    _prices().to_csv(prices_path, lineterminator="\n")

    allocation_path, manifest_path, manifest = write_scanner_candidate_allocation(
        candidate_history_path=history_path,
        prices_path=prices_path,
        as_of_session="2026-10-12",
        output_root=output_root,
    )

    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["candidate_allocation_sha256"] == sha256_file(allocation_path)
    assert saved_manifest["candidate_allocation_sha256"] == sha256_file(
        allocation_path
    )
    assert saved_manifest["source_candidate_history_sha256"] == sha256_file(
        history_path
    )
    assert saved_manifest["source_prices_sha256"] == sha256_file(prices_path)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_scanner_candidate_allocation(
            candidate_history_path=history_path,
            prices_path=prices_path,
            as_of_session="2026-10-12",
            output_root=output_root,
        )


def test_allocation_requires_sufficient_synchronized_history() -> None:
    history = _candidate_history()
    prices = _prices().tail(200)

    with pytest.raises(ValueError, match="Insufficient synchronized price history"):
        build_scanner_candidate_allocation(
            candidate_history=history,
            prices=prices,
            as_of_session="2026-10-12",
            source_candidate_history_sha256="e" * 64,
            source_prices_sha256="f" * 64,
        )


def test_prepare_return_matrix_includes_as_of_session() -> None:
    history = _candidate_history()
    prices = _prices()

    allocation, manifest = build_scanner_candidate_allocation(
        candidate_history=history,
        prices=prices,
        as_of_session="2026-10-12",
        source_candidate_history_sha256="e" * 64,
        source_prices_sha256="f" * 64,
    )

    assert manifest["as_of_session"] == "2026-10-12"
    assert len(allocation) == 20


def test_allocation_rejects_intraday_price_matrix() -> None:
    history = _candidate_history()
    base = _prices()
    extra = base.iloc[[-1]].copy()
    extra.index = [pd.Timestamp("2026-10-12 15:00:00")]
    prices = pd.concat([base, extra]).sort_index()

    with pytest.raises(ValueError, match="multiple observations per calendar date"):
        build_scanner_candidate_allocation(
            candidate_history=history,
            prices=prices,
            as_of_session="2026-10-12",
            source_candidate_history_sha256="e" * 64,
            source_prices_sha256="f" * 64,
        )
