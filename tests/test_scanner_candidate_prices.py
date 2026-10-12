from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.scanner_candidate_prices import (
    build_price_matrix_from_capture_summary,
    sha256_file,
    write_price_matrix_from_capture_summary,
)


def _write_price_artifact(root: Path, symbol: str, offset: float) -> tuple[Path, str]:
    path = root / "price_history" / f"{symbol}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "datetime_utc": [
                "2026-10-09T04:00:00+00:00",
                "2026-10-12T04:00:00+00:00",
            ],
            "close": [100.0 + offset, 101.0 + offset],
        }
    )
    frame.to_csv(path, index=False, lineterminator="\n")
    return path, sha256_file(path)


def _write_summary(tmp_path: Path) -> tuple[Path, Path]:
    processed = tmp_path / "processed"
    a_path, a_sha = _write_price_artifact(processed, "AAA", 0.0)
    b_path, b_sha = _write_price_artifact(processed, "BBB", 10.0)

    summary = {
        "run_id": "run-123",
        "artifacts": [
            {
                "scope": "AAA",
                "dataset": "price_history",
                "normalized": {
                    "relative_path": str(a_path.relative_to(processed)).replace("\\", "/"),
                    "normalized_sha256": a_sha,
                },
            },
            {
                "scope": "BBB",
                "dataset": "price_history",
                "normalized": {
                    "relative_path": str(b_path.relative_to(processed)).replace("\\", "/"),
                    "normalized_sha256": b_sha,
                },
            },
        ],
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path, processed


def test_build_price_matrix_verifies_artifacts_and_symbol_order(tmp_path: Path) -> None:
    summary_path, processed = _write_summary(tmp_path)

    matrix, metadata = build_price_matrix_from_capture_summary(
        summary_path=summary_path,
        processed_root=processed,
        expected_symbols=["BBB", "AAA"],
    )

    assert matrix.columns.tolist() == ["BBB", "AAA"]
    assert len(matrix) == 2
    assert metadata["capture_run_id"] == "run-123"
    assert metadata["symbol_count"] == 2


def test_price_matrix_rejects_hash_mismatch(tmp_path: Path) -> None:
    summary_path, processed = _write_summary(tmp_path)
    path = processed / "price_history" / "AAA.csv"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        build_price_matrix_from_capture_summary(
            summary_path=summary_path,
            processed_root=processed,
            expected_symbols=["AAA", "BBB"],
        )


def test_write_price_matrix_is_immutable(tmp_path: Path) -> None:
    summary_path, processed = _write_summary(tmp_path)
    output = tmp_path / "candidate_prices.csv"

    output_path, manifest_path = write_price_matrix_from_capture_summary(
        summary_path=summary_path,
        processed_root=processed,
        output_path=output,
        expected_symbols=["AAA", "BBB"],
    )

    assert output_path.is_file()
    assert manifest_path.is_file()

    with pytest.raises(FileExistsError):
        write_price_matrix_from_capture_summary(
            summary_path=summary_path,
            processed_root=processed,
            output_path=output,
            expected_symbols=["AAA", "BBB"],
        )
