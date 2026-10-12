from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("Capture summary must be a JSON object")
    return dict(value)


def build_price_matrix_from_capture_summary(
    *,
    summary_path: Path,
    processed_root: Path,
    expected_symbols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary_path = Path(summary_path)
    processed_root = Path(processed_root)
    summary = _read_json(summary_path)

    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Capture summary artifacts must be a list")

    series_by_symbol: dict[str, pd.Series] = {}
    artifact_hashes: dict[str, str] = {}

    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            continue
        if str(artifact.get("dataset", "")).strip() != "price_history":
            continue

        scope = str(artifact.get("scope", "")).strip().upper()
        normalized = artifact.get("normalized")
        if not scope or not isinstance(normalized, Mapping):
            raise ValueError("Price-history artifact is missing scope/normalized metadata")

        relative_path = str(normalized.get("relative_path", "")).strip()
        expected_sha = str(normalized.get("normalized_sha256", "")).strip().lower()
        if not relative_path or len(expected_sha) != 64:
            raise ValueError(f"Invalid normalized metadata for {scope}")

        path = processed_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Normalized price-history file is missing: {path}")

        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"Normalized price-history hash mismatch for {scope}: "
                f"expected {expected_sha}, got {actual_sha}"
            )

        frame = pd.read_csv(path, low_memory=False)
        required = {"datetime_utc", "close"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{scope} price history missing columns: {missing}")

        timestamps = pd.to_datetime(frame["datetime_utc"], errors="coerce", utc=True)
        close = pd.to_numeric(frame["close"], errors="coerce")
        valid = timestamps.notna() & close.notna() & close.gt(0)
        if not valid.any():
            raise ValueError(f"{scope} has no usable price-history rows")

        values = pd.Series(
            close.loc[valid].to_numpy(dtype=float),
            index=timestamps.loc[valid].dt.tz_convert("UTC").dt.tz_localize(None),
            name=scope,
        )
        values = values[~values.index.duplicated(keep="last")].sort_index()
        series_by_symbol[scope] = values
        artifact_hashes[scope] = actual_sha

    if not series_by_symbol:
        raise ValueError("Capture summary contains no price_history artifacts")

    if expected_symbols is not None:
        expected = [str(symbol).strip().upper() for symbol in expected_symbols]
        missing = [symbol for symbol in expected if symbol not in series_by_symbol]
        extra = [symbol for symbol in series_by_symbol if symbol not in expected]
        if missing or extra:
            raise ValueError(
                f"Captured price-history universe mismatch; missing={missing}, extra={extra}"
            )
        ordered_symbols = expected
    else:
        ordered_symbols = sorted(series_by_symbol)

    matrix = pd.concat(
        [series_by_symbol[symbol] for symbol in ordered_symbols],
        axis=1,
        join="outer",
    ).sort_index()
    matrix.columns = ordered_symbols

    metadata = {
        "capture_run_id": str(summary.get("run_id", "")),
        "capture_summary_sha256": sha256_file(summary_path),
        "symbol_count": len(ordered_symbols),
        "symbols": ordered_symbols,
        "normalized_price_history_sha256": artifact_hashes,
    }
    return matrix, metadata


def write_price_matrix_from_capture_summary(
    *,
    summary_path: Path,
    processed_root: Path,
    output_path: Path,
    expected_symbols: list[str] | None = None,
) -> tuple[Path, Path]:
    output_path = Path(output_path)
    manifest_path = output_path.with_suffix(".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Price matrix output already exists: {output_path}")

    matrix, metadata = build_price_matrix_from_capture_summary(
        summary_path=summary_path,
        processed_root=processed_root,
        expected_symbols=expected_symbols,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_path, lineterminator="\n")
    metadata = {
        **metadata,
        "price_matrix_sha256": sha256_file(output_path),
        "rows": int(len(matrix)),
        "columns": int(len(matrix.columns)),
        "immutable": True,
    }
    manifest_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path, manifest_path
