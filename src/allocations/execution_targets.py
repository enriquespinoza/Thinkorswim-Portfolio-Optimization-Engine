from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config.execution_handoff import (
    EXECUTION_STATES,
    EXPECTED_ALLOCATION_CONTRACT_VERSION,
    PORTFOLIO_TARGET_HANDOFF_VERSION,
    PAPER_POSITION_SNAPSHOT_CONTRACT,
    REFERENCE_PRICE_SOURCE_ALLOCATION,
    REFERENCE_PRICE_SOURCE_POSITION,
    SHARE_ROUNDING_POLICY,
    TARGET_BUILDER_VERSION,
    TARGET_COLUMNS,
)


_REQUIRED_ALLOCATION_COLUMNS = {
    "allocation_contract_version",
    "as_of_session",
    "scanner_snapshot_sha256",
    "symbol",
    "scanner_rank",
    "close",
    "target_weight",
}

_REQUIRED_POSITION_COLUMNS = {
    "symbol",
    "current_position",
    "reference_price",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"allocation manifest contains invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("allocation manifest must contain a JSON object")
    return value


def _validate_allocation(
    allocation: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    allocation_sha256: str,
) -> tuple[str, str]:
    missing = sorted(_REQUIRED_ALLOCATION_COLUMNS.difference(allocation.columns))
    if missing:
        raise ValueError(f"allocation missing required columns: {missing}")
    if allocation.empty:
        raise ValueError("allocation is empty")

    contracts = allocation["allocation_contract_version"].astype(str).unique().tolist()
    if contracts != [EXPECTED_ALLOCATION_CONTRACT_VERSION]:
        raise ValueError(
            "allocation contract mismatch: "
            f"expected {EXPECTED_ALLOCATION_CONTRACT_VERSION}, found {contracts}"
        )

    manifest_contract = str(manifest.get("allocation_contract_version", ""))
    if manifest_contract != EXPECTED_ALLOCATION_CONTRACT_VERSION:
        raise ValueError("allocation manifest contract does not match execution handoff")

    expected_sha = str(manifest.get("candidate_allocation_sha256", ""))
    if expected_sha != allocation_sha256:
        raise ValueError("allocation SHA-256 does not match allocation manifest")

    sessions = allocation["as_of_session"].astype(str).unique().tolist()
    if len(sessions) != 1:
        raise ValueError(f"allocation must contain exactly one session: {sessions}")
    as_of_session = sessions[0]
    if str(manifest.get("as_of_session", "")) != as_of_session:
        raise ValueError("allocation session does not match manifest")

    snapshots = allocation["scanner_snapshot_sha256"].astype(str).unique().tolist()
    if len(snapshots) != 1:
        raise ValueError("allocation must contain exactly one scanner snapshot")
    scanner_snapshot_sha256 = snapshots[0]
    if str(manifest.get("scanner_snapshot_sha256", "")) != scanner_snapshot_sha256:
        raise ValueError("scanner snapshot SHA does not match manifest")

    symbols = allocation["symbol"].astype(str).str.upper().str.strip()
    if symbols.eq("").any() or symbols.duplicated().any():
        raise ValueError("allocation symbols must be non-empty and unique")

    weights = pd.to_numeric(allocation["target_weight"], errors="coerce")
    prices = pd.to_numeric(allocation["close"], errors="coerce")
    ranks = pd.to_numeric(allocation["scanner_rank"], errors="coerce")

    if weights.isna().any() or (weights < -1e-12).any() or (weights > 1 + 1e-12).any():
        raise ValueError("allocation target_weight must be finite within [0, 1]")
    if prices.isna().any() or (prices <= 0).any():
        raise ValueError("allocation close must be positive")
    if ranks.isna().any() or (ranks <= 0).any():
        raise ValueError("scanner_rank must be positive")

    if not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-9):
        raise ValueError("allocation target weights must sum to 1.0")

    return as_of_session, scanner_snapshot_sha256


def _validate_positions(positions: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(_REQUIRED_POSITION_COLUMNS.difference(positions.columns))
    if missing:
        raise ValueError(f"positions snapshot missing required columns: {missing}")

    result = positions.copy()
    result["symbol"] = result["symbol"].astype(str).str.upper().str.strip()
    if result["symbol"].eq("").any() or result["symbol"].duplicated().any():
        raise ValueError("position symbols must be non-empty and unique")

    qty = pd.to_numeric(result["current_position"], errors="coerce")
    price = pd.to_numeric(result["reference_price"], errors="coerce")
    if qty.isna().any() or (qty < 0).any() or not qty.map(lambda x: float(x).is_integer()).all():
        raise ValueError("current_position must contain non-negative whole shares")
    if price.isna().any() or (price <= 0).any():
        raise ValueError("position reference_price must be positive")

    result["current_position"] = qty.astype(int)
    result["reference_price"] = price.astype(float)
    return result


def _validate_position_manifest(
    *,
    manifest: dict[str, Any],
    positions_path: Path,
    positions: pd.DataFrame,
    positions_sha256: str,
) -> tuple[float, dict[str, Any]]:
    if str(manifest.get("position_snapshot_contract", "")) != (
        PAPER_POSITION_SNAPSHOT_CONTRACT
    ):
        raise ValueError("position snapshot manifest contract mismatch")

    if manifest.get("immutable") is not True:
        raise ValueError("position snapshot manifest must be immutable")
    if manifest.get("broker_submission_permitted") is not False:
        raise ValueError(
            "position snapshot manifest must forbid broker submission"
        )

    if str(manifest.get("positions_sha256", "")) != positions_sha256:
        raise ValueError(
            "position snapshot CSV SHA-256 does not match manifest"
        )

    filename = str(manifest.get("positions_filename", "")).strip()
    if filename and filename != positions_path.name:
        raise ValueError(
            "position snapshot filename does not match manifest"
        )

    try:
        position_count = int(manifest.get("position_count"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "position snapshot position_count must be an integer"
        ) from exc
    if position_count != len(positions):
        raise ValueError(
            "position snapshot row count does not match manifest"
        )

    try:
        portfolio_capital = float(manifest.get("portfolio_capital"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "position snapshot portfolio_capital must be numeric"
        ) from exc
    if not math.isfinite(portfolio_capital) or portfolio_capital <= 0:
        raise ValueError(
            "position snapshot portfolio_capital must be positive and finite"
        )

    lineage_fields = (
        "paper_state_sha256",
        "market_snapshot_sha256",
    )
    for field in lineage_fields:
        value = str(manifest.get(field, "")).strip().lower()
        if (
            len(value) != 64
            or any(ch not in "0123456789abcdef" for ch in value)
        ):
            raise ValueError(
                f"position snapshot {field} must be a SHA-256 digest"
            )

    try:
        paper_state_version = int(manifest.get("paper_state_version"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "position snapshot paper_state_version must be an integer"
        ) from exc
    if paper_state_version < 1:
        raise ValueError(
            "position snapshot paper_state_version must be positive"
        )

    lineage = {
        "position_snapshot_contract": PAPER_POSITION_SNAPSHOT_CONTRACT,
        "paper_state_version": paper_state_version,
        "paper_state_sha256": str(
            manifest["paper_state_sha256"]
        ).lower(),
        "paper_previous_state_sha256": manifest.get(
            "paper_previous_state_sha256"
        ),
        "source_optimizer_artifact_sha256": manifest.get(
            "source_optimizer_artifact_sha256"
        ),
        "position_market_snapshot_sha256": str(
            manifest["market_snapshot_sha256"]
        ).lower(),
        "paper_cash": float(manifest.get("cash", 0.0)),
        "paper_position_market_value": float(
            manifest.get("position_market_value", 0.0)
        ),
    }
    return portfolio_capital, lineage


def build_execution_safe_targets(
    *,
    allocation: pd.DataFrame,
    allocation_manifest: dict[str, Any],
    allocation_sha256: str,
    allocation_manifest_sha256: str,
    positions: pd.DataFrame,
    portfolio_capital: float,
    execution_state: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not math.isfinite(float(portfolio_capital)) or portfolio_capital <= 0:
        raise ValueError("portfolio_capital must be positive and finite")

    execution_state = str(execution_state).upper().strip()
    if execution_state not in EXECUTION_STATES:
        raise ValueError(
            f"execution_state must be one of {EXECUTION_STATES}, found {execution_state!r}"
        )

    as_of_session, scanner_snapshot_sha256 = _validate_allocation(
        allocation,
        allocation_manifest,
        allocation_sha256=allocation_sha256,
    )
    positions = _validate_positions(positions)

    alloc = allocation.copy()
    alloc["symbol"] = alloc["symbol"].astype(str).str.upper().str.strip()
    alloc["target_weight"] = pd.to_numeric(alloc["target_weight"], errors="raise").astype(float)
    alloc["close"] = pd.to_numeric(alloc["close"], errors="raise").astype(float)
    alloc["scanner_rank"] = pd.to_numeric(alloc["scanner_rank"], errors="raise").astype(int)

    allocation_map = alloc.set_index("symbol")
    position_map = positions.set_index("symbol")
    symbols = sorted(set(allocation_map.index) | set(position_map.index))

    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        in_allocation = symbol in allocation_map.index
        in_positions = symbol in position_map.index

        if in_allocation:
            target_weight = float(allocation_map.at[symbol, "target_weight"])
            reference_price = float(allocation_map.at[symbol, "close"])
            reference_price_source = REFERENCE_PRICE_SOURCE_ALLOCATION
            scanner_rank = int(allocation_map.at[symbol, "scanner_rank"])
        else:
            target_weight = 0.0
            reference_price = float(position_map.at[symbol, "reference_price"])
            reference_price_source = REFERENCE_PRICE_SOURCE_POSITION
            scanner_rank = pd.NA

        current_position = (
            int(position_map.at[symbol, "current_position"])
            if in_positions
            else 0
        )

        target_dollar_value = float(portfolio_capital) * target_weight
        target_shares = math.floor(target_dollar_value / reference_price)
        target_share_notional = float(target_shares) * reference_price
        realized_target_weight = target_share_notional / float(portfolio_capital)
        weight_rounding_error = target_weight - realized_target_weight
        share_delta = target_shares - current_position

        if target_weight <= 1e-15:
            target_status = "EXIT" if current_position > 0 else "ZERO_WEIGHT"
        elif target_shares == 0:
            target_status = "ROUNDING_TO_ZERO"
        else:
            target_status = "TARGETED"

        if share_delta > 0:
            trade_side = "BUY"
        elif share_delta < 0:
            trade_side = "SELL"
        else:
            trade_side = "HOLD"

        rows.append(
            {
                "handoff_contract_version": PORTFOLIO_TARGET_HANDOFF_VERSION,
                "target_builder_version": TARGET_BUILDER_VERSION,
                "as_of_session": as_of_session,
                "allocation_contract_version": EXPECTED_ALLOCATION_CONTRACT_VERSION,
                "allocation_sha256": allocation_sha256,
                "allocation_manifest_sha256": allocation_manifest_sha256,
                "scanner_snapshot_sha256": scanner_snapshot_sha256,
                "symbol": symbol,
                "scanner_rank": scanner_rank,
                "approved_target_weight": target_weight,
                "reference_price": reference_price,
                "reference_price_source": reference_price_source,
                "portfolio_capital": float(portfolio_capital),
                "target_dollar_value": target_dollar_value,
                "target_shares": int(target_shares),
                "target_share_notional": target_share_notional,
                "realized_target_weight": realized_target_weight,
                "weight_rounding_error": weight_rounding_error,
                "target_status": target_status,
                "current_position": int(current_position),
                "share_delta": int(share_delta),
                "trade_side": trade_side,
                "execution_state": execution_state,
            }
        )

    targets = pd.DataFrame(rows, columns=TARGET_COLUMNS)
    target_notional = float(targets["target_share_notional"].sum())
    residual_cash = float(portfolio_capital) - target_notional
    realized_weight_sum = float(targets["realized_target_weight"].sum())
    total_weight_rounding_error = float(targets["weight_rounding_error"].sum())

    manifest = {
        "handoff_contract_version": PORTFOLIO_TARGET_HANDOFF_VERSION,
        "target_builder_version": TARGET_BUILDER_VERSION,
        "as_of_session": as_of_session,
        "allocation_contract_version": EXPECTED_ALLOCATION_CONTRACT_VERSION,
        "allocation_sha256": allocation_sha256,
        "allocation_manifest_sha256": allocation_manifest_sha256,
        "scanner_snapshot_sha256": scanner_snapshot_sha256,
        "portfolio_capital": float(portfolio_capital),
        "execution_state": execution_state,
        "share_rounding_policy": SHARE_ROUNDING_POLICY,
        "row_count": int(len(targets)),
        "target_symbol_count": int((targets["approved_target_weight"] > 0).sum()),
        "targeted_count": int((targets["target_status"] == "TARGETED").sum()),
        "zero_weight_count": int((targets["target_status"] == "ZERO_WEIGHT").sum()),
        "rounding_to_zero_count": int(
            (targets["target_status"] == "ROUNDING_TO_ZERO").sum()
        ),
        "exit_count": int((targets["target_status"] == "EXIT").sum()),
        "buy_count": int((targets["trade_side"] == "BUY").sum()),
        "sell_count": int((targets["trade_side"] == "SELL").sum()),
        "hold_count": int((targets["trade_side"] == "HOLD").sum()),
        "target_notional": target_notional,
        "residual_cash": residual_cash,
        "realized_weight_sum": realized_weight_sum,
        "total_weight_rounding_error": total_weight_rounding_error,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "immutable": True,
        "broker_submission_permitted": False,
    }
    return targets, manifest


def write_execution_safe_targets(
    *,
    allocation_path: Path,
    allocation_manifest_path: Path,
    positions_path: Path,
    portfolio_capital: float | None,
    execution_state: str,
    output_root: Path,
    positions_manifest_path: Path | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    allocation_path = Path(allocation_path)
    allocation_manifest_path = Path(allocation_manifest_path)
    positions_path = Path(positions_path)
    positions_manifest_path = (
        Path(positions_manifest_path)
        if positions_manifest_path is not None
        else None
    )

    if not allocation_path.is_file():
        raise FileNotFoundError(f"allocation is missing: {allocation_path}")
    if not allocation_manifest_path.is_file():
        raise FileNotFoundError(f"allocation manifest is missing: {allocation_manifest_path}")
    if not positions_path.is_file():
        raise FileNotFoundError(f"positions snapshot is missing: {positions_path}")
    if (
        positions_manifest_path is not None
        and not positions_manifest_path.is_file()
    ):
        raise FileNotFoundError(
            f"positions manifest is missing: {positions_manifest_path}"
        )

    allocation_sha = sha256_file(allocation_path)
    allocation_manifest_sha = sha256_file(allocation_manifest_path)
    allocation = pd.read_csv(allocation_path, low_memory=False)
    allocation_manifest = _read_manifest(allocation_manifest_path)
    positions = pd.read_csv(positions_path, low_memory=False)
    positions = _validate_positions(positions)
    positions_sha = sha256_file(positions_path)

    position_lineage: dict[str, Any] = {}
    positions_manifest_sha: str | None = None

    if positions_manifest_path is not None:
        positions_manifest = _read_manifest(
            positions_manifest_path
        )
        manifest_capital, position_lineage = (
            _validate_position_manifest(
                manifest=positions_manifest,
                positions_path=positions_path,
                positions=positions,
                positions_sha256=positions_sha,
            )
        )
        positions_manifest_sha = sha256_file(
            positions_manifest_path
        )

        if portfolio_capital is None:
            portfolio_capital = manifest_capital
        elif not math.isclose(
            float(portfolio_capital),
            manifest_capital,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError(
                "portfolio_capital does not match positions manifest"
            )

    if portfolio_capital is None:
        raise ValueError(
            "portfolio_capital is required unless --positions-manifest "
            "provides it"
        )

    targets, manifest = build_execution_safe_targets(
        allocation=allocation,
        allocation_manifest=allocation_manifest,
        allocation_sha256=allocation_sha,
        allocation_manifest_sha256=allocation_manifest_sha,
        positions=positions,
        portfolio_capital=portfolio_capital,
        execution_state=execution_state,
    )

    output_dir = (
        Path(output_root)
        / str(manifest["as_of_session"])
        / str(manifest["target_builder_version"])
    )
    target_path = output_dir / "execution_safe_targets.csv"
    manifest_path = output_dir / "manifest.json"

    if target_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"execution target already exists for {manifest['as_of_session']}; refusing to overwrite"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    targets.to_csv(target_path, index=False, lineterminator="\n")

    manifest = {
        **manifest,
        "positions_sha256": positions_sha,
        "positions_manifest_sha256": positions_manifest_sha,
        **position_lineage,
        "execution_safe_targets_sha256": sha256_file(target_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target_path, manifest_path, manifest
