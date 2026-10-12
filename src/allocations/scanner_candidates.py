from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.scanner_allocation import (
    ALLOCATION_COLUMNS,
    ALLOCATION_CONTRACT_VERSION,
    COVARIANCE_ESTIMATOR,
    DEFAULT_LOOKBACK_RETURNS,
    DEFAULT_MAX_WEIGHT,
    DEFAULT_MIN_WEIGHT,
    OPTIMIZER_NAME,
    OPTIMIZER_VERSION,
)
from src.optimizers.minimum_variance import optimize_minimum_variance
from src.portfolio.returns import calculate_daily_returns
from src.risk.covariance import calculate_ledoit_wolf_covariance


_REQUIRED_HISTORY_COLUMNS = {
    "source_candidate_csv_sha256",
    "source_candidate_manifest_sha256",
    "as_of_session",
    "scanner_snapshot_sha256",
    "symbol",
    "scanner_rank",
    "scanner_score",
    "scanner_percentile",
    "close",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_dataframe_hash(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    payload = normalized.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_candidate_snapshot(
    candidate_history: pd.DataFrame,
    *,
    as_of_session: str,
) -> pd.DataFrame:
    missing = sorted(_REQUIRED_HISTORY_COLUMNS.difference(candidate_history.columns))
    if missing:
        raise ValueError(f"Candidate history missing required columns: {missing}")

    selected = candidate_history.loc[
        candidate_history["as_of_session"].astype(str).eq(as_of_session)
    ].copy()

    if selected.empty:
        raise ValueError(f"No scanner candidates found for session {as_of_session}")

    snapshots = selected["scanner_snapshot_sha256"].astype(str).unique().tolist()
    if len(snapshots) != 1:
        raise ValueError(
            f"Session {as_of_session} contains multiple scanner snapshots: {snapshots}"
        )

    selected["scanner_rank"] = pd.to_numeric(
        selected["scanner_rank"], errors="raise"
    ).astype(int)
    selected = selected.sort_values(["scanner_rank", "symbol"], kind="stable")

    expected_ranks = list(range(1, len(selected) + 1))
    if selected["scanner_rank"].tolist() != expected_ranks:
        raise ValueError(
            f"Scanner ranks must be contiguous 1..{len(selected)} for allocation"
        )

    if selected["symbol"].astype(str).str.upper().duplicated().any():
        raise ValueError("Candidate snapshot contains duplicate symbols")

    return selected.reset_index(drop=True)


def prepare_return_matrix(
    prices: pd.DataFrame,
    *,
    symbols: list[str],
    as_of_session: str,
    lookback_returns: int = DEFAULT_LOOKBACK_RETURNS,
) -> pd.DataFrame:
    if lookback_returns < 2:
        raise ValueError("lookback_returns must be at least 2")

    missing = [symbol for symbol in symbols if symbol not in prices.columns]
    if missing:
        raise ValueError(f"Price data missing candidate symbols: {missing}")

    selected = prices.loc[:, symbols].copy()
    selected.index = pd.to_datetime(selected.index)
    index = pd.DatetimeIndex(selected.index)
    if index.tz is not None:
        selected.index = index.tz_convert("UTC").tz_localize(None)

    session_dates = pd.DatetimeIndex(selected.index).normalize()
    if session_dates.duplicated().any():
        raise ValueError(
            "Price matrix contains multiple observations per calendar date; "
            "daily price history is required"
        )

    selected.index = session_dates
    cutoff = pd.Timestamp(as_of_session).normalize()
    selected = selected.loc[selected.index <= cutoff]
    selected = selected.apply(pd.to_numeric, errors="coerce")
    selected = selected.replace([np.inf, -np.inf], np.nan).dropna(how="any")

    required_prices = lookback_returns + 1
    if len(selected) < required_prices:
        raise ValueError(
            "Insufficient synchronized price history: "
            f"need at least {required_prices} rows, found {len(selected)}"
        )

    selected = selected.tail(required_prices)
    returns = calculate_daily_returns(selected, dropna=True)

    if len(returns) != lookback_returns:
        raise ValueError(
            f"Expected {lookback_returns} synchronized returns, found {len(returns)}"
        )

    return returns


def build_scanner_candidate_allocation(
    *,
    candidate_history: pd.DataFrame,
    prices: pd.DataFrame,
    as_of_session: str,
    source_candidate_history_sha256: str,
    source_prices_sha256: str,
    lookback_returns: int = DEFAULT_LOOKBACK_RETURNS,
    min_weight: float = DEFAULT_MIN_WEIGHT,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates = load_candidate_snapshot(
        candidate_history,
        as_of_session=as_of_session,
    )
    symbols = candidates["symbol"].astype(str).str.upper().tolist()

    returns = prepare_return_matrix(
        prices,
        symbols=symbols,
        as_of_session=as_of_session,
        lookback_returns=lookback_returns,
    )
    covariance, shrinkage = calculate_ledoit_wolf_covariance(returns)

    result = optimize_minimum_variance(
        covariance=covariance,
        min_weight=min_weight,
        max_weight=max_weight,
    )
    weights = result.weights.reindex(symbols)
    annualized_volatility = pd.Series(
        np.sqrt(np.diag(covariance.to_numpy(dtype=float))),
        index=covariance.index,
    ).reindex(symbols)

    if weights.isna().any():
        raise RuntimeError("Optimizer returned missing candidate weights")
    if not np.isclose(float(weights.sum()), 1.0, atol=1e-9):
        raise RuntimeError("Target weights do not sum to 1.0")
    if (weights < min_weight - 1e-10).any():
        raise RuntimeError("Optimizer violated minimum weight")
    if (weights > max_weight + 1e-10).any():
        raise RuntimeError("Optimizer violated maximum weight")

    candidate_snapshot_sha256 = stable_dataframe_hash(candidates)
    scanner_snapshot_sha256 = str(candidates["scanner_snapshot_sha256"].iloc[0])
    source_candidate_csv_values = (
        candidates["source_candidate_csv_sha256"].astype(str).unique().tolist()
    )
    source_candidate_manifest_values = (
        candidates["source_candidate_manifest_sha256"].astype(str).unique().tolist()
    )
    if len(source_candidate_csv_values) != 1 or len(source_candidate_manifest_values) != 1:
        raise ValueError("Candidate snapshot must originate from one immutable handoff")

    allocation = pd.DataFrame(
        {
            "allocation_contract_version": ALLOCATION_CONTRACT_VERSION,
            "as_of_session": as_of_session,
            "scanner_snapshot_sha256": scanner_snapshot_sha256,
            "source_candidate_history_sha256": source_candidate_history_sha256,
            "source_candidate_csv_sha256": source_candidate_csv_values[0],
            "source_candidate_manifest_sha256": source_candidate_manifest_values[0],
            "symbol": symbols,
            "scanner_rank": candidates["scanner_rank"].to_numpy(dtype=int),
            "scanner_score": pd.to_numeric(
                candidates["scanner_score"], errors="raise"
            ).to_numpy(dtype=float),
            "scanner_percentile": pd.to_numeric(
                candidates["scanner_percentile"], errors="raise"
            ).to_numpy(dtype=float),
            "close": pd.to_numeric(
                candidates["close"], errors="raise"
            ).to_numpy(dtype=float),
            "optimizer_name": OPTIMIZER_NAME,
            "optimizer_version": OPTIMIZER_VERSION,
            "covariance_estimator": COVARIANCE_ESTIMATOR,
            "lookback_returns": int(lookback_returns),
            "annualized_volatility": annualized_volatility.to_numpy(dtype=float),
            "target_weight": weights.to_numpy(dtype=float),
        },
        columns=ALLOCATION_COLUMNS,
    )

    manifest = {
        "allocation_contract_version": ALLOCATION_CONTRACT_VERSION,
        "as_of_session": as_of_session,
        "scanner_snapshot_sha256": scanner_snapshot_sha256,
        "candidate_snapshot_sha256": candidate_snapshot_sha256,
        "source_candidate_history_sha256": source_candidate_history_sha256,
        "source_candidate_csv_sha256": source_candidate_csv_values[0],
        "source_candidate_manifest_sha256": source_candidate_manifest_values[0],
        "source_prices_sha256": source_prices_sha256,
        "candidate_count": int(len(allocation)),
        "optimizer_name": OPTIMIZER_NAME,
        "optimizer_version": OPTIMIZER_VERSION,
        "covariance_estimator": COVARIANCE_ESTIMATOR,
        "ledoit_wolf_shrinkage": float(shrinkage),
        "lookback_returns": int(lookback_returns),
        "min_weight": float(min_weight),
        "max_weight": float(max_weight),
        "portfolio_volatility": float(result.portfolio_volatility),
        "weights_sum": float(allocation["target_weight"].sum()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "immutable": True,
    }
    return allocation, manifest


def write_scanner_candidate_allocation(
    *,
    candidate_history_path: Path,
    prices_path: Path,
    as_of_session: str,
    output_root: Path,
    lookback_returns: int = DEFAULT_LOOKBACK_RETURNS,
    min_weight: float = DEFAULT_MIN_WEIGHT,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> tuple[Path, Path, dict[str, Any]]:
    candidate_history_path = Path(candidate_history_path)
    prices_path = Path(prices_path)
    if not candidate_history_path.is_file():
        raise FileNotFoundError(f"Candidate history is missing: {candidate_history_path}")
    if not prices_path.is_file():
        raise FileNotFoundError(f"Price input is missing: {prices_path}")

    history_sha = sha256_file(candidate_history_path)
    prices_sha = sha256_file(prices_path)

    candidate_history = pd.read_csv(candidate_history_path, low_memory=False)
    prices = pd.read_csv(prices_path, index_col=0)

    allocation, manifest = build_scanner_candidate_allocation(
        candidate_history=candidate_history,
        prices=prices,
        as_of_session=as_of_session,
        source_candidate_history_sha256=history_sha,
        source_prices_sha256=prices_sha,
        lookback_returns=lookback_returns,
        min_weight=min_weight,
        max_weight=max_weight,
    )

    output_dir = Path(output_root) / as_of_session
    allocation_path = output_dir / "candidate_allocation.csv"
    manifest_path = output_dir / "manifest.json"

    if allocation_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Scanner allocation already exists for {as_of_session}; refusing to overwrite"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    allocation.to_csv(allocation_path, index=False, lineterminator="\n")
    manifest = {
        **manifest,
        "candidate_allocation_sha256": sha256_file(allocation_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return allocation_path, manifest_path, manifest
