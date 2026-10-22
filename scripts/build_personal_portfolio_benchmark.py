from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.personal_portfolio_benchmark import (
    BENCHMARK_VERSION,
    DEFAULT_BENCHMARK_SYMBOL,
    DEFAULT_LOOKBACK_RETURNS,
)
from src.research.portfolio_comparison import (
    build_summary,
    normalize_weights,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_personal_weights(path: Path) -> tuple[pd.Series, float]:
    frame = pd.read_csv(path, low_memory=False)
    required = {"symbol", "market_value"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"personal snapshot missing columns: {missing}")

    frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
    if frame["symbol"].eq("").any() or frame["symbol"].duplicated().any():
        raise ValueError("personal snapshot symbols must be non-empty and unique")

    values = pd.to_numeric(frame["market_value"], errors="coerce")
    if values.isna().any() or (values < 0).any():
        raise ValueError("market_value must be non-negative numeric")

    total = float(values.sum())
    if total <= 0:
        raise ValueError("personal portfolio value must be positive")

    return pd.Series(values.to_numpy() / total, index=frame["symbol"]), total


def load_optimizer_weights(
    allocation_path: Path,
    manifest_path: Path,
    as_of_session: str,
) -> tuple[pd.Series, str]:
    allocation = pd.read_csv(allocation_path, low_memory=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allocation_sha = sha256_file(allocation_path)

    if manifest.get("candidate_allocation_sha256") != allocation_sha:
        raise ValueError("optimizer allocation SHA-256 does not match manifest")
    if manifest.get("as_of_session") != as_of_session:
        raise ValueError("optimizer manifest session mismatch")

    selected = allocation.loc[
        allocation["as_of_session"].astype(str).eq(as_of_session)
    ].copy()
    if selected.empty:
        raise ValueError(f"no optimizer allocation for {as_of_session}")

    selected["symbol"] = selected["symbol"].astype(str).str.upper().str.strip()
    weights = pd.Series(
        pd.to_numeric(selected["target_weight"], errors="raise").to_numpy(dtype=float),
        index=selected["symbol"],
    )
    return normalize_weights(weights), allocation_sha


def load_returns(
    prices_path: Path,
    symbols: list[str],
    as_of_session: str,
    lookback_returns: int,
) -> pd.DataFrame:
    prices = pd.read_csv(prices_path, index_col=0, low_memory=False)
    prices.index = pd.to_datetime(prices.index)
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()

    required = sorted(set(symbols))
    missing = [symbol for symbol in required if symbol not in prices.columns]
    if missing:
        raise ValueError(f"price matrix missing symbols: {missing}")

    selected = prices.loc[prices.index <= pd.Timestamp(as_of_session), required]
    selected = selected.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    selected = selected.tail(lookback_returns + 1)

    returns = selected.pct_change(fill_method=None).dropna(how="any")
    if len(returns) != lookback_returns:
        raise ValueError(
            f"need {lookback_returns} synchronized returns, found {len(returns)}"
        )
    return returns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Personal Portfolio Benchmark V1 aggregate validation summary."
    )
    parser.add_argument("--as-of-session", required=True)
    parser.add_argument("--personal-snapshot", type=Path, required=True)
    parser.add_argument("--optimizer-allocation", type=Path, required=True)
    parser.add_argument("--optimizer-manifest", type=Path, required=True)
    parser.add_argument("--prices-csv", type=Path, required=True)
    parser.add_argument("--benchmark-symbol", default=DEFAULT_BENCHMARK_SYMBOL)
    parser.add_argument("--lookback-returns", type=int, default=DEFAULT_LOOKBACK_RETURNS)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("reports/validation/portfolio_comparison"),
    )
    args = parser.parse_args()

    personal_weights, portfolio_value = load_personal_weights(args.personal_snapshot)
    optimizer_weights, allocation_sha = load_optimizer_weights(
        args.optimizer_allocation,
        args.optimizer_manifest,
        args.as_of_session,
    )

    positive_personal = personal_weights[
        personal_weights.gt(0)
    ]
    positive_optimizer = optimizer_weights[
        optimizer_weights.gt(0)
    ]

    symbols = sorted(
        set(positive_personal.index)
        | set(positive_optimizer.index)
        | {args.benchmark_symbol.upper()}
    )
    returns = load_returns(
        args.prices_csv,
        symbols,
        args.as_of_session,
        args.lookback_returns,
    )

    summary = build_summary(
        as_of_session=args.as_of_session,
        personal_weights=personal_weights,
        optimizer_weights=optimizer_weights,
        returns=returns,
        benchmark_symbol=args.benchmark_symbol,
    )
    summary.update(
        {
            "personal_portfolio_value": portfolio_value,
            "portfolio_snapshot_sha256": sha256_file(args.personal_snapshot),
            "system_allocation_sha256": allocation_sha,
            "system_allocation_manifest_sha256": sha256_file(args.optimizer_manifest),
            "price_data_sha256": sha256_file(args.prices_csv),
            "lookback_returns": args.lookback_returns,
            "benchmark_symbol": args.benchmark_symbol.upper(),
            "contains_symbol_level_personal_holdings": False,
            "benchmark_contract": BENCHMARK_VERSION,
        }
    )

    output_dir = args.output_root / args.as_of_session
    output_path = output_dir / "summary.json"
    if output_path.exists():
        raise FileExistsError(
            f"benchmark summary already exists for {args.as_of_session}; refusing to overwrite"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=" * 88)
    print("PERSONAL PORTFOLIO BENCHMARK V1")
    print("=" * 88)
    for key in (
        "as_of_session",
        "personal_portfolio_value",
        "personal_portfolio_volatility",
        "optimizer_portfolio_volatility",
        "equal_weight_volatility",
        "spy_volatility",
        "personal_max_drawdown",
        "optimizer_max_drawdown",
        "personal_concentration",
        "optimizer_concentration",
        "tracking_error",
        "portfolio_overlap",
        "weight_difference",
        "turnover",
    ):
        print(f"{key}: {summary[key]}")
    print("Read-only research:", summary["read_only_research"])
    print("Order submission permitted:", summary["order_submission_permitted"])
    print("Saved ->", output_path)


if __name__ == "__main__":
    main()
