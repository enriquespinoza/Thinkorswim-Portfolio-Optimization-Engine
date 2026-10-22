from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PROCESSED_DATA_DIR
from scripts.capture_schwab_market_data import capture_schwab_market_data
from src.data.scanner_candidate_prices import (
    write_price_matrix_from_capture_summary,
)
from src.integrations.schwab.auth_bootstrap import (
    create_market_data_schwab_client,
    load_schwab_auth_profile,
)


DEFAULT_PRIVATE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "private"
    / "personal_portfolio"
)

DEFAULT_PROCESSED_ROOT = (
    PROCESSED_DATA_DIR
    / "schwab"
)


def load_benchmark_symbols(
    *,
    personal_snapshot: Path,
    optimizer_allocation: Path,
    as_of_session: str,
    benchmark_symbol: str,
) -> list[str]:
    personal = pd.read_csv(personal_snapshot, low_memory=False)
    required_personal = {"symbol", "market_value"}
    missing_personal = sorted(required_personal.difference(personal.columns))
    if missing_personal:
        raise ValueError(
            f"personal snapshot missing required columns: {missing_personal}"
        )

    personal["market_value"] = pd.to_numeric(
        personal["market_value"],
        errors="coerce",
    )
    if personal["market_value"].isna().any():
        raise ValueError("personal snapshot market_value must be numeric")

    personal = personal.loc[
        personal["market_value"].gt(0)
    ].copy()

    allocation = pd.read_csv(optimizer_allocation, low_memory=False)
    required = {"as_of_session", "symbol"}
    missing = sorted(required.difference(allocation.columns))
    if missing:
        raise ValueError(
            f"optimizer allocation missing required columns: {missing}"
        )

    selected = allocation.loc[
        allocation["as_of_session"].astype(str).eq(as_of_session)
    ].copy()
    if selected.empty:
        raise ValueError(
            f"no optimizer allocation rows found for {as_of_session}"
        )

    symbols = {
        str(value).strip().upper()
        for value in personal["symbol"]
        if str(value).strip()
    }
    symbols.update(
        str(value).strip().upper()
        for value in selected["symbol"]
        if str(value).strip()
    )
    symbols.add(str(benchmark_symbol).strip().upper())

    if "" in symbols:
        symbols.remove("")

    if not symbols:
        raise ValueError("benchmark symbol universe is empty")

    return sorted(symbols)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Capture Schwab daily price history for the Personal Portfolio "
            "Benchmark V1 universe and build a private synchronized price matrix."
        )
    )
    parser.add_argument("--as-of-session", required=True)
    parser.add_argument(
        "--personal-snapshot",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--optimizer-allocation",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--benchmark-symbol",
        default="SPY",
    )
    parser.add_argument(
        "--private-output-root",
        type=Path,
        default=DEFAULT_PRIVATE_ROOT,
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
    )
    args = parser.parse_args()

    symbols = load_benchmark_symbols(
        personal_snapshot=args.personal_snapshot,
        optimizer_allocation=args.optimizer_allocation,
        as_of_session=args.as_of_session,
        benchmark_symbol=args.benchmark_symbol,
    )

    config = load_schwab_auth_profile("market_data")
    client = create_market_data_schwab_client(config)

    capture = capture_schwab_market_data(
        client=client,
        symbols=symbols,
        datasets=("price_history",),
        processed_root=args.processed_root,
        price_history_period_type="YEAR",
        price_history_period="TWO_YEARS",
        price_history_frequency_type="DAILY",
        price_history_frequency="DAILY",
        price_history_extended_hours=False,
    )

    output_dir = Path(args.private_output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_stem = args.personal_snapshot.stem.lower()
    scope = (
        "all_accounts"
        if "all_accounts" in snapshot_stem
        else "single_account"
    )
    output_path = (
        output_dir
        / f"benchmark_prices_{scope}_{args.as_of_session}.csv"
    )

    matrix_path, manifest_path = write_price_matrix_from_capture_summary(
        summary_path=Path(capture["summary_path"]),
        processed_root=args.processed_root,
        output_path=output_path,
        expected_symbols=symbols,
    )

    print("=" * 88)
    print("PERSONAL PORTFOLIO BENCHMARK PRICE MATRIX")
    print("=" * 88)
    print("Session:", args.as_of_session)
    print("Symbols:", len(symbols))
    print("Benchmark:", args.benchmark_symbol.upper())
    print("Market-data profile: market_data")
    print("Saved ->", matrix_path)
    print("Saved ->", manifest_path)
    print("=" * 88)


if __name__ == "__main__":
    main()
