from __future__ import annotations

import argparse
from pathlib import Path

from config.settings import DATA_DIR, GENERATED_REPORTS_DIR
from src.allocations.scanner_candidates import write_scanner_candidate_allocation


DEFAULT_HISTORY_PATH = (
    DATA_DIR / "history" / "scanner" / "scanner_candidate_history_v1.csv"
)
DEFAULT_OUTPUT_ROOT = GENERATED_REPORTS_DIR / "scanner_allocations"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an immutable minimum-variance allocation for one "
            "scanner candidate snapshot."
        )
    )
    parser.add_argument("--as-of-session", required=True)
    parser.add_argument(
        "--candidate-history",
        type=Path,
        default=DEFAULT_HISTORY_PATH,
    )
    parser.add_argument(
        "--prices-csv",
        type=Path,
        required=True,
        help=(
            "Wide daily close/adjusted-close CSV with date index and one "
            "column per scanner candidate. Data after as-of-session is ignored."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--lookback-returns", type=int, default=252)
    parser.add_argument("--min-weight", type=float, default=0.0)
    parser.add_argument("--max-weight", type=float, default=0.10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    allocation_path, manifest_path, manifest = write_scanner_candidate_allocation(
        candidate_history_path=args.candidate_history,
        prices_path=args.prices_csv,
        as_of_session=args.as_of_session,
        output_root=args.output_root,
        lookback_returns=args.lookback_returns,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
    )

    print()
    print("=" * 88)
    print("SCANNER CANDIDATE ALLOCATION V1")
    print("=" * 88)
    print("Session:", manifest["as_of_session"])
    print("Candidates:", manifest["candidate_count"])
    print("Optimizer:", manifest["optimizer_name"])
    print("Covariance:", manifest["covariance_estimator"])
    print("Lookback returns:", manifest["lookback_returns"])
    print("Max weight:", manifest["max_weight"])
    print("Portfolio volatility:", manifest["portfolio_volatility"])
    print("Weights sum:", manifest["weights_sum"])
    print("Scanner snapshot SHA-256:", manifest["scanner_snapshot_sha256"])
    print("Allocation SHA-256:", manifest["candidate_allocation_sha256"])
    print()
    print("Saved ->", allocation_path)
    print("Saved ->", manifest_path)
    print("=" * 88)


if __name__ == "__main__":
    main()
