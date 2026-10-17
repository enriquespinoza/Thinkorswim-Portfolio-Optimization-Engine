from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import GENERATED_REPORTS_DIR
from src.allocations.execution_targets import write_execution_safe_targets


DEFAULT_OUTPUT_ROOT = GENERATED_REPORTS_DIR / "execution_targets"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a frozen scanner candidate allocation into an immutable "
            "execution-safe target artifact. This command never submits orders."
        )
    )
    parser.add_argument("--allocation-csv", type=Path, required=True)
    parser.add_argument("--allocation-manifest", type=Path, required=True)
    parser.add_argument(
        "--positions-csv",
        type=Path,
        required=True,
        help=(
            "Managed-sleeve position snapshot with columns: "
            "symbol,current_position,reference_price"
        ),
    )
    parser.add_argument("--portfolio-capital", type=float, required=True)
    parser.add_argument(
        "--execution-state",
        choices=("DRY_RUN", "PAPER"),
        default="DRY_RUN",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    target_path, manifest_path, manifest = write_execution_safe_targets(
        allocation_path=args.allocation_csv,
        allocation_manifest_path=args.allocation_manifest,
        positions_path=args.positions_csv,
        portfolio_capital=args.portfolio_capital,
        execution_state=args.execution_state,
        output_root=args.output_root,
    )

    print()
    print("=" * 88)
    print("OPTIMIZER -> EXECUTION SAFE TARGET V1")
    print("=" * 88)
    print("Session:", manifest["as_of_session"])
    print("Handoff contract:", manifest["handoff_contract_version"])
    print("Execution state:", manifest["execution_state"])
    print("Portfolio capital:", manifest["portfolio_capital"])
    print("Rows:", manifest["row_count"])
    print("Target symbols:", manifest["target_symbol_count"])
    print("Buys:", manifest["buy_count"])
    print("Sells:", manifest["sell_count"])
    print("Holds:", manifest["hold_count"])
    print("Target notional:", manifest["target_notional"])
    print("Residual cash:", manifest["residual_cash"])
    print("Broker submission permitted:", manifest["broker_submission_permitted"])
    print("Allocation SHA-256:", manifest["allocation_sha256"])
    print("Execution target SHA-256:", manifest["execution_safe_targets_sha256"])
    print()
    print("Saved ->", target_path)
    print("Saved ->", manifest_path)
    print("=" * 88)


if __name__ == "__main__":
    main()
