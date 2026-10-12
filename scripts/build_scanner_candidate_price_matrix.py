from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config.settings import DATA_DIR, PROCESSED_DATA_DIR
from src.data.scanner_candidate_prices import (
    write_price_matrix_from_capture_summary,
)


DEFAULT_HISTORY_PATH = (
    DATA_DIR / "history" / "scanner" / "scanner_candidate_history_v1.csv"
)
DEFAULT_PROCESSED_ROOT = PROCESSED_DATA_DIR / "schwab"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an immutable wide candidate price matrix from one "
            "Schwab capture-run summary."
        )
    )
    parser.add_argument("--capture-summary", type=Path, required=True)
    parser.add_argument("--as-of-session", required=True)
    parser.add_argument(
        "--candidate-history",
        type=Path,
        default=DEFAULT_HISTORY_PATH,
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    history = pd.read_csv(args.candidate_history, low_memory=False)
    selected = history.loc[
        history["as_of_session"].astype(str).eq(args.as_of_session)
    ].copy()
    if selected.empty:
        raise ValueError(
            f"No scanner candidates found for session {args.as_of_session}"
        )
    selected = selected.sort_values(["scanner_rank", "symbol"], kind="stable")
    symbols = selected["symbol"].astype(str).str.upper().tolist()

    output_path, manifest_path = write_price_matrix_from_capture_summary(
        summary_path=args.capture_summary,
        processed_root=args.processed_root,
        output_path=args.output,
        expected_symbols=symbols,
    )

    print()
    print("=" * 88)
    print("SCANNER CANDIDATE SCHWAB PRICE MATRIX")
    print("=" * 88)
    print("Session:", args.as_of_session)
    print("Symbols:", len(symbols))
    print("Saved ->", output_path)
    print("Saved ->", manifest_path)
    print("=" * 88)


if __name__ == "__main__":
    main()
