from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integrations.schwab.auth_bootstrap import (
    create_market_data_schwab_client,
    load_schwab_auth_config,
)


CAPTURE_CONTRACT = "schwab-market-data-capture-v1"


def _symbols_from_targets(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"execution target CSV not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "symbol" not in reader.fieldnames:
            raise ValueError(
                "execution target CSV must contain a symbol column"
            )
        symbols = [
            str(row["symbol"]).strip().upper()
            for row in reader
            if str(row.get("symbol", "")).strip()
        ]

    if not symbols:
        raise ValueError("execution target CSV contains no symbols")
    if len(symbols) != len(set(symbols)):
        raise ValueError(
            "execution target CSV contains duplicate symbols"
        )
    return sorted(symbols)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Capture current Schwab Market Data quotes for an execution "
            "target universe. No Schwab account or trading endpoint is used."
        )
    )
    parser.add_argument("--targets", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    target_path = Path(args.targets)
    output_path = Path(args.output)
    symbols = _symbols_from_targets(target_path)

    config = load_schwab_auth_config()
    client = create_market_data_schwab_client(config)

    captured_at = datetime.now(timezone.utc)
    quotes = client.get_quotes(symbols)

    payload = {
        "capture_contract": CAPTURE_CONTRACT,
        "captured_at_utc": captured_at.isoformat(),
        "symbols": symbols,
        "quotes": quotes,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("SCHWAB MARKET-DATA CAPTURE")
    print("=" * 72)
    print("Symbols requested:", len(symbols))
    print("Quotes returned:", len(quotes))
    print("Output:", output_path)
    print("Schwab account API: NOT USED")
    print("Schwab trading API: NOT USED")


if __name__ == "__main__":
    main()
