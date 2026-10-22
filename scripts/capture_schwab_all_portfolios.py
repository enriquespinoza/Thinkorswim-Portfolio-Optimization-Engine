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

from src.integrations.schwab.auth_bootstrap import (
    create_read_only_schwab_client,
    load_schwab_auth_profile,
)
from src.research.schwab_personal_portfolio import normalize_account_positions


DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "data" / "private" / "personal_portfolio"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of-session", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    args = parser.parse_args()

    client = create_read_only_schwab_client(
        load_schwab_auth_profile("accounts")
    )

    records = client.get_account_number_records()
    frames = []

    for record in records:
        locator = str(record["hashValue"]).strip()
        payload = client.get_account_payload(locator)
        frame = normalize_account_positions(payload)
        if not frame.empty:
            frames.append(frame)

    columns = ["symbol", "quantity", "market_value", "asset_type"]
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined = (
            combined.groupby(
                ["symbol", "asset_type"],
                as_index=False,
                dropna=False,
            )
            .agg(
                quantity=("quantity", "sum"),
                market_value=("market_value", "sum"),
            )
            .loc[:, columns]
            .sort_values(["symbol", "asset_type"], kind="stable")
            .reset_index(drop=True)
        )
    else:
        combined = pd.DataFrame(columns=columns)

    output_dir = Path(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = (
        output_dir
        / f"portfolio_all_accounts_{args.as_of_session}.csv"
    )
    manifest_path = (
        output_dir
        / f"portfolio_all_accounts_{args.as_of_session}.manifest.json"
    )

    if snapshot_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "all-account portfolio snapshot already exists for "
            f"{args.as_of_session}; refusing to overwrite"
        )

    combined.to_csv(snapshot_path, index=False, lineterminator="\n")

    manifest = {
        "schema_version": "personal-portfolio-snapshot-v1",
        "scope": "all_accessible_accounts",
        "as_of_session": args.as_of_session,
        "account_count": len(records),
        "position_count": len(combined),
        "portfolio_market_value": float(
            combined["market_value"].sum()
            if not combined.empty
            else 0.0
        ),
        "snapshot_sha256": sha256_file(snapshot_path),
        "contains_account_number": False,
        "contains_account_hash": False,
        "contains_credentials": False,
        "read_only_research": True,
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=" * 88)
    print("SCHWAB ALL-ACCOUNT PERSONAL PORTFOLIO SNAPSHOT")
    print("=" * 88)
    print("Accounts:", manifest["account_count"])
    print("Session:", manifest["as_of_session"])
    print("Positions:", manifest["position_count"])
    print("Portfolio market value:", manifest["portfolio_market_value"])
    print("Read-only research:", manifest["read_only_research"])
    print("Saved ->", snapshot_path)
    print("Saved ->", manifest_path)
    print("=" * 88)


if __name__ == "__main__":
    main()
