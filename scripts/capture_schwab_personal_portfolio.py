from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integrations.schwab.auth_bootstrap import (
    create_read_only_schwab_client,
    load_schwab_auth_profile,
)
from src.research.schwab_personal_portfolio import (
    capture_account_by_index,
)


DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "private"
    / "personal_portfolio"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Capture one Schwab account's positions through the repository's "
            "read-only client and save a private normalized research snapshot."
        )
    )
    parser.add_argument(
        "--account-index",
        type=int,
        required=True,
        help=(
            "1-based ordinal from the locally discovered Schwab accounts. "
            "No account number or hash is written to output."
        ),
    )
    parser.add_argument(
        "--as-of-session",
        required=True,
        help="Research session label, for example 2026-10-16.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    args = parser.parse_args()

    config = load_schwab_auth_profile("accounts")
    client = create_read_only_schwab_client(config)

    capture = capture_account_by_index(
        client,
        account_index=args.account_index,
    )

    output_dir = Path(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = output_dir / f"portfolio_{args.as_of_session}.csv"
    manifest_path = output_dir / f"portfolio_{args.as_of_session}.manifest.json"

    if snapshot_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "private portfolio snapshot already exists for "
            f"{args.as_of_session}; refusing to overwrite"
        )

    capture.positions.to_csv(
        snapshot_path,
        index=False,
        lineterminator="\n",
    )

    manifest = {
        "schema_version": "personal-portfolio-snapshot-v1",
        "as_of_session": args.as_of_session,
        "account_index": int(capture.account_index),
        "fetched_at_utc": capture.fetched_at_utc.isoformat(),
        "position_count": int(len(capture.positions)),
        "portfolio_market_value": float(
            capture.positions["market_value"].sum()
            if not capture.positions.empty
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
    print("SCHWAB PERSONAL PORTFOLIO SNAPSHOT")
    print("=" * 88)
    print("Account index:", manifest["account_index"])
    print("Session:", manifest["as_of_session"])
    print("Positions:", manifest["position_count"])
    print("Portfolio market value:", manifest["portfolio_market_value"])
    print("Read-only research:", manifest["read_only_research"])
    print("Account number stored:", manifest["contains_account_number"])
    print("Account hash stored:", manifest["contains_account_hash"])
    print("Saved ->", snapshot_path)
    print("Saved ->", manifest_path)
    print("=" * 88)


if __name__ == "__main__":
    main()
