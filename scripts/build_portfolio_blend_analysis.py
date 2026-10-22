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

from config.portfolio_blend_analysis import (
    BLEND_ANALYSIS_NAME,
    BLEND_ANALYSIS_VERSION,
    BLEND_BUILDER_VERSION,
    BLEND_SCOPE,
    DEFAULT_OPTIMIZER_BLEND_WEIGHTS,
    DEFAULT_ROLLING_WINDOWS,
    RISK_FREE_RATE,
)
from scripts.build_personal_portfolio_benchmark import (
    load_optimizer_weights,
    load_personal_weights,
    load_returns,
    sha256_file,
)
from src.research.portfolio_blend_analysis import build_blend_table


def parse_blends(value: str) -> tuple[float, ...]:
    values = tuple(
        float(item.strip())
        for item in value.split(",")
        if item.strip()
    )
    if not values:
        raise argparse.ArgumentTypeError("at least one blend is required")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build Portfolio Blend Analysis V1 for fixed personal/optimizer "
            "allocation mixtures. Research only; no order generation."
        )
    )
    parser.add_argument("--as-of-session", required=True)
    parser.add_argument("--personal-snapshot", type=Path, required=True)
    parser.add_argument("--optimizer-allocation", type=Path, required=True)
    parser.add_argument("--optimizer-manifest", type=Path, required=True)
    parser.add_argument("--prices-csv", type=Path, required=True)
    parser.add_argument(
        "--optimizer-blends",
        type=parse_blends,
        default=DEFAULT_OPTIMIZER_BLEND_WEIGHTS,
        help="Comma-separated optimizer sleeve weights, e.g. 0,0.1,0.2,0.3,0.5,1",
    )
    parser.add_argument(
        "--lookback-returns",
        type=int,
        default=252,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("reports/validation/portfolio_blend_analysis"),
    )
    args = parser.parse_args()

    (
        personal_weights,
        portfolio_value,
        personal_pricing_coverage,
    ) = load_personal_weights(args.personal_snapshot)

    optimizer_weights, allocation_sha = load_optimizer_weights(
        args.optimizer_allocation,
        args.optimizer_manifest,
        args.as_of_session,
    )

    symbols = sorted(
        set(personal_weights.index)
        | set(optimizer_weights[optimizer_weights.gt(0)].index)
    )

    returns = load_returns(
        args.prices_csv,
        symbols,
        args.as_of_session,
        args.lookback_returns,
    )

    table = build_blend_table(
        returns=returns,
        personal_weights=personal_weights,
        optimizer_weights=optimizer_weights,
        optimizer_blend_weights=args.optimizer_blends,
        rolling_windows=DEFAULT_ROLLING_WINDOWS,
        risk_free_rate=RISK_FREE_RATE,
    )

    output_dir = args.output_root / args.as_of_session
    csv_path = output_dir / "blend_analysis.csv"
    summary_path = output_dir / "summary.json"

    if csv_path.exists() or summary_path.exists():
        raise FileExistsError(
            f"blend analysis already exists for {args.as_of_session}; "
            "refusing to overwrite"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    table.to_csv(csv_path, index=False, lineterminator="\n")

    best_drawdown = table.loc[
        table["max_drawdown"].idxmax()
    ]
    lowest_volatility = table.loc[
        table["annualized_volatility"].idxmin()
    ]
    lowest_concentration = table.loc[
        table["concentration"].idxmin()
    ]

    summary = {
        "analysis_name": BLEND_ANALYSIS_NAME,
        "analysis_version": BLEND_ANALYSIS_VERSION,
        "builder_version": BLEND_BUILDER_VERSION,
        "as_of_session": args.as_of_session,
        "blend_scope": BLEND_SCOPE,
        "personal_portfolio_value": portfolio_value,
        "personal_pricing_coverage": personal_pricing_coverage,
        "personal_unpriced_weight": 1.0 - personal_pricing_coverage,
        "optimizer_blend_weights": [
            float(value) for value in table["optimizer_blend_weight"]
        ],
        "best_drawdown_optimizer_weight": float(
            best_drawdown["optimizer_blend_weight"]
        ),
        "best_drawdown": float(best_drawdown["max_drawdown"]),
        "lowest_volatility_optimizer_weight": float(
            lowest_volatility["optimizer_blend_weight"]
        ),
        "lowest_volatility": float(
            lowest_volatility["annualized_volatility"]
        ),
        "lowest_concentration_optimizer_weight": float(
            lowest_concentration["optimizer_blend_weight"]
        ),
        "lowest_concentration": float(
            lowest_concentration["concentration"]
        ),
        "portfolio_snapshot_sha256": sha256_file(args.personal_snapshot),
        "system_allocation_sha256": allocation_sha,
        "system_allocation_manifest_sha256": sha256_file(
            args.optimizer_manifest
        ),
        "price_data_sha256": sha256_file(args.prices_csv),
        "blend_table_sha256": sha256_file(csv_path),
        "lookback_returns": args.lookback_returns,
        "risk_free_rate": RISK_FREE_RATE,
        "read_only_research": True,
        "order_submission_permitted": False,
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=" * 88)
    print(BLEND_ANALYSIS_NAME.upper())
    print("=" * 88)
    print("Session:", args.as_of_session)
    print("Portfolio value:", portfolio_value)
    print("Pricing coverage:", personal_pricing_coverage)
    print("Blends tested:", len(table))
    print()
    display_columns = [
        "optimizer_blend_weight",
        "personal_blend_weight",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "sharpe_ratio_zero_rf",
        "sortino_ratio_zero_rf",
        "concentration",
        "turnover_from_personal",
        "worst_rolling_20d_return",
        "worst_rolling_60d_return",
    ]
    print(table[display_columns].to_string(index=False))
    print()
    print("Saved ->", csv_path)
    print("Saved ->", summary_path)
    print("Read-only research:", True)
    print("Order submission permitted:", False)
    print("=" * 88)


if __name__ == "__main__":
    main()
