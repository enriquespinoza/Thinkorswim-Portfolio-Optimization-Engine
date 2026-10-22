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

from config.portfolio_blend_forward_validation import (
    CHECKPOINT_SESSIONS,
    FORWARD_VALIDATION_BUILDER_VERSION,
    FORWARD_VALIDATION_NAME,
    FORWARD_VALIDATION_VERSION,
    FORMAL_MINIMUM_SESSIONS,
    FROZEN_OPTIMIZER_BLEND_WEIGHTS,
    ORDER_SUBMISSION_PERMITTED,
    RETUNING_PERMITTED,
    WEIGHTING_METHOD,
)
from scripts.build_personal_portfolio_benchmark import (
    load_optimizer_weights,
    load_personal_weights,
    sha256_file,
)
from src.research.portfolio_blend_forward_validation import (
    build_forward_blend_table,
    forward_price_returns,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen Portfolio Blend Forward Validation V1 weights "
            "on sessions strictly after the personal portfolio freeze date."
        )
    )
    parser.add_argument("--optimizer-session", required=True)
    parser.add_argument("--personal-freeze-session", required=True)
    parser.add_argument("--evaluation-session", required=True)
    parser.add_argument("--personal-snapshot", type=Path, required=True)
    parser.add_argument("--optimizer-allocation", type=Path, required=True)
    parser.add_argument("--optimizer-manifest", type=Path, required=True)
    parser.add_argument("--prices-csv", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "reports/validation/portfolio_blend_forward_validation"
        ),
    )
    args = parser.parse_args()

    (
        personal_weights,
        portfolio_value,
        pricing_coverage,
    ) = load_personal_weights(args.personal_snapshot)

    optimizer_weights, optimizer_sha = load_optimizer_weights(
        args.optimizer_allocation,
        args.optimizer_manifest,
        args.optimizer_session,
    )

    prices = pd.read_csv(
        args.prices_csv,
        index_col=0,
        low_memory=False,
    )

    symbols = sorted(
        set(personal_weights.index)
        | set(optimizer_weights[optimizer_weights.gt(0)].index)
    )

    returns = forward_price_returns(
        prices,
        symbols=symbols,
        freeze_session=args.personal_freeze_session,
        evaluation_session=args.evaluation_session,
    )

    table, daily = build_forward_blend_table(
        returns=returns,
        personal_weights=personal_weights,
        optimizer_weights=optimizer_weights,
        optimizer_blend_weights=FROZEN_OPTIMIZER_BLEND_WEIGHTS,
    )

    output_dir = (
        args.output_root
        / f"{args.optimizer_session}_optimizer"
        / f"{args.personal_freeze_session}_personal"
        / args.evaluation_session
    )
    summary_path = output_dir / "summary.json"
    table_path = output_dir / "blend_forward_summary.csv"
    daily_path = output_dir / "blend_forward_daily_returns.csv"

    if any(path.exists() for path in (summary_path, table_path, daily_path)):
        raise FileExistsError(
            "forward validation output already exists; refusing to overwrite"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    table.to_csv(table_path, index=False, lineterminator="\n")
    daily.to_csv(daily_path, lineterminator="\n")

    observed_sessions = int(len(returns))
    status = (
        "FORMAL_FORWARD_EVIDENCE"
        if observed_sessions >= FORMAL_MINIMUM_SESSIONS
        else "PROVISIONAL_FORWARD_OBSERVATION"
    )

    summary = {
        "analysis_name": FORWARD_VALIDATION_NAME,
        "analysis_version": FORWARD_VALIDATION_VERSION,
        "builder_version": FORWARD_VALIDATION_BUILDER_VERSION,
        "optimizer_session": args.optimizer_session,
        "personal_freeze_session": args.personal_freeze_session,
        "evaluation_session": args.evaluation_session,
        "first_eligible_forward_session": (
            pd.Timestamp(args.personal_freeze_session)
            + pd.offsets.BDay(1)
        ).date().isoformat(),
        "observed_sessions": observed_sessions,
        "formal_minimum_sessions": FORMAL_MINIMUM_SESSIONS,
        "checkpoint_sessions": list(CHECKPOINT_SESSIONS),
        "status": status,
        "frozen_optimizer_blend_weights": list(
            FROZEN_OPTIMIZER_BLEND_WEIGHTS
        ),
        "weighting_method": WEIGHTING_METHOD,
        "portfolio_value_at_capture": portfolio_value,
        "personal_pricing_coverage": pricing_coverage,
        "personal_unpriced_weight": 1.0 - pricing_coverage,
        "retuning_permitted": RETUNING_PERMITTED,
        "order_submission_permitted": ORDER_SUBMISSION_PERMITTED,
        "personal_snapshot_sha256": sha256_file(args.personal_snapshot),
        "optimizer_allocation_sha256": optimizer_sha,
        "optimizer_manifest_sha256": sha256_file(
            args.optimizer_manifest
        ),
        "price_data_sha256": sha256_file(args.prices_csv),
        "forward_summary_sha256": sha256_file(table_path),
        "forward_daily_returns_sha256": sha256_file(daily_path),
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=" * 88)
    print(FORWARD_VALIDATION_NAME.upper())
    print("=" * 88)
    print("Optimizer session:", args.optimizer_session)
    print("Personal freeze session:", args.personal_freeze_session)
    print("Evaluation session:", args.evaluation_session)
    print("Observed forward sessions:", observed_sessions)
    print("Status:", status)
    print("Frozen optimizer sleeves:", FROZEN_OPTIMIZER_BLEND_WEIGHTS)
    print()
    print(table.to_string(index=False))
    print()
    print("Saved ->", table_path)
    print("Saved ->", daily_path)
    print("Saved ->", summary_path)
    print("Retuning permitted:", RETUNING_PERMITTED)
    print("Order submission permitted:", ORDER_SUBMISSION_PERMITTED)
    print("=" * 88)


if __name__ == "__main__":
    main()
