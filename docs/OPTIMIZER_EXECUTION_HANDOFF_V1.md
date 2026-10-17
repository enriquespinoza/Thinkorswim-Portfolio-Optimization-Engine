# Optimizer to Execution Handoff v1

## Purpose

`portfolio-target-handoff-v1` converts an immutable
`scanner-candidate-allocation-v1` artifact into execution-safe whole-share
portfolio targets. It does not submit orders.

## Inputs

- frozen candidate allocation CSV;
- matching candidate allocation manifest;
- managed-sleeve positions CSV with:
  - `symbol`
  - `current_position`
  - `reference_price`
- portfolio capital;
- execution state: `DRY_RUN` or `PAPER`.

The position snapshot must represent only the sleeve managed by this allocation.
This prevents unrelated account holdings from being liquidated when they are absent
from the scanner candidate allocation.

## Output columns

```text
handoff_contract_version
target_builder_version
as_of_session
allocation_contract_version
allocation_sha256
allocation_manifest_sha256
scanner_snapshot_sha256
symbol
scanner_rank
approved_target_weight
reference_price
reference_price_source
portfolio_capital
target_dollar_value
target_shares
current_position
share_delta
trade_side
execution_state
```

Symbols held in the managed sleeve but absent from the new allocation are retained
with a zero approved target weight, allowing the execution engine to see the
required exit rather than silently dropping the position.

Target shares use the frozen policy:

```text
floor(portfolio_capital * approved_target_weight / reference_price)
```

No fractional shares, margin, or short targets are introduced by this contract.

## Safety boundary

The generated manifest always records:

```text
broker_submission_permitted = false
```

The artifact is an input to the Trading Execution Engine, where order construction,
risk gating, approval, and SimulationBroker/PAPER behavior remain downstream
responsibilities.

`LIVE` is not a valid execution state in v1.

## CLI

```bash
./.venv/Scripts/python.exe scripts/build_execution_safe_targets.py \
  --allocation-csv reports/validation/scanner_allocations/2026-10-16/candidate_allocation.csv \
  --allocation-manifest reports/validation/scanner_allocations/2026-10-16/manifest.json \
  --positions-csv <managed_sleeve_positions.csv> \
  --portfolio-capital <capital> \
  --execution-state DRY_RUN
```

The writer is immutable by session and refuses to overwrite an existing target
artifact.
