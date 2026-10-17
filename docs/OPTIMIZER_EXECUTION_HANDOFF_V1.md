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
target_share_notional
realized_target_weight
weight_rounding_error
target_status
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

Builder version `1.1.0` adds execution accounting without changing the
`portfolio-target-handoff-v1` contract. The additional fields distinguish the
optimizer's approved weight from what can actually be represented in whole shares:

- `target_share_notional = target_shares * reference_price`
- `realized_target_weight = target_share_notional / portfolio_capital`
- `weight_rounding_error = approved_target_weight - realized_target_weight`

`target_status` is one of:

```text
TARGETED
ZERO_WEIGHT
ROUNDING_TO_ZERO
EXIT
```

`ROUNDING_TO_ZERO` means the optimizer approved a positive target weight but the
allocated dollar amount is smaller than the price of one whole share. It remains
distinct from a true optimizer zero weight. `EXIT` is used when the managed
sleeve currently holds shares but the approved target weight is zero.

The manifest records status counts, realized weight sum, total weight-rounding
error, target notional, and residual cash.

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

The writer is immutable. Builder outputs are versioned beneath the session so a
new accounting implementation can coexist with a previously generated artifact:

```text
reports/generated/execution_targets/<session>/<target_builder_version>/
```

The writer refuses to overwrite an existing artifact for the same session and
builder version.
