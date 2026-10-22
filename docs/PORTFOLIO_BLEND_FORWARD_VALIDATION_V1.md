# Portfolio Blend Forward Validation V1

## Purpose

Portfolio Blend Forward Validation V1 evaluates frozen personal/optimizer
allocation blends prospectively after the personal portfolio freeze date.

No blend weight is selected or retuned using forward observations.

## Frozen candidates

The only permitted optimizer sleeve weights are:

- 0% optimizer / 100% personal
- 20% optimizer / 80% personal
- 30% optimizer / 70% personal
- 100% optimizer / 0% personal

The 20% and 30% sleeves were frozen after Portfolio Blend Analysis V1 because
they materially improved historical risk characteristics without requiring a
full portfolio replacement.

## Point-in-time boundary

Two dates are deliberately separated:

- optimizer_session: the frozen optimizer allocation date;
- personal_freeze_session: the date the personal portfolio snapshot was
  actually captured and frozen.

For the initial experiment:

- optimizer_session = 2026-10-16
- personal_freeze_session = 2026-10-22

No observation on or before 2026-10-22 is counted as forward evidence.

## Weighting method

The validator uses fixed constant-mix daily weights for research measurement.
It does not simulate brokerage fills, taxes, spreads, commissions, or execution
slippage.

Those effects belong in a later execution/paper-money experiment.

## Evidence status

Fewer than 20 observed forward sessions is labeled:

PROVISIONAL_FORWARD_OBSERVATION

At 20 or more sessions it becomes:

FORMAL_FORWARD_EVIDENCE

Planned checkpoints are 20, 60, 120, and 252 sessions.

## Metrics

Each frozen blend reports:

- cumulative return;
- realized annualized volatility;
- maximum drawdown;
- best day;
- worst day;
- positive-session rate;
- cumulative excess return versus the frozen personal portfolio.

Return-based metrics are now genuinely forward relative to the freeze date,
assuming the supplied price data itself contains only legitimate market
observations.

## Command

The forward price matrix must include prices after the personal freeze date.

Example:

```bash
./.venv/Scripts/python.exe \
  scripts/build_portfolio_blend_forward_validation.py \
  --optimizer-session 2026-10-16 \
  --personal-freeze-session 2026-10-22 \
  --evaluation-session 2026-11-20 \
  --personal-snapshot data/private/personal_portfolio/portfolio_all_accounts_2026-10-16.csv \
  --optimizer-allocation reports/validation/scanner_allocations/2026-10-16/candidate_allocation.csv \
  --optimizer-manifest reports/validation/scanner_allocations/2026-10-16/manifest.json \
  --prices-csv <forward_price_matrix.csv>
```

The personal snapshot filename may retain the original research label, but the
validator records the actual freeze date separately. Future snapshots should be
named using their actual capture date to avoid ambiguity.

## Boundary

This experiment is read-only research.

It does not create execution targets, submit orders, or authorize live trading.
A favorable forward result must still pass paper-money and execution-engine
validation before any implementation decision.
