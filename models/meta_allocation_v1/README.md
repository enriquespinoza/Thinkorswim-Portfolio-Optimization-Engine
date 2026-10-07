# Meta Allocation V1

Version: 1.0.0

Status: **FROZEN RESEARCH CANDIDATE**

Frozen: 2026-10-07T00:05:57.641932+00:00

## Purpose

Meta Allocation V1 is a tactical gate between an Equal Weight portfolio
and the Maximum Sharpe optimizer.

The model forecasts Maximum Sharpe's three-month excess return relative
to Equal Weight.

If predicted excess is greater than zero, the allocation state is
Maximum Sharpe.

Otherwise, the allocation state is Equal Weight.

## Frozen Architecture

Market-state features
→ StandardScaler
→ Ridge(alpha=10.0)
→ predicted three-month Maximum-Sharpe excess
→ zero threshold
→ Equal Weight or Maximum Sharpe
→ portfolio constraints
→ execution

## Files

- `meta_allocation_v1.joblib` — frozen fitted sklearn pipeline
- `model_spec.json` — model and training specification
- `feature_list.json` — immutable V1 feature set
- `coefficients.csv` — standardized Ridge coefficients at freeze
- `research_results.csv` — final portfolio-bootstrap validation
- `quarterly_returns.csv` — realized non-overlapping validation blocks
- `validation_summary.md` — final historical research summary
- `artifact_manifest.json` — reproducibility hashes
- `FROZEN.txt` — historical-development freeze notice

## Freeze Policy

Historical tuning of V1 is closed.

Do not change any of the following using pre-freeze history:

- feature list
- Ridge alpha
- three-month target horizon
- zero decision threshold
- Equal Weight / Maximum Sharpe decision states
- maximum portfolio weight
- training methodology

Any modification creates Meta Allocation V2.

V1 should now be evaluated using genuinely unseen forward observations
and paperMoney execution.

This artifact is for research and forward testing, not an assertion of
future investment performance.
