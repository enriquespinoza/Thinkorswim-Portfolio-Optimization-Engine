# Scanner Candidate Handoff Contract v1

## Purpose

This contract defines the immutable boundary between the standalone Stock Scanner Engine and the Portfolio Optimization Engine.

The scanner owns security-selection research, alpha scoring, ranking, and candidate generation. The portfolio optimizer does not recompute scanner alpha or alter scanner ranks. It ingests a versioned candidate snapshot with provenance, enriches those symbols with downstream data such as validated Schwab option features, and later decides portfolio allocation.

## Upstream source

The initial handoff is designed for the scanner's frozen alpha specification:

```text
scanner-alpha-v1
```

The current scanner holdout snapshot contains:

```text
timestamp
symbol
close
momentum_252d_ex_20d
ema_50_to_200
alpha_score
alpha_rank
alpha_percentile
alpha_complete
alpha_spec_version
```

Only rows where `alpha_complete == True` are eligible to become handoff candidates.

## Contract version

```text
scanner-candidate-handoff-v1
```

The consumer must reject incompatible handoff versions rather than silently coercing them.

## Candidate record schema

Each candidate row contains:

```text
handoff_contract_version
scanner_alpha_spec_version
scanner_provider
as_of_session
generated_at_utc
scanner_snapshot_sha256
source_alpha_manifest_sha256
source_timestamp_utc
symbol
scanner_rank
scanner_score
scanner_percentile
scanner_eligible
close
primary_signal_name
primary_signal_value
ema_50_to_200
```

Field meanings:

- `handoff_contract_version`: immutable optimizer/scanner interface version.
- `scanner_alpha_spec_version`: scanner research specification that produced the ranking.
- `scanner_provider`: upstream market-data provider used by the scanner.
- `as_of_session`: common market session represented by the scanner ranking.
- `generated_at_utc`: timestamp when the handoff artifact is created.
- `scanner_snapshot_sha256`: SHA-256 recorded by the scanner for the immutable alpha-ranking snapshot.
- `source_alpha_manifest_sha256`: SHA-256 of the scanner's immutable holdout manifest.
- `source_timestamp_utc`: original timestamp carried by the scanner row.
- `symbol`: normalized uppercase candidate symbol.
- `scanner_rank`: frozen `alpha_rank`.
- `scanner_score`: frozen `alpha_score`.
- `scanner_percentile`: frozen `alpha_percentile`.
- `scanner_eligible`: must be true for v1 handoff rows.
- `close`: scanner snapshot close used for traceability, not execution pricing.
- `primary_signal_name`: `momentum_252d_ex_20d` for Scanner Alpha V1.
- `primary_signal_value`: frozen value of that primary alpha signal.
- `ema_50_to_200`: context-only scanner feature retained for diagnosis.

## Selection policy

Initial policy:

```text
top_ranked_alpha_complete
```

Candidates are ordered by `scanner_rank` ascending. The handoff generator may cap the candidate count, but it may not re-rank, rescore, or change scanner eligibility.

The initial research default should be a top-20 shortlist. A different shortlist size is a handoff-generation parameter, not a change to scanner alpha.

## Manifest

Each candidate CSV must have a companion manifest containing at least:

```text
handoff_contract_version
scanner_alpha_spec_version
scanner_provider
as_of_session
generated_at_utc
scanner_snapshot_sha256
source_alpha_manifest_sha256
candidate_count
candidate_limit
selection_policy
candidate_csv_sha256
immutable
```

The manifest should also preserve the source scanner snapshot and source manifest references when available.

## Ownership boundary

The scanner owns:

```text
feature engineering
alpha specification
alpha score
rank
percentile
scanner completeness / eligibility
```

The optimizer owns:

```text
handoff validation
candidate-history ingestion
Schwab option enrichment
portfolio constraints
allocation
rebalance planning
paper-trading execution handoff
```

The optimizer must not feed downstream option information back into Scanner Alpha V1. Any scanner model that incorporates option-market information requires a new scanner research specification and holdout.

## Downstream join

The candidate snapshot can later be enriched with the option-feature history using symbol and time-safe as-of logic:

```text
scanner candidate
    +
latest validated option snapshot available no later than decision time
    ->
candidate research record
```

A future join must never use an option capture timestamp later than the portfolio decision timestamp.

## Next implementation

The consumer-side ingestion layer is:

```text
scripts/ingest_scanner_candidates.py
```

It should validate this contract, verify hashes and version fields, enforce unique candidate keys, and append immutable scanner candidate history without recomputing scanner scores.
