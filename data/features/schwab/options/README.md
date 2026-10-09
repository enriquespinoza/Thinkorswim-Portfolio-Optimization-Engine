# Schwab Option Features

This directory is the project output boundary for derived option-feature tables built from validated Schwab Market Data captures.

## Intended flow

```text
data/raw/schwab/
    -> normalization
data/processed/schwab/
    -> structural audit
    -> schema/profile review
    -> feature-input validation
    -> src/features/schwab_option_features.py
data/features/schwab/options/
```

Feature artifacts in this directory must be derived from a specific immutable Schwab capture-run summary rather than from an unrestricted scan of all processed CSV files.

## Initial V2 policy

Contract-level eligibility follows the shared Schwab option-feature policy:

- positive bid
- ask greater than or equal to bid
- mark inside the bid/ask interval
- relative bid/ask spread no greater than 25%
- quote age no greater than 30 minutes
- DTE greater than zero for the standard feature set
- positive Schwab volatility
- delta within [-1, 1]
- non-negative gamma
- non-negative vega
- valid underlying spot price

The initial feature layer uses canonical project-calculated intrinsic value, extrinsic value, and moneyness rather than the provider's corresponding value-component fields.

The following provider fields are intentionally excluded from initial V2 aggregate modeling features:

- `percentChange`
- `markPercentChange`
- `intrinsicValue`
- `extrinsicValue`
- `theoreticalVolatility`

## Recommended artifact naming

Generated tables should include the source capture run ID in their filename, for example:

```text
<run_id>__option_features.csv
<run_id>__option_eligibility.csv
<run_id>__option_feature_metadata.json
```

Metadata should preserve the source run ID, feature schema version, feature configuration, source artifact IDs, row counts, and hashes needed for reproducibility.
