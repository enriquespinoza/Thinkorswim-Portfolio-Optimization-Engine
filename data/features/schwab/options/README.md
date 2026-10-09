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


## Frozen feature-quality policy v1.0.0

Symbol quality:

- PASS: eligible contract ratio >= 0.80, eligible contracts >= 100, and eligible expirations >= 10.
- CAUTION: eligible contract ratio >= 0.50, eligible contracts >= 50, and eligible expirations >= 5.
- SUPPRESS: anything below the CAUTION floor.

Feature-specific support:

- overall matched-expiration skew requires at least 10 matched expiration pairs;
- DTE-bucket matched skew requires at least 5 matched expiration pairs;
- ATM-IV bucket features require at least 20 eligible ATM contracts;
- the 91-365d minus 8-30d ATM-IV term spread requires both component buckets to independently meet the 20-contract ATM support floor.

ATM unique-expiration counts are retained as support diagnostics for every DTE bucket. They are not assigned a separate minimum threshold in quality-policy v1.0.0; the policy records them so a later research version can test whether an additional expiration-count gate is warranted without retroactively changing v1.0.0.

The quality gate preserves every underlying-symbol row and its identity/support fields. For a SUPPRESS symbol, model feature values are written as unavailable. For PASS or CAUTION symbols, individual skew, ATM-IV, or term-structure features are written as unavailable when their own support rule fails.

The quality gate writes:

```text
<run_id>__<feature_schema>__validated_option_features.csv
<run_id>__<feature_schema>__option_feature_quality.csv
<run_id>__<feature_schema>__option_feature_validation_metadata.json
```
