# Reproducibility Guide

This guide describes how to reproduce the portfolio research and forward-testing infrastructure without relying on undocumented local state.

## 1. Record the exact code state

Always record the Git commit SHA used for an experiment:

```bash
git rev-parse HEAD
```

For frozen research artifacts, also record the model version and artifact manifest.

## 2. Recreate the Python environment

The project was developed with Python 3.11.

Create a fresh environment and install the recorded dependencies:

```bash
python -m venv .venv

# Windows Git Bash
source .venv/Scripts/activate

pip install -r requirements-lock.txt
```

If the lock file has been regenerated on a different platform, record that fact.

## 3. Keep credentials and brokerage data private

Never commit:

- `.env`;
- Schwab App Key;
- Schwab App Secret;
- OAuth access/refresh tokens;
- Schwab account numbers;
- account hashes;
- raw private brokerage payloads.

Use `.env.example` as a template only.

## 4. Record market-data provenance

For each research or forward decision, preserve:

- data source;
- retrieval date;
- observation cutoff;
- adjusted/unadjusted status;
- symbols;
- exact return-slice hash;
- source-data end date.

Forward Meta Allocation V1 records use the hash of the exact input return slice rather than the entire source CSV. This makes later upstream revisions detectable without making unrelated future rows part of an earlier decision identity.

## 5. Distinguish research stages

Every published result should be labeled as one of:

### Exploratory in-sample

Used for idea development or diagnostic research. Not evidence of forward performance.

### Historical out-of-sample

Chronological walk-forward or held-out historical evaluation. Still subject to research-selection bias, revised data, and repeated experimentation.

### Synthetic forward infrastructure

Uses synthetic/mock broker state to validate implementation and interfaces. No market-performance evidence.

### Genuine forward observation

A decision produced after the model was frozen using only information available at that time.

### Paper/simulated execution

A simulated account result. Not live-money performance.

### Live execution

Requires a separately reviewed execution layer. No live execution results are part of Meta Allocation V1 at the current project stage.

## 6. Historical walk-forward protocol

The frozen historical framework used:

- expanding training windows;
- minimum training history of 756 observations;
- monthly evaluation;
- strict information-at-t to next-period usage;
- 5 bps transaction cost per unit turnover;
- 40% maximum asset weight where applicable;
- risk-free rate = 0 for the frozen Maximum-Sharpe implementation;
- Ledoit-Wolf covariance;
- robust expected returns with EWMA span 126.

Readers should verify the exact implementation in the cited commit rather than relying only on this prose description.

## 7. Meta Allocation V1 protocol

Frozen parameters:

```text
universe                SPY, QQQ, TLT, GLD, SCHD
model                   StandardScaler + Ridge
ridge alpha             10
feature count           12
target horizon          3 months
target                  MS excess return vs Equal Weight
decision threshold      0
positive prediction     Maximum Sharpe
non-positive prediction Equal Weight
decision months         Jan / Apr / Jul / Oct
state holding period    3 months
weight refresh          monthly
max asset weight        40%
transaction costs       5 bps / unit turnover
```

Do not change these parameters and still call the result Meta Allocation V1.

## 8. Forward schedule semantics

Each target record distinguishes:

- `information_date`: final relevant monthly trading-session date used in the calculation;
- `decision_signal_date`: quarterly signal controlling the state;
- `effective_from`: earliest eligible trading session after the information date;
- `state_age_months`: months elapsed since the controlling quarterly decision.

XNYS exchange-session logic is used rather than generic weekdays.

Example:

```text
information date  2026-12-31
effective from    2027-01-04
```

January 1 is a weekday in some generic calendars but is not an eligible NYSE session.

## 9. Portfolio-snapshot protocol

For each synthetic or real read-only snapshot, preserve:

- snapshot timestamp;
- price-as-of timestamp;
- account mode;
- current cash;
- shares for each frozen asset;
- reference prices;
- holdings snapshot hash;
- price snapshot hash;
- portfolio snapshot hash.

The snapshot boundary is broker-independent.

## 10. Order-plan protocol

Order plans are currently DRY_RUN only.

Record:

- target-weights hash;
- holdings snapshot hash;
- price snapshot hash;
- transaction-cost assumption;
- order-plan hash;
- current shares;
- target shares;
- trade shares;
- side;
- trade notional;
- residual cash;
- turnover;
- estimated transaction cost.

The accounting invariant is:

```text
ending portfolio value
=
beginning portfolio value
-
estimated transaction costs
```

No margin, shorting, or fractional shares are allowed in V1.

## 11. Audit protocol

The order-plan audit layer uses:

- one summary row per `order_plan_hash`;
- one detail row per asset per `order_plan_hash`;
- idempotent appends;
- conflict rejection.

Re-running an identical plan should not create duplicate records.

## 12. Test protocol

Run the full suite before publishing a result:

```bash
pytest -q
```

The last recorded development checkpoint before real Schwab authorization was:

```text
262 passed
```

A test count is only meaningful together with the commit SHA that produced it.

## 13. Schwab read-only reproduction

The project uses Charles Schwab Trader API - Individual and `schwab-py` for OAuth/bootstrap.

Current architecture:

```text
OAuth
  ↓
raw schwab-py client
  ↓
SchwabReadOnlyClient
  ↓
account / position / quote reads only
  ↓
SchwabReadOnlyPayloads
  ↓
SCHWAB_READ_ONLY PortfolioSnapshot
```

The application boundary intentionally does not expose order-placement or cancellation methods.

For public reproduction, each user must create and authorize their own eligible Schwab application and comply with Schwab's current developer terms. Credentials are not portable between users.

## 14. Data-vintage warning

Historical adjusted prices may be revised by the upstream data provider.

FRED current histories may also reflect revisions. If a macro strategy depends on what was known at a historical point in time, use vintage-aware data such as ALFRED where appropriate.

This is one reason exploratory macro features were excluded from frozen Meta Allocation V1.

## 15. Publication checklist

Before publishing a chart, table, model comparison, or performance claim, include:

- commit SHA;
- model version;
- sample start/end;
- universe;
- rebalance cadence;
- constraints;
- transaction costs;
- data source;
- whether data are revised or vintage-aware;
- number of observations;
- number of non-overlapping decisions where applicable;
- whether the result is exploratory, historical OOS, synthetic, forward, paper, or live;
- relevant citations from `SOURCES_AND_ATTRIBUTION.md`;
- limitations.

## 16. Independent replication

A useful replication should be able to disagree with the original result.

Readers are encouraged to:

- rerun with a fresh environment;
- independently obtain the data;
- verify hashes and dates;
- inspect chronological splits;
- test alternative reasonable transaction-cost assumptions;
- compare against simple benchmarks;
- report failed replications as well as successful ones.

The purpose of this repository is not to present one optimized result as certain. It is to make the research process inspectable.
