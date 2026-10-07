# Thinkorswim Portfolio Optimization Engine

A research-first, risk-aware portfolio optimization and forward-testing project for individual investors and learners. The repository combines classical portfolio construction, regularized statistical models, chronological backtesting, deterministic audit trails, and a deliberately read-only Schwab integration boundary.

> **Status:** research / forward-testing infrastructure. This project does not currently submit brokerage orders. Historical, simulated, synthetic, and forward-test results must not be interpreted as live trading performance.

## Why this project exists

The project started as a personal portfolio-research engine, but its architecture is intentionally documented so others can study, reproduce, critique, and extend the work without requiring paid research software.

The design goals are:

- reproducible portfolio research;
- explicit separation of research, forward testing, and execution;
- auditable inputs and outputs through deterministic hashes;
- realistic transaction-cost and whole-share constraints;
- chronological evaluation to reduce look-ahead bias;
- broker integration that is read-only until execution is separately designed and reviewed;
- clear attribution to the researchers, data providers, exchanges, brokers, and open-source projects that make the work possible.

## Current research universe

The frozen Meta Allocation V1 research candidate uses:

- SPY
- QQQ
- TLT
- GLD
- SCHD

The research engine also contains infrastructure for equal weight, minimum variance, risk parity, hierarchical risk parity, consensus allocation, and maximum-Sharpe portfolios.

## Meta Allocation V1

Meta Allocation V1 is a **frozen research candidate** intended for forward testing and paper/simulated validation, not a claim of production readiness.

The frozen gate:

- chooses between Equal Weight and Maximum Sharpe;
- uses 12 market-derived features;
- uses a standardized Ridge model with alpha = 10;
- predicts 3-month Maximum-Sharpe excess return versus Equal Weight;
- uses a zero decision threshold;
- applies a 40% maximum asset weight;
- assumes 5 bps transaction cost per unit of turnover;
- makes quarterly state decisions in January, April, July, and October;
- refreshes target weights monthly while holding the quarterly allocation state.

The historical research artifact is stored under `models/meta_allocation_v1/`. Historical tuning is frozen; future improvements should be treated as a new model version rather than silently changing V1.

## Historical experiment summary

The historical walk-forward evaluation used an expanding training window with strict chronological information flow. The corrected monthly out-of-sample evaluation contained 104 monthly observations through 2026-09-30.

Selected historical results:

| Model | Annualized return | Annualized volatility | Sharpe | Max drawdown |
| --- | ---: | ---: | ---: | ---: |
| Equal Weight | 11.55% | 11.85% | 0.985 | -22.38% |
| Maximum Sharpe | 11.08% | 11.69% | 0.961 | -23.04% |
| HRP | 9.68% | 10.92% | 0.903 | -21.22% |
| Risk Parity | 9.05% | 10.90% | 0.852 | -22.45% |
| Consensus | 8.51% | 10.71% | 0.818 | -21.50% |
| Minimum Variance | 6.78% | 10.59% | 0.674 | -20.84% |

These figures are historical research results, not live performance.

The final binary Meta Allocation V1 research stage produced 63 overlapping 3-month out-of-sample predictions and 21 non-overlapping quarterly decisions on the canonical phase-0 schedule. The signal showed stronger historical evidence versus Equal Weight than versus Maximum Sharpe, and the contribution was concentrated in a small number of periods. That limitation is one reason the model was frozen for genuine forward evaluation rather than further historical tuning.

See [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) for the research chronology, tests, failures, fixes, and interpretation.

## Forward-test architecture

```text
Frozen Meta Allocation V1
        ↓
forward signal
        ↓
target-weight record
        ↓
PortfolioSnapshot
        ↓
whole-share OrderPlanResult
        ↓
append-only audit
        ↓
DRY_RUN only
```

The forward infrastructure now includes:

- exact input-slice hashing rather than whole-source-file hashing;
- XNYS trading-calendar month-end and next-session resolution;
- historical/forward Maximum-Sharpe construction parity tests;
- whole-share, long-only, no-margin order planning;
- transaction-cost accounting invariants;
- deterministic holdings, price, target-weight, and order-plan hashes;
- append-only summary and per-asset audit ledgers;
- broker-independent portfolio snapshots;
- synthetic paperMoney-style snapshot normalization;
- a Schwab read-only client boundary;
- a Schwab read-only snapshot adapter;
- OAuth bootstrap code that immediately wraps the authenticated client in the read-only boundary;
- a synthetic end-to-end test from mocked authentication through audit.

## Schwab integration status

The intended Schwab product is **Trader API - Individual** for personal access to the developer's own self-directed brokerage account.

The repository currently exposes only read operations:

- account-hash discovery;
- balances and positions;
- quotes.

The wrapper deliberately exposes no order-placement, replacement, cancellation, or preview methods.

As of 2026-10-07, Schwab Trader API - Individual access for this project was pending administrator approval. No real Schwab token had been created and no real brokerage order had been submitted.

## Testing

The current local development suite reached:

```text
262 passed
```

before the first real Schwab OAuth connection.

Coverage includes optimizer primitives, walk-forward logic, frozen model parity, schedule semantics, XNYS holidays, forward-weight records, order planning, accounting invariants, provenance hashes, audit idempotency, portfolio snapshots, synthetic paperMoney adapters, Schwab read-only transport, Schwab snapshot normalization, OAuth bootstrap behavior, and a fully synthetic end-to-end Schwab pipeline.

The exact test count is a development checkpoint, not a substitute for code review or forward validation.

## Reproducing the work

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

At minimum, preserve:

- Git commit SHA;
- Python version;
- `requirements-lock.txt`;
- data source and observation cutoff;
- model artifact version;
- input-market-data hash;
- target-weight hash;
- holdings snapshot hash;
- price snapshot hash;
- order-plan hash;
- transaction-cost assumption;
- test-suite status.

## Sources and attribution

The research and implementation build on published academic work, public data sources, broker documentation, and open-source software.

See [docs/SOURCES_AND_ATTRIBUTION.md](docs/SOURCES_AND_ATTRIBUTION.md) for citations covering Markowitz mean-variance selection, Ledoit-Wolf covariance shrinkage, risk parity, HRP, Sharpe ratio, Ridge regression, bootstrap methods, walk-forward/time-series evaluation, SciPy optimization, FRED/ALFRED, yfinance, exchange_calendars, Charles Schwab, and schwab-py.

For public write-ups, see [docs/PUBLISHING_GUIDE.md](docs/PUBLISHING_GUIDE.md) for evidence-labeling, disclosure, attribution, and citation guidance.

Citations acknowledge prior work and help readers reproduce the implementation. They do not imply endorsement by the cited individuals or organizations.

## Security and secrets

Never commit:

- `.env`;
- Schwab App Key or App Secret;
- OAuth tokens;
- account numbers or account hashes;
- private brokerage payloads.

Use `.env.example` only as a placeholder template.

## Disclaimer

This repository is research and educational software, not investment, legal, tax, or brokerage advice. Historical, simulated, synthetic, and forward-test results can differ materially from live trading. Market data can be revised; APIs can change; transaction costs, taxes, slippage, liquidity, tracking error, and execution risk can materially alter realized results.

Use of Schwab, Yahoo Finance, Federal Reserve, exchange-calendar, and other third-party data or services remains subject to the applicable provider terms. The repository is not affiliated with or endorsed by Charles Schwab, Yahoo, the Federal Reserve Bank of St. Louis, NYSE, or the cited academic authors.

## License

See [LICENSE](LICENSE).
