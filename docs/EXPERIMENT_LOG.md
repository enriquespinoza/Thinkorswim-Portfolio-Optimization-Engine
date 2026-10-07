# Experiment Log

This document records the major research experiments, implementation changes, test checkpoints, and interpretation decisions for the portfolio optimization project. The purpose is to preserve research context so later readers can distinguish exploratory work from frozen methodology and genuine forward evidence.

## Research baseline

The initial portfolio engine compared several allocation approaches on a five-ETF universe: SPY, QQQ, TLT, GLD, and SCHD.

Core methods included:

- Equal Weight;
- Minimum Variance;
- Risk Parity;
- Hierarchical Risk Parity;
- Consensus allocation;
- Maximum Sharpe.

The research pipeline used daily market data, expanding-window walk-forward evaluation, a 40% maximum position constraint where applicable, and 5 bps transaction cost per unit of turnover.

The corrected walk-forward evaluation contained 104 monthly out-of-sample observations through 2026-09-30.

Historical summary:

| Model | Total return | Annualized return | Annualized vol. | Sharpe | Sortino | Max DD | Turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Equal Weight | 157.92% | 11.55% | 11.85% | 0.985 | 1.656 | -22.38% | 2.24 |
| Maximum Sharpe | 148.58% | 11.08% | 11.69% | 0.961 | 1.572 | -23.04% | 17.73 |
| HRP | 122.71% | 9.68% | 10.92% | 0.903 | 1.509 | -21.22% | 2.25 |
| Risk Parity | 111.92% | 9.05% | 10.90% | 0.852 | 1.415 | -22.45% | 2.20 |
| Consensus | 102.94% | 8.51% | 10.71% | 0.818 | 1.353 | -21.50% | 2.24 |
| Minimum Variance | 76.62% | 6.78% | 10.59% | 0.674 | 1.099 | -20.84% | 2.44 |

Interpretation: Equal Weight was a strong benchmark. Maximum Sharpe was competitive but incurred much higher turnover. The more complex allocation methods did not establish standalone historical alpha relative to the simple benchmark.

## Expected-return and covariance construction

Maximum Sharpe used:

- Ledoit-Wolf shrinkage covariance;
- robust expected returns combining strategic and EWMA components;
- SLSQP constrained optimization;
- long-only weights;
- 40% maximum asset weight;
- risk-free rate of 0 in the historical implementation.

The forward implementation was later tested for exact parity against the historical helper. At the 2026-09-30 information set, the forward and historical Maximum-Sharpe weights matched to machine precision:

```text
SPY   0.183113040417
QQQ   0.400000000000
TLT   0.000000000000
GLD   0.338213574990
SCHD  0.078673384593
maximum absolute difference = 0
```

This parity test was added permanently so forward infrastructure cannot silently drift from the frozen historical construction.

## Meta-allocation research

A broader 43-feature regime specification was explored but considered too large relative to the available sample.

A simpler binary meta gate was then tested. Its objective was to predict whether Maximum Sharpe would outperform Equal Weight over the next three months.

Frozen Meta Allocation V1 specification:

- 12 market-derived features;
- StandardScaler;
- Ridge regression;
- alpha = 10;
- target = next 3-month Maximum-Sharpe excess return over Equal Weight;
- threshold = 0;
- positive prediction selects Maximum Sharpe;
- zero or negative prediction selects Equal Weight;
- canonical quarterly decision months = January, April, July, October.

Historical out-of-sample meta results:

- 63 overlapping 3-month predictions;
- Pearson correlation about 0.212;
- Spearman correlation about 0.194;
- directional accuracy about 58.7%;
- 32 Maximum-Sharpe calls;
- active-call win rate about 68.8%;
- mean realized excess during active calls about +1.091%;
- mean selected excess about +0.554%.

These statistics are exploratory historical evidence, not forward results.

## Non-overlapping phase robustness

Three quarterly non-overlapping schedules were examined to understand sensitivity to decision timing.

Canonical phase 0 was retained because it was the prospectively defined/original schedule, not because it produced the best ex-post result.

Phase-0 bootstrap summary at 5 bps transaction costs:

- 21 non-overlapping quarterly blocks;
- selector annualized return about 11.72%;
- Equal Weight annualized return about 9.99%;
- Maximum Sharpe annualized return about 11.26%;
- mean quarterly selector excess versus Equal Weight about +0.402%;
- mean quarterly selector excess versus Maximum Sharpe about +0.112%;
- bootstrap probability selector return exceeded Equal Weight about 94.1%;
- bootstrap probability selector return exceeded Maximum Sharpe about 64.2%.

Interpretation: historical evidence was stronger versus Equal Weight than versus Maximum Sharpe. Performance contribution was concentrated in a small number of periods. Because of limited effective sample size and repeated research decisions, the model was frozen rather than further optimized on history.

## Frozen research decision

Meta Allocation V1 was frozen with status:

```text
FROZEN_RESEARCH_CANDIDATE
```

Intended use:

```text
forward_test_and_paper_money_only
```

The frozen artifact lives under:

```text
models/meta_allocation_v1/
```

Future feature engineering, hyperparameter changes, macro features, different thresholds, or alternative schedules should be treated as a new model version.

## Macro-feature experiment

Federal Reserve / FRED series were explored as possible macro-regime features.

The macro feature set was not promoted into Meta Allocation V1 because current FRED histories can contain revisions and the research pipeline was not fully vintage-safe. ALFRED/vintage-aware reconstruction would be required before making strong historical claims about information that was truly available at each decision date.

## Forward signal provenance

The first version of the forward signal audit hashed the entire source market-data file. This was replaced with hashing of the exact deterministic return slice used by the signal:

- `input_market_data_hash`;
- `input_market_data_end`;
- `source_market_data_end`.

This change prevents unrelated future rows or source-file changes from contaminating decision provenance.

No backward compatibility was retained for the obsolete schema because no genuine forward ledger existed yet.

## Forward scheduling experiment

The frozen quarterly state schedule is:

- January;
- April;
- July;
- October.

The state is held for three months, while optimizer target weights are refreshed monthly.

The first dry schedule simulation validated:

```text
2026-10-30  new quarterly decision
2026-11-02  earliest eligible target date

2026-11-30  retain October state, age 1
2026-12-01  earliest eligible target date

2026-12-31  retain October state, age 2
2027-01-04  earliest eligible target date

2027-01-29  October state age 3
            old state stale
            new quarterly decision required
```

An important bug was discovered during this dry run: generic `pandas.BDay` logic treated 2027-01-01 as a business day even though New Year's Day is not an NYSE trading session.

The implementation was changed to use the XNYS exchange calendar. Regression tests were added for month-end and next-session behavior.

## Order-plan experiment

A synthetic $100,000 portfolio was used to validate whole-share rebalance logic:

Starting holdings:

```text
SPY   100 @ $500
QQQ    50 @ $600
TLT     0 @ $100
GLD     0 @ $200
SCHD    0 @ $80
Cash        $20,000
```

Equal-weight target:

```text
20% each
```

Generated dry-run trades:

```text
SPY   SELL 60
QQQ   SELL 17
TLT   BUY 200
GLD   BUY 100
SCHD  BUY 250
```

At 5 bps transaction cost:

```text
Portfolio value before     $100,000.00
Gross trade notional       $100,200.00
Turnover                       100.20%
Estimated transaction cost     $50.10
Residual cash                  $149.90
Ending portfolio value      $99,949.90
```

The order planner enforces:

- long-only holdings;
- whole shares;
- no margin;
- non-negative residual cash;
- deterministic order-plan hashing;
- value conservation after estimated costs.

No order-submission methods exist in this module.

## Audit-ledger experiment

The order-plan audit layer writes two append-only ledgers:

- one summary row per unique `order_plan_hash`;
- one detail row per asset per plan.

Idempotency tests verify that re-running the same plan returns `EXISTS` rather than duplicating records. Conflicting records with the same plan identity are rejected.

## Broker-independent PortfolioSnapshot

A `PortfolioSnapshot` boundary was introduced to isolate portfolio logic from brokers.

It normalizes:

- snapshot timestamp;
- price timestamp;
- account mode;
- cash;
- holdings;
- reference prices;
- holdings value;
- total portfolio value;
- deterministic snapshot hashes.

Validation rejects negative cash, short positions, fractional holdings, unsupported assets, stale prices, missing prices, non-finite values, naive timestamps, and unsupported live modes.

## Synthetic paperMoney-style adapter

A synthetic broker payload adapter was created to test how paper/simulated account data could be normalized without connecting to Schwab.

This stage was fully synthetic:

```text
SCHWAB AUTHENTICATION USED: NO
BROKER NETWORK CALLS: NO
ORDERS SUBMITTED: NO
```

## Schwab read-only architecture

A dedicated `SchwabReadOnlyClient` boundary was introduced.

Supported operations:

- retrieve account-number/hash records;
- resolve an account hash;
- retrieve one account with balances and positions;
- retrieve quotes for the frozen universe.

Deliberately absent:

- `place_order`;
- `replace_order`;
- `cancel_order`;
- `preview_order`.

The purpose is architectural containment: although the underlying Trader API product supports trading, portfolio-engine code receives only the restricted wrapper.

## Schwab snapshot normalization

A separate adapter converts raw Schwab read-only payloads into `PortfolioSnapshot`.

The account mode is labeled:

```text
SCHWAB_READ_ONLY
```

rather than `PAPERMONEY`, because real Trader API account data must not be mislabeled as thinkorswim paperMoney.

## OAuth bootstrap

The OAuth bootstrap loads local environment configuration, delegates OAuth/token management to `schwab-py`, and immediately wraps the authenticated raw client in `SchwabReadOnlyClient`.

Secrets are excluded from dataclass representation and should never be committed.

The real OAuth smoke test initially returned:

```text
invalid_client
Unauthorized
```

Investigation showed that the literal placeholder `YOUR_REAL_SCHWAB_APP_KEY` was still loaded from `.env`. No token file was created. The code path itself reached the browser-assisted OAuth flow correctly.

The project then submitted a request for Trader API - Individual access. Schwab reported that administrator review could take up to two business days.

As of 2026-10-07, real Trader API access remained pending.

## Test checkpoints

Recorded checkpoints during forward-infrastructure development:

```text
163 passed  weight schema + historical parity
173 passed  XNYS scheduling
184 passed  order planning + provenance
189 passed  order-plan audit
210 passed  portfolio snapshot integration
224 passed  synthetic paperMoney adapter
234 passed  Schwab read-only transport
245 passed  Schwab read-only snapshot adapter
261 passed  OAuth/bootstrap boundary
262 passed  full synthetic auth → read-only → snapshot → plan → audit chain
```

The final synthetic end-to-end test validates interface compatibility across every current boundary without requiring broker credentials or making a network request.

## Current forward-testing rule

Historical model development for Meta Allocation V1 is closed.

Allowed changes to V1:

- bug fixes;
- faithful implementation fixes;
- provenance improvements;
- test coverage;
- broker-independent plumbing;
- safety controls.

Not allowed without creating a new model version:

- retuning alpha;
- changing the feature set;
- changing the target horizon;
- changing the decision threshold;
- choosing a quarterly phase because it performed better historically;
- changing the portfolio universe or weight cap based on additional historical optimization.

## Interpretation and publication notes

Readers should keep these limitations in view:

- 104 monthly walk-forward observations is a modest sample.
- The 63 three-month meta predictions overlap.
- The canonical quarterly analysis contains only 21 non-overlapping decisions.
- Multiple research choices were explored before freezing V1.
- Bootstrap results characterize the observed historical sample; they are not a guarantee of future returns.
- Adjusted market data can be revised.
- Macro histories can be revised and were not fully vintage-safe.
- Transaction-cost assumptions omit taxes, bid/ask dynamics, market impact, and some execution frictions.
- No real-money execution results exist in the current project stage.
- No real Schwab orders have been submitted.

See [SOURCES_AND_ATTRIBUTION.md](SOURCES_AND_ATTRIBUTION.md) for academic, software, data, calendar, and broker references.
