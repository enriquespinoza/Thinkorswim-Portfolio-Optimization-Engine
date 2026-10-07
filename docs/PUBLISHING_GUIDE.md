# Publishing and Attribution Guide

This guide is for blog posts, papers, project pages, demonstrations, portfolio write-ups, and other public descriptions based on this repository.

## 1. Separate attribution from endorsement

Credit the original authors, data providers, and software projects whose work informed the implementation.

Do not imply that any cited researcher, publisher, broker, exchange, software maintainer, or data provider:

- reviewed this repository;
- endorses the implementation;
- guarantees the results;
- accepts responsibility for investment outcomes.

The project is an independent implementation.

## 2. Cite both method and implementation

Where practical, cite:

1. the original academic or technical source for the method;
2. the software implementation actually used.

Example:

- Ledoit & Wolf (2004) for covariance shrinkage;
- scikit-learn `LedoitWolf` for the implementation used in code.

Do the same for Ridge regression, constrained optimization, HRP, and other material methods.

## 3. Cite data provider and retrieval tool separately

For Yahoo Finance-derived research data, identify both:

- Yahoo Finance as the upstream source;
- yfinance as the retrieval library.

For Federal Reserve data, cite FRED or ALFRED as appropriate.

Do not redistribute third-party datasets in a way that conflicts with provider terms.

## 4. Label the evidence class

Every performance figure should be labeled as one of:

- exploratory in-sample;
- historical out-of-sample;
- synthetic infrastructure validation;
- genuine forward observation;
- paper/simulated execution;
- live execution.

At the current project stage, no published result should be described as live trading performance.

## 5. Disclose model-selection history

Meta Allocation V1 was selected after multiple exploratory investigations.

Public reporting should disclose:

- 43-feature regime research was explored and rejected;
- a simpler 12-feature Ridge gate was retained;
- quarterly phase checks were conducted;
- the canonical Jan/Apr/Jul/Oct phase was retained because it was the original/prospectively fixed schedule, not because it was the strongest ex-post phase;
- historical development was frozen before genuine forward evaluation.

This reduces the risk that readers interpret historical results as if no model-selection process occurred.

## 6. Disclose effective sample size

For Meta Allocation V1 historical reporting, distinguish:

- 63 overlapping 3-month predictions;
- 21 non-overlapping quarterly decisions on the canonical phase-0 schedule.

Do not present 63 overlapping observations as if they were 63 independent quarterly decisions.

## 7. State implementation assumptions

At minimum, disclose:

- universe;
- maximum asset weight;
- rebalance schedule;
- transaction-cost assumption;
- whole-share restrictions if relevant;
- long-only/no-margin assumptions if relevant;
- risk-free-rate convention;
- covariance estimator;
- expected-return estimator;
- decision threshold;
- data cutoff.

## 8. Report failed experiments and bugs

Future publication should include material failures that changed the implementation.

Examples already documented:

- the original generic business-day schedule incorrectly treated 2027-01-01 as an eligible session;
- XNYS exchange-calendar logic replaced generic weekday logic;
- an initial OAuth attempt failed because placeholder client credentials were still loaded;
- macro features were not promoted because the historical data path was not fully vintage-safe.

These failures are useful research information and should not be hidden.

## 9. Reproducibility bundle

For a publishable result, archive or record:

- Git commit SHA;
- model artifact version;
- requirements lock;
- data cutoff;
- relevant input hashes;
- test-suite result;
- generated research tables;
- methodology notes;
- citations.

If practical, create a GitHub release or immutable tag for a published result.

## 10. Financial-results language

Prefer precise language such as:

> "In the historical walk-forward sample, the selector produced..."

Avoid language such as:

> "The strategy will return..."

or:

> "The model proves..."

Historical evidence does not establish future performance.

## 11. Broker/API language

Charles Schwab Trader API - Individual supports account information, market data, and trading capability, but the current repository intentionally restricts application code to a read-only wrapper.

Do not describe the current project as an automated trading bot or live execution system.

As of 2026-10-07, real Trader API approval was pending and the documented end-to-end broker tests were mocked/synthetic.

## 12. Suggested project citation

GitHub can render the repository's `CITATION.cff`.

When discussing a specific method or dataset, citing this repository alone is not enough. Also cite the original sources in [SOURCES_AND_ATTRIBUTION.md](SOURCES_AND_ATTRIBUTION.md).

## 13. Disclaimer language

A public write-up should include a disclaimer substantially similar to:

> This material describes research and educational software. It is not investment, legal, tax, or brokerage advice. Historical, simulated, synthetic, and forward-test results do not guarantee future performance. Data and APIs may be revised or unavailable, and real trading introduces costs and risks not fully represented in the research model.

A disclaimer does not replace compliance with applicable licenses, provider terms, laws, or contractual obligations.

## 14. Primary reference list

Use [SOURCES_AND_ATTRIBUTION.md](SOURCES_AND_ATTRIBUTION.md) as the maintained reference list and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) as the experiment-replication protocol.
