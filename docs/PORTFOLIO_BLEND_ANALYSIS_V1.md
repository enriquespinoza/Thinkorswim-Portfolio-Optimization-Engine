# Portfolio Blend Analysis V1

## Purpose

Portfolio Blend Analysis V1 tests whether allocating a fixed sleeve of the
frozen optimizer portfolio improves the risk/return profile of the user's
existing priceable portfolio sleeve.

It is a read-only research experiment. It does not alter scanner ranking,
optimizer methodology, execution targets, or brokerage holdings.

## Research question

The experiment asks:

> Does introducing 10%, 20%, 30%, 50%, or 100% of the frozen optimizer
> allocation improve the existing portfolio relative to remaining 100%
> personal?

The default fixed optimizer sleeve weights are:

- 0%
- 10%
- 20%
- 30%
- 50%
- 100%

These are predetermined research points, not optimized blend weights.

## Pricing scope

The personal portfolio snapshot can contain holdings that do not have usable
daily Schwab price-history data, such as fixed-income CUSIPs.

Portfolio Blend Analysis V1 therefore applies blend weights only to the
priceable personal sleeve used by Personal Portfolio Benchmark V1.

The unpriced sleeve remains outside the historical-return calculations and is
reported through:

- personal_pricing_coverage
- personal_unpriced_weight

This experiment does not assume zero volatility for unpriced holdings.

## Metrics

For each fixed blend:

- annualized geometric return;
- annualized volatility;
- maximum drawdown;
- Sharpe ratio with zero risk-free rate;
- Sortino ratio with zero risk-free rate;
- HHI concentration;
- effective number of positions;
- tracking error versus the personal portfolio;
- tracking error versus the optimizer;
- one-way turnover from the personal portfolio;
- largest absolute component risk contribution;
- worst rolling 20-day return;
- worst rolling 60-day return;
- changes in return, volatility, drawdown, and concentration versus personal;
- indicators showing whether volatility, drawdown, and concentration improved.

## Output

Validation artifacts:

```text
reports/validation/portfolio_blend_analysis/<YYYY-MM-DD>/
├── blend_analysis.csv
└── summary.json
```

The summary contains source SHA-256 hashes for reproducibility and contains no
symbol-level personal holdings.

## Command

```bash
./.venv/Scripts/python.exe \
  scripts/build_portfolio_blend_analysis.py \
  --as-of-session 2026-10-16 \
  --personal-snapshot data/private/personal_portfolio/portfolio_all_accounts_2026-10-16.csv \
  --optimizer-allocation reports/validation/scanner_allocations/2026-10-16/candidate_allocation.csv \
  --optimizer-manifest reports/validation/scanner_allocations/2026-10-16/manifest.json \
  --prices-csv data/private/personal_portfolio/benchmark_prices_all_accounts_2026-10-16.csv
```

## Boundary

Portfolio Blend Analysis V1 is descriptive historical research.

A favorable historical blend does not automatically become an optimizer target
or execution instruction. Any candidate blend would require separate forward
validation and paper-money evaluation before it could be considered for an
execution handoff.
