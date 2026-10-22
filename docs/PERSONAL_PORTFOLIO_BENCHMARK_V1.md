# Personal Portfolio Benchmark V1

## Purpose

Personal Portfolio Benchmark V1 is a read-only research layer that compares a
private local portfolio snapshot with the frozen optimizer allocation, an
equal-weight benchmark, and SPY.

It does not modify optimizer methodology and does not create or submit orders.

## Privacy boundary

Personal holdings belong under:

```text
data/private/personal_portfolio/
```

That path is ignored by Git. Do not store brokerage account numbers,
authentication data, tokens, or other credentials in the repository.

The committed validation summary contains aggregate statistics and source hashes,
not symbol-level personal holdings.

## Local personal snapshot contract

Minimum CSV columns:

```text
symbol
market_value
```

Example:

```csv
symbol,market_value
AAA,12500
BBB,8300
```

## Metrics

The benchmark computes:

- personal portfolio value;
- annualized volatility;
- maximum drawdown;
- HHI concentration;
- effective number of positions;
- optimizer volatility, drawdown, and concentration;
- equal-weight volatility;
- SPY volatility;
- annualized tracking error between personal and optimizer portfolios;
- portfolio overlap;
- L1 weight difference;
- implied one-way turnover;
- component risk contributions for local diagnostics.

Definitions:

```text
portfolio_overlap = sum(min(personal_weight, optimizer_weight))
weight_difference = sum(abs(personal_weight - optimizer_weight))
turnover = 0.5 * weight_difference
concentration = sum(weight^2)
effective_positions = 1 / concentration
```

## Validation output

Aggregate publication-safe summary:

```text
reports/validation/portfolio_comparison/<YYYY-MM-DD>/summary.json
```

The summary records source SHA-256 hashes so the private source snapshot can be
reproduced locally without publishing the holdings themselves.

## Command

```bash
./.venv/Scripts/python.exe scripts/build_personal_portfolio_benchmark.py \
  --as-of-session 2026-10-16 \
  --personal-snapshot data/private/personal_portfolio/portfolio_2026-10-16.csv \
  --optimizer-allocation reports/validation/scanner_allocations/2026-10-16/candidate_allocation.csv \
  --optimizer-manifest reports/validation/scanner_allocations/2026-10-16/manifest.json \
  --prices-csv <synchronized_price_matrix.csv>
```

The price matrix must include every non-cash symbol in the personal portfolio,
the optimizer allocation, and SPY over the requested lookback.

## Roadmap boundary

This experiment answers:

> Does the frozen optimizer produce a materially different risk profile from the
> user's existing portfolio, equal weight, and SPY?

Only after the research comparison is established should a separate execution
experiment use a private current-portfolio snapshot as an initial state for
simulated SELL/HOLD/BUY reconciliation.
