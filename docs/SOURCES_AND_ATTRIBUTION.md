# Sources, Attribution, and Reproducibility References

This project is a personal research and educational implementation of portfolio construction, walk-forward evaluation, model selection, and broker-integration patterns. It is intended to make the methods, assumptions, and software dependencies inspectable and reproducible.

Citations below acknowledge the individuals, organizations, software projects, and data providers whose work informed the implementation. A citation does **not** imply endorsement of this repository by the cited author, publisher, exchange, broker, or software project.

## Academic and methodological references

### Portfolio selection and benchmarking

- Markowitz, H. (1952). "Portfolio Selection." *The Journal of Finance*, 7(1), 77-91. https://doi.org/10.1111/j.1540-6261.1952.tb01525.x
  - Foundational mean-variance portfolio-selection framework.
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). "Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?" *The Review of Financial Studies*, 22(5), 1915-1953. https://doi.org/10.1093/rfs/hhm075
  - Motivation for retaining an equal-weight 1/N benchmark and evaluating optimized portfolios out of sample.

### Covariance estimation

- Ledoit, O., & Wolf, M. (2004). "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices." *Journal of Multivariate Analysis*, 88(2), 365-411. https://doi.org/10.1016/S0047-259X(03)00096-4
  - Basis for the Ledoit-Wolf shrinkage covariance estimator used by the engine.
- scikit-learn, `LedoitWolf` documentation: https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html
  - Software implementation used in this repository.

### Risk parity and hierarchical allocation

- Maillard, S., Roncalli, T., & Teïletche, J. (2010). "The Properties of Equally Weighted Risk Contribution Portfolios." *The Journal of Portfolio Management*, 36(4), 60-70. https://doi.org/10.3905/jpm.2010.36.4.060
  - Reference for equal-risk-contribution / risk-parity concepts.
- López de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out-of-Sample." *The Journal of Portfolio Management*, 42(4), 59-69. https://doi.org/10.3905/jpm.2016.42.4.059
  - Reference for Hierarchical Risk Parity (HRP).

### Performance measurement

- Sharpe, W. F. (1994). "The Sharpe Ratio." *The Journal of Portfolio Management*, 21(1), 49-58. https://doi.org/10.3905/jpm.1994.409501
  - Reference for risk-adjusted performance measurement.

### Regularized regression

- Hoerl, A. E., & Kennard, R. W. (1970). "Ridge Regression: Biased Estimation for Nonorthogonal Problems." *Technometrics*, 12(1), 55-67. https://doi.org/10.1080/00401706.1970.10488634
  - Statistical basis for the Ridge model used by the Meta Allocation V1 gate.
- scikit-learn, `Ridge` documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
  - Software implementation used by the project.

### Walk-forward / time-series evaluation

- Hyndman, R. J., & Athanasopoulos, G. *Forecasting: Principles and Practice*, section on time-series cross-validation / rolling forecasting origin: https://otexts.robjhyndman.com/fpp2/accuracy.html
  - Reference for chronological evaluation in which training data precede the test observation.
- Efron, B. (1979). "Bootstrap Methods: Another Look at the Jackknife." *The Annals of Statistics*, 7(1), 1-26. https://doi.org/10.1214/aos/1176344552
  - Foundational reference for bootstrap resampling. This project uses bootstrap-style resampling for robustness analysis; readers should inspect the implementation and experiment log for the exact resampling unit and interpretation.

### Numerical optimization

- SciPy, `scipy.optimize.minimize(method="SLSQP")` documentation: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html
  - Numerical constrained-optimization routine used by portfolio optimizers.

## Data sources and market-calendar software

### Yahoo Finance / yfinance

Historical market data used in the research pipeline were retrieved through the open-source `yfinance` package.

- yfinance project/documentation: https://github.com/ranaroussi/yfinance
- yfinance documentation and legal disclaimer: https://github.com/ranaroussi/yfinance/blob/main/doc/source/index.rst

Important: yfinance states that it is not affiliated with or endorsed by Yahoo, that it is intended for research and educational purposes, and that users must consult Yahoo's terms for rights governing downloaded data. This repository does not redistribute Yahoo Finance datasets as a substitute for the original provider.

Adjusted historical data can be revised by upstream providers. The forward-test design therefore hashes the exact input slice used for each decision so later revisions can be detected.

### Federal Reserve Economic Data (FRED / ALFRED)

- Federal Reserve Bank of St. Louis, FRED API overview: https://fred.stlouisfed.org/docs/api/fred/overview.html
- FRED API documentation: https://fred.stlouisfed.org/docs/api/fred/

FRED provides current historical series, while ALFRED provides archival/vintage information. Research results that use revised current histories are not automatically vintage-safe. Meta Allocation V1 therefore did **not** promote the exploratory macro feature set into the frozen forward candidate.

### Exchange calendars

- `exchange_calendars` project: https://github.com/gerrymanoim/exchange_calendars

The project uses the XNYS calendar for NYSE trading-session month ends and next-session calculations. This replaced a generic weekday-only `BusinessDay` convention after a dry-run exposed January 1, 2027 as a false eligible execution date.

## Schwab / broker integration

### Charles Schwab

- Charles Schwab Developer Portal: https://developer.schwab.com/
- Trader API - Individual product page: https://developer.schwab.com/products/trader-api--individual%26gt%3B

Schwab describes Trader API - Individual as a personal-use application interface for a developer's own self-directed brokerage account, including real-time account information, market data, and trade capability.

This repository intentionally exposes only a **read-only application boundary** at the current stage. The internal `SchwabReadOnlyClient` supports account-hash discovery, account/position reads, and quotes. It deliberately exposes no `place_order`, `replace_order`, `cancel_order`, or `preview_order` methods.

As of 2026-10-07, real Trader API access for this project was pending Schwab approval. All broker-facing integration tests documented in the repository were synthetic/mocked; no real order was submitted.

### schwab-py

- schwab-py authentication documentation: https://schwab-py.readthedocs.io/en/latest/auth.html
- schwab-py client documentation: https://schwab-py.readthedocs.io/en/stable/client.html
- PyPI project: https://pypi.org/project/schwab-py/

The project uses `schwab-py` as an OAuth/client helper rather than reimplementing Schwab OAuth. Token and credential files are local secrets and must not be committed.

## Software and reproducibility

The Python environment is defined by `requirements.txt` and `requirements-lock.txt`. Important implementation libraries include pandas, NumPy, SciPy, scikit-learn, yfinance, exchange_calendars, python-dotenv, requests, pytest, and schwab-py.

For a research result to be considered reproducible, record at minimum:

1. Git commit SHA and, when applicable, the frozen model artifact version.
2. Python version and `requirements-lock.txt`.
3. Data source and observation cutoff.
4. Exact input-data hash used by the forward decision.
5. Target-weights hash, holdings snapshot hash, price snapshot hash, and order-plan hash where applicable.
6. Transaction-cost assumption.
7. Test-suite status.
8. Whether the result is exploratory, historical out-of-sample, synthetic forward infrastructure, or genuine forward observation.

## Citation policy for future publication

When publishing results from this project:

- Cite the original academic source for each method, not only the Python package implementing it.
- Cite the software package used for the implementation when software behavior is material.
- Cite the original data provider and retrieval tool separately.
- State the sample dates, universe, rebalance frequency, transaction-cost assumption, and any weight constraints.
- Distinguish exploratory model selection from held-out or genuine forward evidence.
- Disclose overlapping observations and the effective number of non-overlapping decisions.
- Do not characterize simulated, historical, or paper results as live trading performance.
- Do not imply endorsement by Charles Schwab, Yahoo, the Federal Reserve Bank of St. Louis, NYSE, or any cited researcher.

## Disclaimer

This repository is research and educational software, not investment, legal, tax, or brokerage advice. Historical and simulated results do not guarantee future performance. Data and API use remain subject to the applicable provider terms. References and attribution improve transparency and reproducibility but do not remove the developer's or user's responsibility to comply with applicable laws, licenses, contracts, and broker/data-provider terms.
