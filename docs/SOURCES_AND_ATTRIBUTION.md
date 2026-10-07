# Sources, Attribution, and Reproducibility References

This project is a personal research and educational implementation of portfolio construction, walk-forward evaluation, model selection, and broker-integration patterns. It is intended to make the methods, assumptions, source provenance, and software dependencies inspectable and reproducible.

A citation in this document may serve one of several different purposes. To avoid implying that every cited paper or website was directly supplied by the project owner or consulted during the original experiment, this file separates sources by provenance.

A citation does **not** imply endorsement of this repository by the cited author, publisher, exchange, broker, data provider, or software project.

## Source-provenance categories

The references below use three provenance categories.

### A. Project-owner supplied source material

These are documents or source materials directly supplied by the project owner during development.

At the current publication checkpoint, the clearly documented supplied source is:

- **Charles Schwab Developer Portal product catalog PDF** supplied by the project owner on 2026-10-07.
  - The supplied material identifies **Trader API - Individual** as the personal-use API product for a developer's own self-directed Schwab Brokerage account and describes access to real-time account information, market data, and trade capability.
  - This supplied document was used to confirm which Schwab API product matched the project's intended personal-use integration.

If additional papers, datasets, documentation, or other materials are supplied by the project owner later, they should be added to this section with the date and the part of the project they informed.

### B. Project-selected data sources, APIs, and software

These are sources, services, libraries, or APIs that were explicitly used or selected as part of the project implementation. The formal URLs and bibliographic details in this document may have been added later for reproducibility and attribution.

Examples include:

- Yahoo Finance data accessed through `yfinance`;
- Federal Reserve Economic Data (FRED);
- `exchange_calendars`;
- SciPy;
- scikit-learn;
- Charles Schwab Trader API;
- `schwab-py`.

Listing a formal reference here does **not** mean the project owner originally supplied that exact webpage or publication.

### C. External methodological references added for attribution

These references were added to credit the original or canonical literature behind methods implemented in the repository.

They should be understood as **methodological attribution references**, not as a claim that the project owner supplied, read, or relied on that exact publication during the original research process.

Examples include the original literature associated with:

- mean-variance portfolio selection;
- covariance shrinkage;
- equal-risk-contribution / risk parity;
- Hierarchical Risk Parity;
- the Sharpe ratio;
- Ridge regression;
- bootstrap resampling;
- time-series cross-validation.

This distinction is important for accurate research provenance. Future public writing should avoid phrases such as "the experiment was based on [paper]" unless the historical development record actually supports that statement.

---

# A. Project-owner supplied source material

## Charles Schwab Developer Portal product catalog

**Provenance:** directly supplied by the project owner.

The supplied Schwab Developer Portal document identifies **Trader API - Individual** as a personal-use API for the developer's own self-directed brokerage account, including real-time account information, market data, and trade capability.

The document also distinguishes other Schwab API products, including account/data aggregation products, from Trader API - Individual.

This source was used to confirm the appropriate Schwab product for the project's planned personal account integration.

For public reproduction, readers should consult the current Schwab Developer Portal because product descriptions, eligibility, approval processes, and API terms can change.

Public Schwab portal:

- https://developer.schwab.com/

---

# B. Project-selected data sources, APIs, and software

## Yahoo Finance / yfinance

**Provenance:** project-selected data/retrieval source. Formal citation links added for reproducibility and attribution.

Historical market data used in the research pipeline were retrieved through the open-source `yfinance` package.

- yfinance project/documentation: https://github.com/ranaroussi/yfinance
- yfinance documentation and legal disclaimer: https://github.com/ranaroussi/yfinance/blob/main/doc/source/index.rst

Important: yfinance states that it is not affiliated with or endorsed by Yahoo, that it is intended for research and educational purposes, and that users must consult Yahoo's terms for rights governing downloaded data.

This repository should not redistribute Yahoo Finance datasets as a substitute for the original provider.

Adjusted historical data can be revised by upstream providers. The forward-test design therefore hashes the exact input slice used for each decision so later revisions can be detected.

## Federal Reserve Economic Data (FRED / ALFRED)

**Provenance:** project-selected macroeconomic data source. Formal documentation links added later for reproducibility.

- Federal Reserve Bank of St. Louis, FRED API overview: https://fred.stlouisfed.org/docs/api/fred/overview.html
- FRED API documentation: https://fred.stlouisfed.org/docs/api/fred/

FRED provides current historical series, while ALFRED provides archival/vintage information.

Research results that use revised current histories are not automatically vintage-safe. Meta Allocation V1 therefore did **not** promote the exploratory macro feature set into the frozen forward candidate.

A future macro model making claims about what was known at a historical date should use an appropriate vintage-aware process.

## exchange_calendars

**Provenance:** project-selected software dependency.

- `exchange_calendars` project: https://github.com/gerrymanoim/exchange_calendars

The project uses the XNYS calendar for NYSE trading-session month ends and next-session calculations.

This replaced a generic weekday-only business-day convention after a dry-run exposed 2027-01-01 as a false eligible execution date.

## SciPy

**Provenance:** project-selected numerical software dependency.

- SciPy, `scipy.optimize.minimize(method="SLSQP")` documentation: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html

The project uses SLSQP for constrained numerical portfolio optimization.

## scikit-learn

**Provenance:** project-selected machine-learning/statistical software dependency.

Relevant implementations include:

- `LedoitWolf`: https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html
- `Ridge`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

These are the software implementations used in the repository.

The corresponding academic references are listed separately under methodological attribution.

## Charles Schwab Trader API

**Provenance:** project-selected broker/API integration; product choice confirmed using the project-owner supplied Schwab Developer Portal document.

- Charles Schwab Developer Portal: https://developer.schwab.com/
- Trader API - Individual product page: https://developer.schwab.com/products/trader-api--individual%26gt%3B

The repository intentionally exposes only a **read-only application boundary** at the current stage.

The internal `SchwabReadOnlyClient` supports:

- account-hash discovery;
- account and position reads;
- quote retrieval.

It deliberately exposes no:

- `place_order`;
- `replace_order`;
- `cancel_order`;
- `preview_order`.

As of 2026-10-07, real Trader API access for this project was pending Schwab approval. All broker-facing integration tests documented at this checkpoint were synthetic/mocked, and no real order had been submitted.

## schwab-py

**Provenance:** project-selected OAuth/client library.

- schwab-py authentication documentation: https://schwab-py.readthedocs.io/en/latest/auth.html
- schwab-py client documentation: https://schwab-py.readthedocs.io/en/stable/client.html
- PyPI project: https://pypi.org/project/schwab-py/

The project uses `schwab-py` as an OAuth/client helper rather than reimplementing Schwab OAuth.

Token and credential files are local secrets and must not be committed.

---

# C. External methodological references added for attribution

The references in this section were added to credit the intellectual foundations of methods used in the repository.

Unless another project record explicitly says otherwise, they should **not** be described as sources directly supplied by the project owner or necessarily consulted during the original implementation.

## Portfolio selection

- Markowitz, H. (1952). "Portfolio Selection." *The Journal of Finance*, 7(1), 77-91. https://doi.org/10.1111/j.1540-6261.1952.tb01525.x
  - Methodological reference for mean-variance portfolio-selection concepts used by the project.

## Equal-weight benchmarking

- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). "Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?" *The Review of Financial Studies*, 22(5), 1915-1953. https://doi.org/10.1093/rfs/hhm075
  - Methodological context for evaluating optimized portfolios against a simple 1/N benchmark.

## Covariance shrinkage

- Ledoit, O., & Wolf, M. (2004). "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices." *Journal of Multivariate Analysis*, 88(2), 365-411. https://doi.org/10.1016/S0047-259X(03)00096-4
  - Methodological reference for Ledoit-Wolf covariance shrinkage.

Software implementation used by the project:

- scikit-learn `LedoitWolf`: https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html

## Risk parity / equal-risk contribution

- Maillard, S., Roncalli, T., & Teïletche, J. (2010). "The Properties of Equally Weighted Risk Contribution Portfolios." *The Journal of Portfolio Management*, 36(4), 60-70. https://doi.org/10.3905/jpm.2010.36.4.060
  - Methodological reference for equal-risk-contribution / risk-parity concepts.

## Hierarchical Risk Parity

- López de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out-of-Sample." *The Journal of Portfolio Management*, 42(4), 59-69. https://doi.org/10.3905/jpm.2016.42.4.059
  - Methodological reference for Hierarchical Risk Parity.

## Sharpe ratio

- Sharpe, W. F. (1994). "The Sharpe Ratio." *The Journal of Portfolio Management*, 21(1), 49-58. https://doi.org/10.3905/jpm.1994.409501
  - Methodological reference for risk-adjusted performance measurement.

## Ridge regression

- Hoerl, A. E., & Kennard, R. W. (1970). "Ridge Regression: Biased Estimation for Nonorthogonal Problems." *Technometrics*, 12(1), 55-67. https://doi.org/10.1080/00401706.1970.10488634
  - Methodological reference for Ridge regression.

Software implementation used by the project:

- scikit-learn `Ridge`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

## Time-series cross-validation / rolling-origin evaluation

- Hyndman, R. J., & Athanasopoulos, G. *Forecasting: Principles and Practice*, section on time-series cross-validation / rolling forecasting origin: https://otexts.robjhyndman.com/fpp2/accuracy.html
  - Methodological reference for chronological evaluation in which training information precedes the test observation.

This is an attribution/reference point for the general validation concept. Readers should inspect the repository's own walk-forward code for the exact implementation.

## Bootstrap resampling

- Efron, B. (1979). "Bootstrap Methods: Another Look at the Jackknife." *The Annals of Statistics*, 7(1), 1-26. https://doi.org/10.1214/aos/1176344552
  - Foundational methodological reference for bootstrap resampling.

The project's robustness analysis should be reproduced from its implementation and experiment log; this citation does not by itself define the project's exact resampling unit or interpretation.

---

# Software and reproducibility

The Python environment is defined by `requirements.txt` and `requirements-lock.txt`.

Important implementation libraries include:

- pandas;
- NumPy;
- SciPy;
- scikit-learn;
- yfinance;
- exchange_calendars;
- python-dotenv;
- requests;
- pytest;
- schwab-py.

For a research result to be considered reproducible, record at minimum:

1. Git commit SHA and, when applicable, the frozen model artifact version.
2. Python version and `requirements-lock.txt`.
3. Data source and observation cutoff.
4. Exact input-data hash used by the forward decision.
5. Target-weights hash, holdings snapshot hash, price snapshot hash, and order-plan hash where applicable.
6. Transaction-cost assumption.
7. Test-suite status.
8. Whether the result is exploratory, historical out-of-sample, synthetic forward infrastructure, genuine forward observation, paper/simulated execution, or live execution.

See `REPRODUCIBILITY.md` for the detailed replication protocol.

---

# Citation policy for future publication

When publishing results from this project:

- identify whether each source was **project-owner supplied**, **project-selected**, or **added later for methodological attribution**;
- do not say the project owner "used" or "relied on" an academic paper unless the development record supports that claim;
- cite the original academic source for a method when presenting the method publicly;
- also cite the software package used for the implementation when software behavior is material;
- cite the original data provider and retrieval tool separately;
- state sample dates, universe, rebalance frequency, transaction-cost assumption, and weight constraints;
- distinguish exploratory model selection from historical out-of-sample, synthetic, genuine forward, paper, and live evidence;
- disclose overlapping observations and the effective number of non-overlapping decisions;
- report material implementation failures or methodological changes that affected results;
- do not characterize simulated, historical, or paper results as live trading performance;
- do not imply endorsement by Charles Schwab, Yahoo, the Federal Reserve Bank of St. Louis, NYSE, any software project, or any cited researcher.

For a formal public write-up, also follow `PUBLISHING_GUIDE.md`.

---

# Disclaimer

This repository is research and educational software, not investment, legal, tax, or brokerage advice.

Historical, simulated, synthetic, and forward-test results do not guarantee future performance.

Data and API use remain subject to applicable provider terms, licenses, and contracts.

References and attribution improve transparency and reproducibility, but they do not transfer responsibility to the cited author or organization and do not by themselves provide a liability shield.

The developer and each user remain responsible for complying with applicable laws, licenses, broker agreements, data-provider terms, and publication requirements.
