from __future__ import annotations

BENCHMARK_NAME = "Personal Portfolio Benchmark V1"
BENCHMARK_VERSION = "personal-portfolio-benchmark-v1"
BENCHMARK_BUILDER_VERSION = "1.0.0"

DEFAULT_LOOKBACK_RETURNS = 252
DEFAULT_BENCHMARK_SYMBOL = "SPY"
CASH_SYMBOL = "CASH"

CONCENTRATION_METRIC = "herfindahl_hirschman_weight_index"
OVERLAP_METRIC = "sum_min_weights"
WEIGHT_DIFFERENCE_METRIC = "l1_weight_distance"
TURNOVER_METRIC = "half_l1_weight_distance"
TRACKING_ERROR_METRIC = "annualized_std_personal_minus_optimizer"
