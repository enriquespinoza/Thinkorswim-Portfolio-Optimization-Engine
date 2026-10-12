from __future__ import annotations

ALLOCATION_CONTRACT_VERSION = "scanner-candidate-allocation-v1"
OPTIMIZER_NAME = "minimum_variance_ledoit_wolf"
OPTIMIZER_VERSION = "1.0.0"
COVARIANCE_ESTIMATOR = "ledoit_wolf"

DEFAULT_LOOKBACK_RETURNS = 252
DEFAULT_MIN_WEIGHT = 0.0
DEFAULT_MAX_WEIGHT = 0.10

ALLOCATION_COLUMNS = (
    "allocation_contract_version",
    "as_of_session",
    "scanner_snapshot_sha256",
    "source_candidate_history_sha256",
    "source_candidate_csv_sha256",
    "source_candidate_manifest_sha256",
    "symbol",
    "scanner_rank",
    "scanner_score",
    "scanner_percentile",
    "close",
    "optimizer_name",
    "optimizer_version",
    "covariance_estimator",
    "lookback_returns",
    "annualized_volatility",
    "target_weight",
)
