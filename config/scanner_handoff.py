from __future__ import annotations

HANDOFF_CONTRACT_VERSION = "scanner-candidate-handoff-v1"
EXPECTED_SCANNER_ALPHA_SPEC_VERSION = "scanner-alpha-v1"
EXPECTED_SCANNER_PROVIDER = "schwab"

PRIMARY_SIGNAL_NAME = "momentum_252d_ex_20d"
SELECTION_POLICY = "top_ranked_alpha_complete"

CANDIDATE_COLUMNS = (
    "handoff_contract_version",
    "scanner_alpha_spec_version",
    "scanner_provider",
    "as_of_session",
    "generated_at_utc",
    "scanner_snapshot_sha256",
    "source_alpha_manifest_sha256",
    "source_timestamp_utc",
    "symbol",
    "scanner_rank",
    "scanner_score",
    "scanner_percentile",
    "scanner_eligible",
    "close",
    "primary_signal_name",
    "primary_signal_value",
    "ema_50_to_200",
)

MANIFEST_REQUIRED_FIELDS = (
    "handoff_contract_version",
    "scanner_alpha_spec_version",
    "scanner_provider",
    "as_of_session",
    "generated_at_utc",
    "scanner_snapshot_sha256",
    "source_alpha_manifest_sha256",
    "candidate_count",
    "candidate_limit",
    "selection_policy",
    "candidate_csv_sha256",
    "immutable",
)
