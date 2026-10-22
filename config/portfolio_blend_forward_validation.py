from __future__ import annotations

FORWARD_VALIDATION_NAME = "Portfolio Blend Forward Validation V1"
FORWARD_VALIDATION_VERSION = "portfolio-blend-forward-validation-v1"
FORWARD_VALIDATION_BUILDER_VERSION = "1.0.0"

FROZEN_OPTIMIZER_BLEND_WEIGHTS = (
    0.00,
    0.20,
    0.30,
    1.00,
)

FORMAL_MINIMUM_SESSIONS = 20
CHECKPOINT_SESSIONS = (
    20,
    60,
    120,
    252,
)

BLEND_SCOPE = "priceable_personal_sleeve_only"
WEIGHTING_METHOD = "constant_mix_daily"
RETUNING_PERMITTED = False
ORDER_SUBMISSION_PERMITTED = False
