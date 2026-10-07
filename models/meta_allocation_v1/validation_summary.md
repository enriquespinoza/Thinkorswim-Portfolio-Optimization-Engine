# Meta Allocation V1 Validation Summary

## Status

**FROZEN RESEARCH CANDIDATE**

Historical model development ended when this artifact was created.

No feature, target, threshold, alpha, optimizer, horizon, or historical
decision-rule changes should be made to V1 after this point.

Any such modification must become a separately named V2 research model.

## Model

- Model: Ridge regression
- Alpha: 10.0
- Features: 12
- Forecast horizon: 3 months
- Decision threshold: 0.0
- Positive state: Maximum Sharpe
- Negative state: Equal Weight
- Research transaction cost: 5.0 bps
- Historical training rows at freeze: 102
- Information as of: 2026-09-30

## Final Paired Portfolio Bootstrap

The final validation resamples complete, non-overlapping three-month
portfolio blocks.

It uses realized portfolio returns after the configured transaction-cost
assumption rather than overlapping prediction labels.

 phase  quarterly_blocks  selector_annualized_return  equal_weight_annualized_return  maximum_sharpe_annualized_return  selector_quarterly_sharpe  equal_weight_quarterly_sharpe  maximum_sharpe_quarterly_sharpe  mean_quarterly_excess_vs_ew  mean_quarterly_excess_vs_ms  p_return_gt_equal_weight  p_return_gt_maximum_sharpe  p_sharpe_gt_equal_weight  p_sharpe_gt_maximum_sharpe  ew_excess_ci_low  ew_excess_ci_high  ms_excess_ci_low  ms_excess_ci_high  top3_contribution_vs_ew  top3_contribution_vs_ms
     0                21                    0.117223                        0.099905                          0.112569                   1.004747                       0.874034                         0.977275                     0.004016                     0.001116                    0.9414                      0.6421                    0.9349                      0.5909         -0.001096           0.009504         -0.004007           0.006617                 0.956458                 3.058108
     1                21                    0.130639                        0.104544                          0.120296                   1.240002                       1.243239                         1.129358                     0.006448                     0.002318                    0.9282                      0.7943                    0.4813                      0.7686         -0.001582           0.014929         -0.002690           0.008467                 0.957297                 1.752095
     2                21                    0.121533                        0.096095                          0.113324                   0.996982                       0.833812                         0.919325                     0.006018                     0.001834                    0.9906                      0.6841                    0.9727                      0.7419          0.000880           0.011409         -0.006060           0.010008                 0.727371                 2.442721

## Across-Phase Mean Results

Selector annualized return:
12.3132%

Equal Weight annualized return:
10.0181%

Maximum Sharpe annualized return:
11.5396%

Mean bootstrap probability selector return > Equal Weight:
95.34%

Mean bootstrap probability selector return > Maximum Sharpe:
70.68%

Mean bootstrap probability selector Sharpe > Equal Weight:
79.63%

Mean bootstrap probability selector Sharpe > Maximum Sharpe:
70.05%

## Interpretation

This artifact is not a production-approved investment strategy.

The historical sample is small and the three phase tests overlap in
calendar time. Phase robustness therefore should not be interpreted as
three independent experiments.

The purpose of freezing V1 is to stop historical optimization and begin
collecting genuinely unseen forward evidence.
