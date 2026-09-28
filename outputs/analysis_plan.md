# Pre-registered analysis plan: promotion experiment

## Hypothesis
The promotion increases average customer spend in the two weeks after send.

## Primary metric
Average spend per customer in the post period.

## Estimator
OLS of spend on the treatment indicator with heteroskedasticity-robust (HC1)
standard errors. Two-sided test at alpha = 0.05.

## Sample size
Sized for 80 percent power to detect a $12 lift per customer.

## Randomization check
Before looking at outcomes, compare pre-treatment covariates (past spend,
tenure, channel) across arms. Any standardized mean difference above 0.1
triggers an investigation of the assignment mechanism.

## Guardrails (must all pass before any launch decision)
1. Net margin per customer (spend margin minus promo cost) must increase.
2. Refund rate must not increase at the 5 percent significance level.

## Decision rule
Ship the promotion only if the primary effect is positive and significant
AND every guardrail passes.
