"""Randomized experiment analysis: checks, sizing, estimation, guardrails.

Randomization is the gold standard because it breaks the link between who
gets the promotion and how they would have behaved anyway. This module
covers the full workflow a data scientist runs around an A/B test: verify
the randomization worked, size the test before launch, estimate the effect
after, and confirm no guardrail metric moved the wrong way.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.power import NormalIndPower


def randomization_check(df, covariates, smd_threshold=0.1):
    """Balance table for the experiment: are the arms comparable?

    Returns one row per covariate with the standardized mean difference and
    a two-sample t-test p-value. Flags anything with |SMD| above threshold.
    """
    t = df[df["treated"] == 1]
    c = df[df["treated"] == 0]
    rows = []
    for col in covariates:
        a, b = t[col].to_numpy(), c[col].to_numpy()
        pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
        smd = (np.mean(a) - np.mean(b)) / pooled if pooled else 0.0
        p = stats.ttest_ind(a, b, equal_var=False).pvalue
        rows.append({"covariate": col,
                     "mean_treated": float(np.mean(a)),
                     "mean_control": float(np.mean(b)),
                     "smd": float(smd),
                     "p_value": float(p),
                     "flagged": bool(abs(smd) > smd_threshold)})
    return pd.DataFrame(rows)


def required_sample_size(std, effect, alpha=0.05, power=0.8):
    """Customers needed per arm to detect `effect` with the given power."""
    analysis = NormalIndPower()
    n = analysis.solve_power(effect_size=effect / std, alpha=alpha,
                             power=power, alternative="two-sided")
    return int(np.ceil(n))


def analyze_experiment(df):
    """Treatment effect on spend with heteroskedasticity-robust SEs."""
    model = smf.ols("spend ~ treated", data=df).fit(cov_type="HC1")
    est = model.params["treated"]
    se = model.bse["treated"]
    return {
        "estimate": float(est),
        "se": float(se),
        "ci_low": float(est - 1.96 * se),
        "ci_high": float(est + 1.96 * se),
        "p_value": float(model.pvalues["treated"]),
        "n_treated": int((df["treated"] == 1).sum()),
        "n_control": int((df["treated"] == 0).sum()),
    }


def guardrail_metrics(df, promo_cost_per_customer, margin_rate):
    """Business guardrails: did the promo hurt anything it should not have?

    Primary guardrail is net margin per customer (spend margin minus the promo
    cost). Secondary guardrail is the refund rate, which must not rise.
    """
    t = df[df["treated"] == 1]
    c = df[df["treated"] == 0]
    margin_diff = t["net_margin"].mean() - c["net_margin"].mean()
    refund_diff = t["refunded"].mean() - c["refunded"].mean()
    refund_p = stats.ttest_ind(t["refunded"], c["refunded"],
                               equal_var=False).pvalue
    return {
        "net_margin_lift_per_customer": float(margin_diff),
        "net_margin_guardrail_ok": bool(margin_diff > 0),
        "refund_rate_diff": float(refund_diff),
        "refund_rate_p_value": float(refund_p),
        "refund_guardrail_ok": bool(refund_p > 0.05 or refund_diff <= 0),
    }


def analysis_plan(effect_target=12.0, alpha=0.05):
    """Pre-registered analysis plan, written before seeing the results.

    In a real experiment this document is frozen before launch so the analyst
    cannot shop for a flattering specification after the fact.
    """
    return f"""# Pre-registered analysis plan: promotion experiment

## Hypothesis
The promotion increases average customer spend in the two weeks after send.

## Primary metric
Average spend per customer in the post period.

## Estimator
OLS of spend on the treatment indicator with heteroskedasticity-robust (HC1)
standard errors. Two-sided test at alpha = {alpha}.

## Sample size
Sized for 80 percent power to detect a ${effect_target:.0f} lift per customer.

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
"""
