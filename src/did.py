"""Difference-in-differences estimation with parallel-trends diagnostics.

DiD is the right tool for the observational panel because the promotion was
not randomized: high spenders were more likely to receive it. DiD removes any
fixed differences between the groups and any shared time trend, isolating the
change that coincides with the promotion.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def estimate_did(panel):
    """Two-way DiD via OLS with the interaction term as the estimate.

    Standard errors are clustered by customer because each customer is
    observed in multiple periods.
    """
    df = panel.copy()
    df["post"] = (df["period"] == 0).astype(int)
    model = smf.ols("spend ~ treated * post", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["customer_id"]})
    term = "treated:post"
    est = model.params[term]
    se = model.bse[term]
    return {
        "estimate": float(est),
        "se": float(se),
        "ci_low": float(est - 1.96 * se),
        "ci_high": float(est + 1.96 * se),
        "p_value": float(model.pvalues[term]),
    }


def placebo_pretrend(panel):
    """Placebo DiD on pre-periods only: the last pre period plays "post".

    If parallel trends hold, this estimate should sit near zero. A large
    placebo estimate would warn that the groups were already diverging before
    the promotion, which would undermine the real DiD result.
    """
    pre = panel[panel["period"] < 0].copy()
    last_pre = pre["period"].max()
    pre["post"] = (pre["period"] == last_pre).astype(int)
    model = smf.ols("spend ~ treated * post", data=pre).fit(
        cov_type="cluster", cov_kwds={"groups": pre["customer_id"]})
    term = "treated:post"
    return {"estimate": float(model.params[term]),
            "se": float(model.bse[term])}


def plot_parallel_trends(panel, path):
    """Group means per period with 95 percent confidence bands."""
    grouped = (panel.groupby(["period", "treated"])["spend"]
               .agg(["mean", "std", "count"]).reset_index())
    grouped["se"] = grouped["std"] / np.sqrt(grouped["count"])

    fig, ax = plt.subplots(figsize=(8, 5))
    for treated, label in [(0, "Control"), (1, "Treated (promo)")]:
        sub = grouped[grouped["treated"] == treated].sort_values("period")
        ax.errorbar(sub["period"], sub["mean"], yerr=1.96 * sub["se"],
                    marker="o", capsize=4, label=label)
    ax.axvline(-0.5, color="gray", linestyle="--", linewidth=1)
    ax.text(-0.45, ax.get_ylim()[1] * 0.97, "promotion starts",
            fontsize=9, color="gray")
    ax.set_xlabel("Period (0 is the promotion period)")
    ax.set_ylabel("Average spend ($)")
    ax.set_title("Parallel trends check: average spend by group and period")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
