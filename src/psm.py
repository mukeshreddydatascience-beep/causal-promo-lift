"""Propensity-score matching for the observational panel.

Matching attacks the same selection problem as DiD but from a different
angle: instead of differencing out fixed group differences over time, it
rebuilds a comparable control group by pairing each treated customer with a
lookalike who did not get the promotion. Agreement between the two methods
is the real credibility check in this project.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _smd(a, b):
    """Standardized mean difference between two samples."""
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    if pooled == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled


def balance_table(df, covariates, treated_col="treated"):
    """Standardized mean differences per covariate, treated vs control."""
    t = df[df[treated_col] == 1]
    c = df[df[treated_col] == 0]
    return {col: _smd(t[col].to_numpy(), c[col].to_numpy()) for col in covariates}


def match_att(df_post, covariates, caliper=0.05, with_replacement=True,
              seed=42):
    """Nearest-neighbor propensity matching with an ATT estimate.

    Steps: fit a logistic propensity model on the confounders, then match
    each treated customer to the closest control within the caliper. The ATT
    is the mean treated outcome minus the mean matched-control outcome.
    """
    rng = np.random.default_rng(seed)
    X = df_post[covariates].to_numpy()
    y = df_post["treated"].to_numpy()
    model = LogisticRegression(max_iter=1000)
    scores = model.fit(X, y).predict_proba(X)[:, 1]

    df = df_post.reset_index(drop=True).copy()
    df["propensity"] = scores
    treated_pos = np.flatnonzero(df["treated"].to_numpy() == 1)
    control_pos = np.flatnonzero(df["treated"].to_numpy() == 0)
    treated = df.iloc[treated_pos].reset_index(drop=True)
    control_scores = df["propensity"].to_numpy()[control_pos]

    used = np.zeros(len(control_pos), dtype=bool)
    pairs = []
    matched_control_rows = []
    order = rng.permutation(len(treated))
    for i in order:
        dist = np.abs(control_scores - treated.loc[i, "propensity"])
        if not with_replacement:
            dist = np.where(used, np.inf, dist)
        j = int(np.argmin(dist))
        if dist[j] <= caliper:
            pairs.append((treated.loc[i, "spend"],
                          df["spend"].to_numpy()[control_pos[j]]))
            matched_control_rows.append(control_pos[j])
            used[j] = True

    if not pairs:
        raise ValueError("No matches found inside the caliper; widen it.")
    pairs = np.array(pairs)
    diffs = pairs[:, 0] - pairs[:, 1]
    att = float(diffs.mean())
    se = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))

    balance_before = balance_table(df, covariates)
    matched_controls = df.loc[matched_control_rows]
    balance_after = {
        col: _smd(treated[col].to_numpy(), matched_controls[col].to_numpy())
        for col in covariates
    }
    return {
        "att": att,
        "se": se,
        "ci_low": att - 1.96 * se,
        "ci_high": att + 1.96 * se,
        "n_matched_pairs": len(pairs),
        "n_treated": len(treated),
        "match_rate": len(pairs) / len(treated),
        "balance_before": balance_before,
        "balance_after": balance_after,
        "propensity_scores": df[["treated", "propensity"]],
    }


def plot_score_overlap(match_result, path):
    """Overlaid propensity-score histograms for treated and control."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = match_result["propensity_scores"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores[scores["treated"] == 1]["propensity"], bins=40,
            alpha=0.6, label="Treated")
    ax.hist(scores[scores["treated"] == 0]["propensity"], bins=40,
            alpha=0.6, label="Control")
    ax.set_xlabel("Estimated propensity score")
    ax.set_ylabel("Customers")
    ax.set_title("Propensity score overlap: treated vs control")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
