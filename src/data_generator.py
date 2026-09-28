"""Synthetic customer panel data with a KNOWN treatment effect.

Everything here is simulated. No real customer or company data is used.
The data generating process is built so the true average treatment effect
is a fixed, known number. That lets every estimator in this project be
validated against ground truth instead of against another estimate.
"""

import numpy as np
import pandas as pd

CHANNELS = {0: "online", 1: "store", 2: "mixed"}


def _baseline_spend(past_spend, tenure, channel):
    """Pre-promotion spend level implied by customer confounders."""
    channel_lift = np.where(channel == 0, 3.0, np.where(channel == 1, 0.0, -2.0))
    return 18.0 + 0.55 * past_spend + 0.08 * tenure + channel_lift


def generate_panel(n_customers=20000, pre_periods=3, true_ate=12.0,
                   noise_sd=6.0, seed=42):
    """Customer panel with a confounded (non-random) promotion.

    Treatment assignment depends on past spend, tenure, and channel, so a
    naive treated-vs-control comparison is biased. The promotion effect only
    lands in the post period for treated customers, and the pre-period trend
    is identical for both groups, so the parallel-trends assumption holds.

    Returns a long-format DataFrame (one row per customer per period) and the
    true average treatment effect.
    """
    rng = np.random.default_rng(seed)
    past_spend = rng.lognormal(mean=3.4, sigma=0.6, size=n_customers)
    tenure = rng.integers(1, 61, size=n_customers).astype(float)
    channel = rng.choice([0, 1, 2], size=n_customers, p=[0.5, 0.3, 0.2])

    # Confounded assignment: bigger historical spenders are more likely treated.
    z_past = (np.log(past_spend) - 3.4) / 0.6
    z_tenure = (tenure - 30.0) / 17.0
    logit = -1.0 + 0.55 * z_past + 0.35 * z_tenure + 0.25 * (channel == 0)
    treated = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))

    base = _baseline_spend(past_spend, tenure, channel)
    periods = list(range(-pre_periods, 1))  # e.g. -3, -2, -1, 0

    frames = []
    for t in periods:
        period_effect = 1.5 * t  # common trend, same for both groups
        noise = rng.normal(0.0, noise_sd, size=n_customers)
        effect = np.where((treated == 1) & (t == 0), true_ate, 0.0)
        spend = np.maximum(base + period_effect + effect + noise, 0.0)
        frames.append(pd.DataFrame({
            "customer_id": np.arange(n_customers),
            "period": t,
            "treated": treated,
            "spend": spend,
            "past_spend": past_spend,
            "tenure": tenure,
            "channel": channel,
        }))
    panel = pd.concat(frames, ignore_index=True)
    return panel, float(true_ate)


def generate_experiment(n_customers=6000, true_ate=12.0, seed=123,
                        promo_cost_per_customer=8.0, margin_rate=0.35):
    """Randomized promotion experiment (the A/B test arm).

    Assignment is a fair coin flip, so confounders are balanced in
    expectation. Each customer also gets a net-margin figure and a refund
    flag, which feed the guardrail metrics.
    """
    rng = np.random.default_rng(seed)
    past_spend = rng.lognormal(mean=3.4, sigma=0.6, size=n_customers)
    tenure = rng.integers(1, 61, size=n_customers).astype(float)
    channel = rng.choice([0, 1, 2], size=n_customers, p=[0.5, 0.3, 0.2])
    treated = rng.binomial(1, 0.5, size=n_customers)

    base = _baseline_spend(past_spend, tenure, channel)
    spend = np.maximum(base + treated * true_ate + rng.normal(0, 6.0, n_customers), 0.0)
    net_margin = spend * margin_rate - treated * promo_cost_per_customer
    refund_prob = 0.06 + 0.0004 * (spend - spend.mean())
    refunded = rng.binomial(1, np.clip(refund_prob, 0.01, 0.2))

    return pd.DataFrame({
        "customer_id": np.arange(n_customers),
        "treated": treated,
        "spend": spend,
        "net_margin": net_margin,
        "refunded": refunded,
        "past_spend": past_spend,
        "tenure": tenure,
        "channel": channel,
    }), float(true_ate)
