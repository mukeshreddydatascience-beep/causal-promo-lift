"""Estimator tests against the known ground-truth treatment effect.

Because the data generator fixes the true effect in advance, these tests
assert that each method recovers it within a tolerance instead of merely
checking that the code runs.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import abtest, data_generator, did, psm
from src.pipeline import load_config, naive_comparison


def test_naive_comparison_is_biased_upward():
    # Treatment went to bigger historical spenders, so the raw
    # treated-minus-control gap must overstate the true effect.
    panel, true_ate = data_generator.generate_panel(
        n_customers=4000, pre_periods=3, true_ate=TRUE_ATE,
        noise_sd=6.0, seed=7)
    naive = naive_comparison(panel)
    assert naive["estimate"] > true_ate + 3.0

TRUE_ATE = 12.0
COVARIATES = ["past_spend", "tenure", "channel"]


def test_did_recovers_true_effect():
    panel, true_ate = data_generator.generate_panel(
        n_customers=8000, pre_periods=3, true_ate=TRUE_ATE,
        noise_sd=6.0, seed=7)
    result = did.estimate_did(panel)
    assert abs(result["estimate"] - true_ate) < 1.5
    assert result["ci_low"] <= true_ate <= result["ci_high"]


def test_placebo_pretrend_is_near_zero():
    panel, _ = data_generator.generate_panel(
        n_customers=4000, pre_periods=3, true_ate=TRUE_ATE,
        noise_sd=6.0, seed=7)
    placebo = did.placebo_pretrend(panel)
    assert abs(placebo["estimate"]) < 1.0


def test_psm_recovers_true_effect():
    panel, true_ate = data_generator.generate_panel(
        n_customers=8000, pre_periods=3, true_ate=TRUE_ATE,
        noise_sd=6.0, seed=11)
    post = panel[panel["period"] == 0]
    result = psm.match_att(post, COVARIATES, caliper=0.05,
                           with_replacement=True, seed=11)
    assert result["match_rate"] > 0.9
    assert abs(result["att"] - true_ate) < 2.5
    for col in COVARIATES:
        assert abs(result["balance_after"][col]) < 0.1


def test_ab_experiment_recovers_true_effect():
    exp_df, true_ate = data_generator.generate_experiment(
        n_customers=3000, true_ate=TRUE_ATE, seed=21)
    result = abtest.analyze_experiment(exp_df)
    assert abs(result["estimate"] - true_ate) < 2.0
    assert result["ci_low"] <= true_ate <= result["ci_high"]


def test_randomization_check_passes_on_random_assignment():
    exp_df, _ = data_generator.generate_experiment(
        n_customers=3000, true_ate=TRUE_ATE, seed=21)
    balance = abtest.randomization_check(exp_df, COVARIATES)
    assert not balance["flagged"].any()


def test_guardrail_catches_money_losing_promo():
    # True lift ($12) at 35% margin nets $4.20 against an $8 promo cost,
    # so the guardrail must flag this promotion as value-destroying.
    exp_df, _ = data_generator.generate_experiment(
        n_customers=3000, true_ate=TRUE_ATE, seed=21,
        promo_cost_per_customer=8.0, margin_rate=0.35)
    guard = abtest.guardrail_metrics(exp_df, 8.0, 0.35)
    assert not guard["net_margin_guardrail_ok"]
    assert guard["net_margin_lift_per_customer"] < 0
    assert guard["refund_guardrail_ok"]


def test_config_loads_with_expected_keys():
    config = load_config(os.path.join(os.path.dirname(__file__),
                                      "..", "config.yaml"))
    for key in ("seed", "observational", "matching", "experiment"):
        assert key in config
