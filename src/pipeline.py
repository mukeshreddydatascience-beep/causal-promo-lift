"""End-to-end pipeline: data -> three estimators -> validation vs ground truth.

Everything is driven by config.yaml. Run with `python run.py`.
"""

import logging
import os

import yaml

from src import abtest, did, data_generator, plots, psm

log = logging.getLogger(__name__)

COVARIATES = ["past_spend", "tenure", "channel"]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def naive_comparison(panel):
    """The number a dashboard would show: post-period treated mean minus
    control mean, with no adjustment for who got the promotion."""
    post = panel[panel["period"] == 0]
    t = post[post["treated"] == 1]["spend"]
    c = post[post["treated"] == 0]["spend"]
    diff = t.mean() - c.mean()
    se = float((t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c)) ** 0.5)
    return {"estimate": float(diff), "se": se,
            "ci_low": float(diff - 1.96 * se),
            "ci_high": float(diff + 1.96 * se),
            "n_treated": int(len(t)), "n_control": int(len(c))}


def recommend(net_value, guardrails):
    """Business recommendation from the causal evidence."""
    if not guardrails["net_margin_guardrail_ok"]:
        return ("KILL the blanket promotion. The causal estimate shows the "
                "promo destroys margin after its cost: the lift is real but "
                "too small to pay for the discount. Reshape before any "
                "relaunch: retarget to price-sensitive segments, test a "
                "smaller incentive, and re-run the experiment with the "
                "guardrails in this repo.")
    if not guardrails["refund_guardrail_ok"]:
        return ("HOLD. The promo lifts spend but the refund guardrail "
                "failed, so the net effect on the business is unclear. "
                "Investigate return reasons before scaling.")
    if net_value > 0:
        return ("SCALE. The promotion clears its cost with margin to spare "
                "and every guardrail passes. Roll out to the full base and "
                "keep the measurement framework running.")
    return ("REDESIGN. The effect is not reliably positive. Do not scale "
            "until a retargeted variant proves itself in a new experiment.")


def _fmt_est(name, result, key, true_value):
    err = result[key] - true_value
    return (f"{name:<28} estimate={result[key]:7.2f}  "
            f"95% CI=({result['ci_low']:.2f}, {result['ci_high']:.2f})  "
            f"error vs truth={err:+.2f}")


def run(config):
    out_dir = config.get("output_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    seed = config.get("seed", 42)

    # ---- Observational panel: promotion went to selected customers ----
    obs = config["observational"]
    log.info("Generating observational panel (n=%d, seed=%d)",
             obs["n_customers"], seed)
    panel, true_ate = data_generator.generate_panel(
        n_customers=obs["n_customers"], pre_periods=obs["pre_periods"],
        true_ate=obs["true_ate"], noise_sd=obs["noise_sd"], seed=seed)
    log.info("True average treatment effect: $%.2f", true_ate)

    log.info("Naive dashboard comparison (no adjustment)")
    naive = naive_comparison(panel)
    log.info(_fmt_est("Naive", naive, "estimate", true_ate))

    log.info("Running difference-in-differences")
    did_result = did.estimate_did(panel)
    log.info(_fmt_est("DiD", did_result, "estimate", true_ate))

    log.info("Running placebo pre-trend check")
    placebo = did.placebo_pretrend(panel)
    log.info("Placebo pre-trend estimate: %.3f (se=%.3f), expect near 0",
             placebo["estimate"], placebo["se"])

    trend_path = os.path.join(out_dir, "parallel_trends.png")
    did.plot_parallel_trends(panel, trend_path)
    log.info("Saved %s", trend_path)

    log.info("Running propensity-score matching")
    post = panel[panel["period"] == 0].copy()
    psm_result = psm.match_att(
        post, COVARIATES,
        caliper=config["matching"]["caliper"],
        with_replacement=config["matching"]["with_replacement"],
        seed=seed)
    log.info(_fmt_est("PSM (ATT)", psm_result, "att", true_ate))
    log.info("Matched %d of %d treated customers (%.1f%%)",
             psm_result["n_matched_pairs"], psm_result["n_treated"],
             100 * psm_result["match_rate"])
    for col in COVARIATES:
        log.info("Balance %s: SMD before=%+.3f after=%+.3f", col,
                 psm_result["balance_before"][col],
                 psm_result["balance_after"][col])
    overlap_path = os.path.join(out_dir, "propensity_overlap.png")
    psm.plot_score_overlap(psm_result, overlap_path)
    log.info("Saved %s", overlap_path)

    # ---- Randomized experiment: the A/B test arm ----
    exp = config["experiment"]
    log.info("Generating randomized experiment (n=%d, seed=%d)",
             exp["n_customers"], seed + 1)
    exp_df, exp_true = data_generator.generate_experiment(
        n_customers=exp["n_customers"], true_ate=exp["true_ate"],
        seed=seed + 1,
        promo_cost_per_customer=exp["promo_cost_per_customer"],
        margin_rate=exp["margin_rate"])

    log.info("Randomization check")
    balance = abtest.randomization_check(exp_df, COVARIATES)
    for _, row in balance.iterrows():
        flag = " FLAGGED" if row["flagged"] else ""
        log.info("  %s: SMD=%+.3f p=%.3f%s", row["covariate"], row["smd"],
                 row["p_value"], flag)

    needed = abtest.required_sample_size(
        exp_df["spend"].std(), exp["true_ate"],
        alpha=exp["alpha"], power=exp["target_power"])
    log.info("Sample size for 80%% power at $%.0f lift: %d per arm "
             "(have %d and %d)", exp["true_ate"], needed,
             (exp_df["treated"] == 1).sum(), (exp_df["treated"] == 0).sum())

    log.info("Analyzing experiment")
    ab_result = abtest.analyze_experiment(exp_df)
    log.info(_fmt_est("A/B test", ab_result, "estimate", exp_true))

    log.info("Guardrail metrics")
    guard = abtest.guardrail_metrics(
        exp_df, exp["promo_cost_per_customer"], exp["margin_rate"])
    log.info("  Net margin lift per customer: $%.2f (guardrail %s)",
             guard["net_margin_lift_per_customer"],
             "PASS" if guard["net_margin_guardrail_ok"] else "FAIL")
    log.info("  Refund rate diff: %+.4f p=%.3f (guardrail %s)",
             guard["refund_rate_diff"], guard["refund_rate_p_value"],
             "PASS" if guard["refund_guardrail_ok"] else "FAIL")

    plan = abtest.analysis_plan(effect_target=exp["true_ate"],
                                alpha=exp["alpha"])
    plan_path = os.path.join(out_dir, "analysis_plan.md")
    with open(plan_path, "w") as f:
        f.write(plan)
    log.info("Saved %s", plan_path)

    # ---- Business math: the naive claim vs what the promo truly did ----
    margin = exp["margin_rate"]
    cost = exp["promo_cost_per_customer"]
    n_t = naive["n_treated"]
    naive_margin_claim = naive["estimate"] * margin * n_t
    subsidy = (naive["estimate"] - did_result["estimate"]) * margin * n_t
    true_margin = did_result["estimate"] * margin * n_t
    promo_cost = cost * n_t
    net_value = true_margin - promo_cost

    log.info("Money answer: the naive read claims $%.0f of margin; $%.0f of "
             "that subsidized buyers who would have purchased anyway; true "
             "incremental margin $%.0f against promo cost $%.0f; net $%.0f",
             naive_margin_claim, subsidy, true_margin, promo_cost, net_value)

    comp_path = os.path.join(out_dir, "estimates_vs_truth.png")
    plots.plot_estimates_vs_truth(
        [("Naive comparison",
          naive["estimate"], naive["ci_low"], naive["ci_high"]),
         ("Difference-in-differences",
          did_result["estimate"], did_result["ci_low"], did_result["ci_high"]),
         ("Propensity matching",
          psm_result["att"], psm_result["ci_low"], psm_result["ci_high"]),
         ("Randomized A/B test",
          ab_result["estimate"], ab_result["ci_low"], ab_result["ci_high"])],
        true_ate, comp_path)
    log.info("Saved %s", comp_path)

    wf_path = os.path.join(out_dir, "promo_waterfall.png")
    plots.plot_promo_waterfall(
        "Naive margin claim", naive_margin_claim,
        [("Would-have-bought-anyway", -subsidy),
         ("Promo cost", -promo_cost)],
        "Net value", wf_path)
    log.info("Saved %s", wf_path)

    recommendation = recommend(net_value, guard)
    log.info("Recommendation: %s", recommendation)

    money = {"n_treated": n_t, "margin_rate": margin,
             "promo_cost_per_customer": cost,
             "naive_margin_claim": naive_margin_claim,
             "subsidy": subsidy, "true_margin": true_margin,
             "promo_cost": promo_cost, "net_value": net_value,
             "control_baseline": float(
                 panel[(panel["period"] == 0) &
                       (panel["treated"] == 0)]["spend"].mean())}

    _write_results(out_dir, true_ate, naive, did_result, placebo,
                   psm_result, ab_result, guard, needed, money,
                   recommendation)
    log.info("Pipeline complete. Results in %s/results.md", out_dir)
    return {
        "true_ate": true_ate,
        "naive": naive,
        "did": did_result,
        "placebo": placebo,
        "psm": psm_result,
        "ab": ab_result,
        "guardrails": guard,
    }


def _write_results(out_dir, true_ate, naive, did_result, placebo,
                   psm_result, ab_result, guard, needed_n, money,
                   recommendation):
    pct = lambda x: f"${x:,.0f}"
    base = money["control_baseline"]
    naive_lift_pct = 100 * naive["estimate"] / base
    true_lift_pct = 100 * did_result["estimate"] / base
    lines = [
        "# Results: did the promotion actually work?",
        "",
        "## Executive summary",
        "",
        "A retailer ran a promotion and the dashboard showed treated "
        f"customers spending ${naive['estimate']:.2f} more "
        f"(about +{naive_lift_pct:.0f}%). Three causal methods agree the "
        f"true lift is only about ${did_result['estimate']:.2f} "
        f"(about +{true_lift_pct:.0f}%). The gap, "
        f"{pct(money['subsidy'])} of margin across the treated cohort, "
        "went to shoppers who would have bought anyway. After the promo "
        f"cost, the campaign's net value is {pct(money['net_value'])}.",
        "",
        f"Recommendation: {recommendation}",
        "",
        "## The money answer",
        "",
        "All figures below come from synthetic data with a known treatment",
        f"effect of ${true_ate:.2f} per customer, so every method is checked",
        "against ground truth instead of against another estimate.",
        "",
        f"Naive margin claim: {pct(money['naive_margin_claim'])}.",
        f"Subsidy to would-have-bought-anyway buyers: "
        f"{pct(money['subsidy'])}.",
        f"True incremental margin: {pct(money['true_margin'])}.",
        f"Promo cost ({money['n_treated']:,} treated customers): "
        f"{pct(money['promo_cost'])}.",
        f"Net value: {pct(money['net_value'])}.",
        "",
        "![Promo economics waterfall](promo_waterfall.png)",
        "",
        "## Estimates vs ground truth",
        "",
        "| Method | Estimate | 95% CI | Error vs truth |",
        "| --- | --- | --- | --- |",
        f"| Naive comparison (no adjustment) | ${naive['estimate']:.2f} "
        f"| (${naive['ci_low']:.2f}, ${naive['ci_high']:.2f}) "
        f"| {naive['estimate'] - true_ate:+.2f} |",
        f"| Difference-in-differences | ${did_result['estimate']:.2f} "
        f"| (${did_result['ci_low']:.2f}, ${did_result['ci_high']:.2f}) "
        f"| {did_result['estimate'] - true_ate:+.2f} |",
        f"| Propensity-score matching (ATT) | ${psm_result['att']:.2f} "
        f"| (${psm_result['ci_low']:.2f}, ${psm_result['ci_high']:.2f}) "
        f"| {psm_result['att'] - true_ate:+.2f} |",
        f"| Randomized A/B test | ${ab_result['estimate']:.2f} "
        f"| (${ab_result['ci_low']:.2f}, ${ab_result['ci_high']:.2f}) "
        f"| {ab_result['estimate'] - true_ate:+.2f} |",
        "",
        "![Estimates vs truth](estimates_vs_truth.png)",
        "",
        "## Diagnostics",
        "",
        f"Placebo pre-trend check (should sit near zero): "
        f"{placebo['estimate']:.3f} (se {placebo['se']:.3f}).",
        f"Matching kept {psm_result['n_matched_pairs']} of "
        f"{psm_result['n_treated']} treated customers.",
        f"Sample size needed for 80% power: {needed_n} per arm.",
        "",
        "## Guardrails",
        "",
        f"Net margin lift per customer: "
        f"${guard['net_margin_lift_per_customer']:.2f} "
        f"({'PASS' if guard['net_margin_guardrail_ok'] else 'FAIL'}).",
        f"Refund rate change: {guard['refund_rate_diff']:+.4f}, "
        f"p={guard['refund_rate_p_value']:.3f} "
        f"({'PASS' if guard['refund_guardrail_ok'] else 'FAIL'}).",
        "",
        "## More plots",
        "",
        "![Parallel trends](parallel_trends.png)",
        "",
        "![Propensity overlap](propensity_overlap.png)",
        "",
    ]
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write("\n".join(lines))
