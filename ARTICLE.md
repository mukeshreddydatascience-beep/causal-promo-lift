# Your Promotion Didn't Work. Here's How I Proved It with Causal Inference

*The dashboard said +$18.65 per customer. Three causal methods said ~$12. After promo cost, the campaign lost $23,675.*

A retail promotion can lift sales and still destroy value. That is what I found when I built a causal inference pipeline to measure promotional lift rather than accept attribution credit. The dashboard reported $18.65 of extra revenue per customer. Difference-in-differences, propensity score matching, and A/B testing converged on a treatment effect of about $12. Once I included the $8 per customer promotion cost, the campaign's net value was negative $23,675.

## Why Your Promotion Dashboard Is Lying to You

Most promotion dashboards answer a convenient question: how much did promoted customers spend compared with everyone else?

That is not the same as asking what the promotion caused.

Customers selected for a campaign are rarely interchangeable with everyone else. A retailer may target loyal customers, high spenders, or recent browsers who already have a higher probability of purchasing. If they receive a coupon and buy, the dashboard gives the coupon credit for the entire difference.

The same mistake appears in advertising. In a well-known [eBay field experiment](https://newsroom.haas.berkeley.edu/study-finds-paid-search-ads-dont-always-pay/), researchers turned off paid search advertising in selected markets and found no measurable sales increase from paid search relative to unpaid channels. Much of the apparent value came from existing intent. People who were already looking for eBay simply reached it another way.

My project recreates that management problem in retail promotions. The treated group contains stronger historical buyers, so a naive comparison looks excellent. It reports $18.65 of additional revenue per customer. But part of that number belongs to customer selection, not promotion effectiveness.

This is the would-have-bought-anyway problem. It is also where promotional budgets disappear.

## What Incrementality Actually Measures

Incrementality testing measures the outcome caused by an intervention above what would have happened without it.

For each promoted customer, there are two possible outcomes. One is the purchase behavior after receiving the offer. The other is the behavior that same person would have shown without it. We can observe only one of those outcomes. Causal inference gives us disciplined ways to estimate the missing counterfactual.

Attribution asks which campaign touched a sale. Incrementality asks whether the sale would have happened without that campaign.

A campaign can receive attribution credit for thousands of orders while producing little causal lift. It can also produce a positive treatment effect and still lose money after discount, media, fulfillment, and refund costs.

The analysis needs two answers. Did the promotion change behavior, and was that change worth more than it cost?

## The Setup: A Promotion With a Known True Effect

I generated a synthetic customer panel with a seeded, reproducible treatment effect of exactly $12 per customer. Promo assignment is intentionally confounded. Customers with stronger prior spending patterns are more likely to receive the offer.

This creates the kind of data problem an analyst sees in production while preserving something real company data never gives us: known ground truth.

Most incrementality examples fit an estimator and stop. They cannot prove the estimate is right because the true counterfactual is unknown. Here, every method must recover the seeded $12 effect within a defined tolerance. If a persuasive estimate misses the truth, the tests fail.

The naive comparison overshoots by $6.65 per customer. Difference-in-differences estimates $11.96, propensity score matching estimates $12.41, and the randomized A/B analysis estimates $12.56. Different assumptions lead to the same practical conclusion.

One estimate can be a model artifact. Three identification strategies converging near the known treatment effect provide stronger evidence that the pipeline works.

## Method 1: Difference-in-Differences

Difference-in-differences compares change over time, not just final levels.

I calculate the pre-to-post change for promoted customers and subtract the same change for the control group. Stable differences between groups are removed, along with time shocks shared by both groups. What remains is the change associated with the promotion.

The core regression is intentionally readable:

```python
model = smf.ols(
    "revenue ~ treated + post + treated:post",
    data=panel,
).fit(
    cov_type="cluster",
    cov_kwds={"groups": panel["customer_id"]},
)

promo_lift = model.params["treated:post"]
```

I cluster standard errors by customer because repeated observations from the same customer are not independent. Ignoring that structure can make uncertainty look smaller than it is.

The key assumption is parallel trends. Without the promotion, treated and control customers should have followed similar revenue trends. The pipeline produces a parallel-trends plot and runs a placebo test on pre-promotion periods. A meaningful placebo effect would show that the groups were already moving differently.

The estimate lands at $11.96 per customer, almost exactly on the seeded $12 truth. More importantly, the diagnostics make clear why I am willing to trust it.

## Method 2: Propensity Score Matching

Propensity score matching approaches the selection problem at the customer level.

I fit a logistic model that estimates each customer's probability of receiving the promotion from pre-treatment characteristics. I then pair each treated customer with a similar untreated customer using nearest-neighbor matching and a caliper. The caliper blocks weak matches rather than forcing every treated customer onto an unsuitable control.

The important output is not the propensity model's accuracy. It is covariate balance after matching.

Before matching, standardized mean differences are around 0.44, a clear sign that treated and control customers are not comparable. After matching, they fall to roughly 0.02. The overlap plot shows where both groups have credible support, while the balance diagnostics show that the reconstructed control group now resembles the promoted group on observed pre-treatment variables.

On that matched sample, the estimated promotional lift is $12.41 per customer.

This method is useful when assignment was not randomized and strong pre-treatment data is available. It cannot solve hidden confounding because no matching algorithm can balance an unmeasured variable. That is why I compare it with difference-in-differences and a randomized benchmark.

## Method 3: A/B Testing Done Right

A randomized experiment is the cleanest way to estimate promotion effectiveness, but randomization alone does not make an analysis reliable.

I start with a randomization check on pre-treatment covariates. If treatment and control are materially imbalanced before the campaign, I investigate before reading the outcome. I also calculate statistical power and required sample size before the test so a noisy result is not mistaken for evidence of no effect.

The analysis plan is pre-registered. The primary outcome, estimator, guardrails, sample rules, and decision criteria are written before results are examined. This reduces the temptation to keep slicing the data until something looks successful.

Revenue is not the only metric. The test tracks contribution margin and refund rate as guardrails. In this run, refunds remain stable, but net margin fails. The experiment estimates a $12.56 revenue lift, which is directionally good and close to the known $12 effect. The economics still reject the campaign.

Statistical significance does not approve a rollout. The decision depends on value after costs and whether customer and operational guardrails remain healthy.

## The Money Math

The dashboard's claim and the causal answer lead to opposite decisions.

| Step | Per customer | Campaign impact |
|---|---:|---:|
| Naive revenue lift | $18.65 | Looks scalable |
| Causal revenue lift | ~$12.00 | Real, but smaller |
| Promo cost | $8.00 | Margin pressure |
| Net value | Negative | -$23,675 |

The gap between the naive claim and the causal estimate represents customers who were likely to buy anyway. Across the treated cohort, that gap translates into $14,543 of margin attributed to the promotion even though the promotion did not create it.

The campaign did generate incremental revenue. That alone was not enough. Once the $8 per customer promotion cost was applied, the blanket campaign destroyed $23,675 of value.

This is the decision line I want a manager to see. The campaign did not fail because the treatment effect was zero. It failed because the true effect could not cover the economics of delivering the offer at scale.

## What the Business Should Do

I would kill the blanket promotion.

The average treatment effect does not say promotions never work. It says this offer, sent to this broad audience at this cost, should not scale.

The next move is to estimate heterogeneous treatment effects or build uplift segments. High-probability buyers with low incremental response should receive no discount. Customers whose behavior may change can be tested with a smaller incentive. New, infrequent, price-sensitive, and category-specific segments deserve separate analysis.

Then I would run a new randomized test with the same discipline. Power it before launch. Pre-register the plan. Keep margin and refunds beside revenue. Use a holdout that remains untouched long enough to measure durable behavior, not only an immediate conversion spike.

The operating recommendation is simple: stop paying everyone, target the customers who can be moved, and retest the economics.

## Run It Yourself

The complete project is public in my [causal-promo-lift GitHub repository](https://github.com/mukeshreddydatascience-beep/causal-promo-lift).

Install the dependencies with `pip install -r requirements.txt`, then run the full pipeline with `python run.py`.

The workflow generates the synthetic panel, runs all three estimators, creates the diagnostic plots, calculates campaign economics, and writes the recommendation. Parameters such as sample size, random seed, true treatment effect, matching settings, and promotion cost live in YAML configuration.

The implementation uses pandas, NumPy, scikit-learn, statsmodels, and matplotlib. Pytest checks every estimator against the known $12 ground truth, verifies that the placebo trend remains near zero, confirms the naive estimate is biased upward, and confirms the margin guardrail rejects the campaign. GitHub Actions runs the tests and the full pipeline on every push.

## FAQ

### What is incrementality testing?

Incrementality testing estimates how much additional behavior an intervention caused compared with what would have happened without it. In promotion analysis, it separates true promotional lift from purchases that would have occurred anyway.

### How is incrementality different from attribution?

Attribution assigns credit to a customer touchpoint. Incrementality estimates causation. A coupon can be attributed to an order even when the customer had already decided to buy. Only the incremental portion should inform promotion effectiveness and budget decisions.

### When should I use difference-in-differences vs propensity matching?

Use difference-in-differences when you have repeated pre-treatment and post-treatment outcomes and a credible parallel-trends assumption. Use propensity score matching when treatment assignment depends on observed customer characteristics and you have enough overlap to construct a comparable control group. When possible, use both as sensitivity checks because they fail under different assumptions.

### Do I need a randomized experiment?

Not always. Strong observational designs can produce useful causal estimates when randomized testing is unavailable. A/B testing remains the best option when it is feasible, ethical, and properly powered. Even then, randomization checks, pre-registration, and financial guardrails are necessary.

---

*Causal Inference · Data Science · Experimentation · Marketing Analytics · A/B Testing*

Mukesh Reddy is a Senior Data Scientist working on causal inference, experimentation, and measurement.
