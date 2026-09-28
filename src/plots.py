"""Presentation plots: estimate comparison and promo-economics waterfall."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_estimates_vs_truth(estimates, true_value, path):
    """Bar chart of every method's estimate with 95% CI whiskers.

    estimates: list of (label, point, ci_low, ci_high).
    """
    labels = [e[0] for e in estimates]
    points = [e[1] for e in estimates]
    lowers = [e[1] - e[2] for e in estimates]
    uppers = [e[3] - e[1] for e in estimates]
    colors = ["#c0392b" if l == "Naive comparison" else "#2c3e50"
              for l in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    ax.bar(x, points, yerr=[lowers, uppers], capsize=6, color=colors,
           edgecolor="black", linewidth=0.8)
    ax.axhline(true_value, color="#27ae60", linestyle="--", linewidth=1.5,
               label=f"True effect (${true_value:.0f})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Estimated lift per customer ($)")
    ax.set_title("Every method vs the known true effect")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_promo_waterfall(start_label, start_value, changes, end_label, path):
    """Waterfall from a starting total through signed changes to an end total.

    changes: list of (label, signed dollar change).
    """
    labels = [start_label] + [c[0] for c in changes] + [end_label]
    deltas = [c[1] for c in changes]
    totals = [start_value]
    for d in deltas:
        totals.append(totals[-1] + d)
    end = totals[-1]
    pad = abs(start_value) * 0.04

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(0, start_value, color="#2c3e50", edgecolor="black",
           linewidth=0.8, width=0.6)
    ax.text(0, start_value + pad, f"${start_value:,.0f}",
            ha="center", fontsize=9)
    for i, (label, d) in enumerate(changes, start=1):
        bottom = totals[i - 1] + min(0, d)
        color = "#27ae60" if d >= 0 else "#c0392b"
        ax.bar(i, abs(d), bottom=bottom, color=color, edgecolor="black",
               linewidth=0.8, width=0.6)
        ax.plot([i - 0.3, i + 0.3], [totals[i - 1], totals[i - 1]],
                color="gray", linewidth=1)
        ax.text(i, bottom + abs(d) + pad, f"${d:+,.0f}",
                ha="center", fontsize=9)
    last = len(labels) - 1
    ax.bar(last, abs(end), bottom=min(0, end), color="#2c3e50",
           edgecolor="black", linewidth=0.8, width=0.6)
    end_y = end - pad if end < 0 else end + pad
    ax.text(last, end_y, f"${end:,.0f}", ha="center", fontsize=9)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=14, ha="right")
    ax.set_ylabel("Dollars (treated cohort)")
    ax.set_title("From the naive margin claim to the net value")
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
