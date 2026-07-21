"""
Shared plotting — every experiment's results are visualized the same way.
Headless (Agg backend), matching embeddings.explore's convention: this
runs on the GPU server, not a desktop with a display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import AggregatedMetric


def plot_metric_comparison(
    aggregated_results: Dict[str, Dict[str, AggregatedMetric]],
    metric_name: str,
    output_path: Path,
    confidence_level: float = 0.95,
) -> None:
    """One bar per model, for a single metric, with confidence-interval
    error bars — the central figure for comparing models (e.g. E1's
    classifier vs. Persistence vs. Identity Dynamics vs. Linear SSM)."""
    model_names = [m for m in aggregated_results if metric_name in aggregated_results[m]]
    if not model_names:
        return
    means = [aggregated_results[m][metric_name].mean for m in model_names]
    ci_lows = [aggregated_results[m][metric_name].ci_low for m in model_names]
    ci_highs = [aggregated_results[m][metric_name].ci_high for m in model_names]
    errors = [
        [m - lo for m, lo in zip(means, ci_lows)],
        [hi - m for m, hi in zip(means, ci_highs)],
    ]

    fig, ax = plt.subplots(figsize=(max(4, 1.2 * len(model_names)), 4))
    ax.bar(model_names, means, yerr=errors, capsize=4)
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} — mean and {int(confidence_level * 100)}% CI across seeds")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_seed_distribution(per_seed_metrics: Dict[int, Dict[str, float]], metric_name: str, output_path: Path) -> None:
    """Every individual seed's value for one metric, one model — makes the
    actual spread across seeds visible, not just its summary. Especially
    important at n=3: the summary bar alone can hide that one seed was an
    outlier."""
    seeds = sorted(per_seed_metrics.keys())
    values = [per_seed_metrics[s].get(metric_name) for s in seeds]
    seeds = [s for s, v in zip(seeds, values) if v is not None and not (isinstance(v, float) and np.isnan(v))]
    values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not values:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(seeds, values)
    ax.axhline(np.mean(values), linestyle="--", color="gray", linewidth=1, label="mean")
    ax.set_xlabel("seed")
    ax.set_ylabel(metric_name)
    ax.set_xticks(seeds)
    ax.set_title(f"{metric_name} across {len(seeds)} seeds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_paired_difference_distribution(
    diff_values: np.ndarray, comparison_name: str, metric_name: str, output_path: Path
) -> None:
    """
    Histogram of a paired bootstrap difference distribution (model B -
    model A, one ladder comparison, one metric), with a vertical line at
    zero. Whether the bulk of the distribution — and specifically the
    percentile CI reported alongside it in the comparison table — sits on
    one side of zero is the direct visual counterpart of
    metrics.paired_bootstrap_comparison's CI and Wilcoxon test.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(diff_values, bins=40, color="steelblue", alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--", label="no difference")
    ax.axvline(float(np.mean(diff_values)), color="firebrick", linewidth=1.5, label="observed mean difference")
    ax.set_xlabel(f"{metric_name} difference ({comparison_name})")
    ax.set_ylabel("bootstrap resamples")
    ax.set_title(f"Paired bootstrap difference — {comparison_name} — {metric_name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
