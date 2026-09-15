"""
The complete E1 analysis: fits all four registered models once each
(deterministic — see the E1 experiment design's sanity-checks section),
evaluates on the test split via patient-level bootstrap, computes the
three PRE-REGISTERED ablation-ladder comparisons (identity_dynamics vs
base_rate, persistence vs identity_dynamics, linear_ssm vs persistence)
with confidence intervals, effect sizes, and a Holm-Bonferroni-corrected
Wilcoxon test, for the two primary metrics (accuracy, auroc) — and
generates the comparison table, pairwise-comparison table, and plots.

This script does NOT interpret its own output — see the accompanying
execution checklist for what to do with the numbers once they exist.

Usage
-----
    cd Training
    python -m experiments.e1_ladder.run_e1_full_analysis \\
        --cache_root ../Embeddings --embedding_model_name resnet18 \\
        --n_bootstrap 1000 --bootstrap_seed 0 \\
        --output_dir ../Results/evaluation/e1_full_analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from embeddings.dataset import EmbeddingDataset

from evaluation.metrics import (
    bootstrap_trajectory_metrics,
    holm_bonferroni,
    paired_bootstrap_comparison,
)
from evaluation.model import get_model_class
from evaluation.plots import plot_metric_comparison, plot_paired_difference_distribution
from evaluation.trajectory import group_into_trajectories

# Imported for @register_model side effects.
import evaluation.models.base_rate  # noqa: F401
import evaluation.models.identity_dynamics  # noqa: F401
import evaluation.models.linear_ssm  # noqa: F401
import evaluation.models.persistence  # noqa: F401

# The full ablation ladder, and the PRE-REGISTERED adjacent comparisons —
# decided here, before any result exists, not selected after seeing
# which comparisons look interesting. See holm_bonferroni's docstring for
# why this is exactly three comparisons, not all six possible pairs.
MODEL_ORDER = ["base_rate", "identity_dynamics", "persistence", "linear_ssm"]
LADDER_COMPARISONS = [
    ("identity_dynamics", "base_rate"),
    ("persistence", "identity_dynamics"),
    ("linear_ssm", "persistence"),
]
PRIMARY_METRICS = ["accuracy", "auroc"]

MODEL_KWARGS: Dict[str, dict] = {
    "base_rate": {},
    "identity_dynamics": {},
    "persistence": {},
    "linear_ssm": {"latent_dim": 16, "ridge_alpha": 1.0},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_root", default="../Embeddings")
    p.add_argument("--embedding_model_name", default="resnet18", choices=["resnet18", "timesformer"])
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--bootstrap_seed", type=int, default=0)
    p.add_argument("--confidence_level", type=float, default=0.95)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--output_dir", default="../Results/evaluation/e1_full_analysis")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_root = Path(args.cache_root) / args.embedding_model_name
    train_trajectories = group_into_trajectories(EmbeddingDataset(cache_root, "train"))
    val_trajectories = group_into_trajectories(EmbeddingDataset(cache_root, "val"))
    test_trajectories = group_into_trajectories(EmbeddingDataset(cache_root, "test"))
    print(f"Loaded {len(train_trajectories)} train / {len(val_trajectories)} val / {len(test_trajectories)} test trajectories.")

    # --- fit each model once; fitting is deterministic given fixed data
    # for all four models (see sanity checks). None of the four current
    # models actually use `val` for anything (no early stopping or
    # hyperparameter selection is implemented yet — see the E1 experiment
    # design's "datasets" section for why that's a deliberate, documented
    # scope decision, not an oversight) — it is still passed correctly
    # rather than passed as train data, so nothing silently breaks the
    # day a future model does use it. ---
    fitted_models = {}
    for name in MODEL_ORDER:
        model = get_model_class(name)(**MODEL_KWARGS[name])
        model.fit(train_trajectories, val_trajectories)
        fitted_models[name] = model
        print(f"Fitted {name}.")

    # --- patient-level bootstrap on the test set, SAME bootstrap_seed for
    # every model so resamples are paired across models ---
    aggregated_by_model = {}
    raw_by_model = {}
    for name, model in fitted_models.items():
        aggregated, raw = bootstrap_trajectory_metrics(
            model, test_trajectories, n_bootstrap=args.n_bootstrap,
            confidence_level=args.confidence_level, rng_seed=args.bootstrap_seed,
        )
        aggregated_by_model[name] = aggregated
        raw_by_model[name] = raw
        print(f"Bootstrapped {name}: accuracy = {aggregated['accuracy'].mean:.4f} "
              f"[{aggregated['accuracy'].ci_low:.4f}, {aggregated['accuracy'].ci_high:.4f}]")

    # --- main comparison plots (one per primary metric, all 4 models) ---
    for metric in PRIMARY_METRICS:
        plot_metric_comparison(aggregated_by_model, metric, output_dir / f"comparison_{metric}.png", args.confidence_level)

    # --- pre-registered pairwise comparisons ---
    comparison_results: Dict[str, Dict[str, dict]] = {}
    p_values_by_metric: Dict[str, Dict[str, float]] = {m: {} for m in PRIMARY_METRICS}

    for later, earlier in LADDER_COMPARISONS:
        comparison_name = f"{later}_vs_{earlier}"
        comparison_results[comparison_name] = {}
        for metric in PRIMARY_METRICS:
            result = paired_bootstrap_comparison(raw_by_model[earlier], raw_by_model[later], metric, args.confidence_level)
            comparison_results[comparison_name][metric] = result
            p_values_by_metric[metric][comparison_name] = result["wilcoxon_p"]

            plot_paired_difference_distribution(
                raw_by_model[later][metric] - raw_by_model[earlier][metric],
                comparison_name, metric,
                output_dir / f"diff_{comparison_name}_{metric}.png",
            )

    # --- Holm-Bonferroni correction, applied per metric across the three
    # pre-registered comparisons ---
    significance_by_metric = {m: holm_bonferroni(p_values_by_metric[m], alpha=args.alpha) for m in PRIMARY_METRICS}
    for metric in PRIMARY_METRICS:
        for comparison_name in significance_by_metric[metric]:
            comparison_results[comparison_name][metric]["significant_after_correction"] = significance_by_metric[metric][comparison_name]

    # --- write raw results ---
    (output_dir / "aggregated_results.json").write_text(
        json.dumps(
            {m: {k: v.__dict__ for k, v in metrics.items()} for m, metrics in aggregated_by_model.items()},
            indent=2, default=list,
        )
    )
    (output_dir / "comparison_results.json").write_text(json.dumps(comparison_results, indent=2))

    # --- main results table ---
    lines = ["# E1 Full Analysis", "", f"Embedding model: `{args.embedding_model_name}`",
             f"Bootstrap: n={args.n_bootstrap}, seed={args.bootstrap_seed}, "
             f"confidence level={args.confidence_level:.0%}", "", "## Main results (bootstrap mean ± 95% CI)", ""]
    metric_names = sorted({m for metrics in aggregated_by_model.values() for m in metrics})
    lines.append("| model | " + " | ".join(metric_names) + " |")
    lines.append("|---" * (len(metric_names) + 1) + "|")
    for model_name in MODEL_ORDER:
        row = [model_name]
        for m in metric_names:
            agg = aggregated_by_model[model_name].get(m)
            row.append(f"{agg.mean:.4f} [{agg.ci_low:.4f}, {agg.ci_high:.4f}]" if agg else "—")
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Pre-registered ladder comparisons", "",
              "| comparison | metric | mean difference | 95% CI | cohen's d | wilcoxon p | significant (Holm-corrected) |",
              "|---|---|---|---|---|---|---|"]
    for later, earlier in LADDER_COMPARISONS:
        comparison_name = f"{later}_vs_{earlier}"
        for metric in PRIMARY_METRICS:
            r = comparison_results[comparison_name][metric]
            lines.append(
                f"| {comparison_name} | {metric} | {r['mean_difference']:.4f} | "
                f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}] | {r['cohens_d']:.3f} | "
                f"{r['wilcoxon_p']:.4g} | {r['significant_after_correction']} |"
            )
    lines.append("")
    lines.append("No interpretation of these numbers is included here by design — see the E1 experiment design's execution checklist.")
    (output_dir / "REPORT.md").write_text("\n".join(lines))

    print(f"\nDone. Report at {output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
