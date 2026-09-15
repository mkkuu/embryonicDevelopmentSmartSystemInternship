"""
STEP 6B — GRU TEST evaluation. First and only time this project's GRU work
touches the TEST split. No GRU hyperparameter is changed after this script
is run — see the STEP 6 migration study's no-tuning-on-test discipline,
already applied identically to E1 (run_e1_full_analysis.py).

Never refits GRU here: loads the already-trained, frozen checkpoints from
STEP 6A (seed=0) and the STEP 6B extra-seed runs (seed=1, seed=2) written
under --gru_checkpoints_root. Only the four closed-form E1 baselines are
fit here, fresh, on train/val — deterministic and cheap, exactly mirroring
run_e1_full_analysis.py's own fitting code so the comparison is
apples-to-apples with E1's own numbers.

Pre-registered comparisons (decided before this script was ever run against
real Test data): each of the 6 (formulation, seed) GRU checkpoints against
identity_dynamics — the strongest E1 baseline — on {accuracy, auroc},
Holm-Bonferroni corrected per metric across those 6 comparisons. This
mirrors the exact pre-specified outcome cases already discussed for STEP 6:
GRU beating identity_dynamics on both formulations favors non-linear
dynamics; on one only, favors it with caveats; on neither, reinforces E1's
negative result for dynamics of any kind tried so far.

gru_forecast vs linear_ssm (the direct linear-vs-nonlinear mechanistic
parity comparison, per gru.py's own module docstring) is reported
separately and explicitly marked DESCRIPTIVE ONLY — not part of the
Holm-Bonferroni family, to avoid retroactively expanding the pre-registered
comparison set after the fact.

Usage
-----
    cd Training
    python -m experiments.gru.run_gru_test_evaluation \\
        --cache_root ../Embeddings --embedding_model_name resnet18 \\
        --gru_checkpoints_root ../Results/evaluation/e1_gru_step6a_full \\
        --seeds 0 1 2 \\
        --n_bootstrap 1000 --bootstrap_seed 0 \\
        --output_dir ../Results/evaluation/e1_gru_step6b_test_evaluation
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
from evaluation.models.gru import GRUDynamicsModel
from evaluation.trajectory import group_into_trajectories

# Imported for @register_model side effects.
import evaluation.models.base_rate  # noqa: F401
import evaluation.models.identity_dynamics  # noqa: F401
import evaluation.models.linear_ssm  # noqa: F401
import evaluation.models.persistence  # noqa: F401

BASELINE_MODEL_ORDER = ["base_rate", "identity_dynamics", "persistence", "linear_ssm"]
BASELINE_MODEL_KWARGS: Dict[str, dict] = {
    "base_rate": {},
    "identity_dynamics": {},
    "persistence": {},
    "linear_ssm": {"latent_dim": 16, "ridge_alpha": 1.0},
}
GRU_FORMULATIONS = ["forecast", "classification"]
PRIMARY_METRICS = ["accuracy", "auroc"]
REFERENCE_BASELINE = "identity_dynamics"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_root", default="../Embeddings")
    p.add_argument("--embedding_model_name", default="resnet18", choices=["resnet18", "timesformer"])
    p.add_argument("--gru_checkpoints_root", default="../Results/evaluation/e1_gru_step6a_full")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument(
        "--gru_device", default="cpu",
        help="Device to run GRU predict() on during bootstrap (CPU is a per-trajectory "
        "Python loop and does not vectorize across resamples — pass 'cuda' when a GPU "
        "is available and free, to avoid an impractically slow bootstrap).",
    )
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--bootstrap_seed", type=int, default=0)
    p.add_argument("--confidence_level", type=float, default=0.95)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--output_dir", default="../Results/evaluation/e1_gru_step6b_test_evaluation")
    return p.parse_args()


def _gru_checkpoint_dir(root: Path, formulation: str, seed: int) -> Path:
    # seed=0 lives at <root>/<formulation>/checkpoint (STEP 6A's own layout);
    # seed>0 lives at <root>/<formulation>_seed<seed>/checkpoint (STEP 6B's
    # extra-seed runs) — both written by the same run_gru_evaluation.py.
    if seed == 0:
        return root / formulation / "checkpoint"
    return root / f"{formulation}_seed{seed}" / "checkpoint"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gru_root = Path(args.gru_checkpoints_root)

    cache_root = Path(args.cache_root) / args.embedding_model_name
    train_trajectories = group_into_trajectories(EmbeddingDataset(cache_root, "train"))
    val_trajectories = group_into_trajectories(EmbeddingDataset(cache_root, "val"))
    test_trajectories = group_into_trajectories(EmbeddingDataset(cache_root, "test"))
    print(
        f"Loaded {len(train_trajectories)} train / {len(val_trajectories)} val / "
        f"{len(test_trajectories)} test trajectories."
    )
    print(
        "NOTE: this is the first time the GRU work touches the TEST split. "
        "No further GRU hyperparameter changes are permitted after this point."
    )

    fitted_baselines = {}
    for name in BASELINE_MODEL_ORDER:
        model = get_model_class(name)(**BASELINE_MODEL_KWARGS[name])
        model.fit(train_trajectories, val_trajectories)
        fitted_baselines[name] = model
        print(f"Fitted baseline {name}.")

    gru_models: Dict[str, GRUDynamicsModel] = {}
    for formulation in GRU_FORMULATIONS:
        for seed in args.seeds:
            ckpt_dir = _gru_checkpoint_dir(gru_root, formulation, seed)
            key = f"gru_{formulation}_seed{seed}"
            gru_models[key] = GRUDynamicsModel.load(ckpt_dir, device=args.gru_device)
            print(f"Loaded frozen checkpoint {key} from {ckpt_dir}")

    all_models = {**fitted_baselines, **gru_models}
    model_order = BASELINE_MODEL_ORDER + list(gru_models.keys())

    aggregated_by_model = {}
    raw_by_model = {}
    for name in model_order:
        aggregated, raw = bootstrap_trajectory_metrics(
            all_models[name], test_trajectories, n_bootstrap=args.n_bootstrap,
            confidence_level=args.confidence_level, rng_seed=args.bootstrap_seed,
        )
        aggregated_by_model[name] = aggregated
        raw_by_model[name] = raw
        print(
            f"Bootstrapped {name}: accuracy={aggregated['accuracy'].mean:.4f} "
            f"[{aggregated['accuracy'].ci_low:.4f},{aggregated['accuracy'].ci_high:.4f}] "
            f"auroc={aggregated['auroc'].mean:.4f} "
            f"[{aggregated['auroc'].ci_low:.4f},{aggregated['auroc'].ci_high:.4f}]"
        )

    comparison_results: Dict[str, Dict[str, dict]] = {}
    p_values_by_metric: Dict[str, Dict[str, float]] = {m: {} for m in PRIMARY_METRICS}
    for formulation in GRU_FORMULATIONS:
        for seed in args.seeds:
            gru_key = f"gru_{formulation}_seed{seed}"
            comparison_name = f"{gru_key}_vs_{REFERENCE_BASELINE}"
            comparison_results[comparison_name] = {}
            for metric in PRIMARY_METRICS:
                result = paired_bootstrap_comparison(
                    raw_by_model[REFERENCE_BASELINE], raw_by_model[gru_key], metric, args.confidence_level
                )
                comparison_results[comparison_name][metric] = result
                p_values_by_metric[metric][comparison_name] = result["wilcoxon_p"]

    significance_by_metric = {m: holm_bonferroni(p_values_by_metric[m], alpha=args.alpha) for m in PRIMARY_METRICS}
    for metric in PRIMARY_METRICS:
        for comparison_name in significance_by_metric[metric]:
            comparison_results[comparison_name][metric]["significant_after_correction"] = (
                significance_by_metric[metric][comparison_name]
            )

    # Descriptive only: gru_forecast vs linear_ssm, the direct linear-vs-nonlinear
    # mechanistic parity comparison — NOT part of the pre-registered/corrected family.
    descriptive_comparisons: Dict[str, Dict[str, dict]] = {}
    for seed in args.seeds:
        gru_key = f"gru_forecast_seed{seed}"
        comparison_name = f"{gru_key}_vs_linear_ssm"
        descriptive_comparisons[comparison_name] = {
            metric: paired_bootstrap_comparison(raw_by_model["linear_ssm"], raw_by_model[gru_key], metric, args.confidence_level)
            for metric in PRIMARY_METRICS
        }

    (output_dir / "aggregated_results.json").write_text(
        json.dumps(
            {m: {k: v.__dict__ for k, v in metrics.items()} for m, metrics in aggregated_by_model.items()},
            indent=2, default=list,
        )
    )
    (output_dir / "comparison_results.json").write_text(json.dumps(comparison_results, indent=2))
    (output_dir / "descriptive_comparisons.json").write_text(json.dumps(descriptive_comparisons, indent=2))

    lines = [
        "# STEP 6B — GRU Test Evaluation", "",
        f"Embedding model: `{args.embedding_model_name}`",
        f"Bootstrap: n={args.n_bootstrap}, seed={args.bootstrap_seed}, confidence={args.confidence_level:.0%}",
        f"GRU seeds evaluated: {args.seeds}", "",
        "## Main results (bootstrap mean ± 95% CI, TEST split)", "",
    ]
    metric_names = PRIMARY_METRICS + ["precision", "recall", "f1", "log_loss"]
    lines.append("| model | " + " | ".join(metric_names) + " |")
    lines.append("|---" * (len(metric_names) + 1) + "|")
    for name in model_order:
        row = [name]
        for m in metric_names:
            agg = aggregated_by_model[name].get(m)
            row.append(f"{agg.mean:.4f} [{agg.ci_low:.4f}, {agg.ci_high:.4f}]" if agg else "—")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "", f"## Pre-registered comparisons vs {REFERENCE_BASELINE} "
        "(Holm-Bonferroni corrected per metric, across the 6 formulation×seed comparisons)", "",
        "| comparison | metric | mean difference (gru − baseline) | 95% CI | cohen's d | wilcoxon p | significant |",
        "|---|---|---|---|---|---|---|",
    ]
    for formulation in GRU_FORMULATIONS:
        for seed in args.seeds:
            gru_key = f"gru_{formulation}_seed{seed}"
            comparison_name = f"{gru_key}_vs_{REFERENCE_BASELINE}"
            for metric in PRIMARY_METRICS:
                r = comparison_results[comparison_name][metric]
                lines.append(
                    f"| {comparison_name} | {metric} | {r['mean_difference']:.4f} | "
                    f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}] | {r['cohens_d']:.3f} | "
                    f"{r['wilcoxon_p']:.4g} | {r['significant_after_correction']} |"
                )

    lines += [
        "", "## Descriptive only (NOT Holm-Bonferroni corrected, NOT pre-registered): gru_forecast vs linear_ssm", "",
        "| comparison | metric | mean difference | 95% CI | wilcoxon p |",
        "|---|---|---|---|---|",
    ]
    for seed in args.seeds:
        gru_key = f"gru_forecast_seed{seed}"
        comparison_name = f"{gru_key}_vs_linear_ssm"
        for metric in PRIMARY_METRICS:
            r = descriptive_comparisons[comparison_name][metric]
            lines.append(
                f"| {comparison_name} | {metric} | {r['mean_difference']:.4f} | "
                f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}] | {r['wilcoxon_p']:.4g} |"
            )

    lines.append("")
    lines.append("No interpretation of these numbers is included here by design.")
    (output_dir / "REPORT.md").write_text("\n".join(lines))

    print(f"\nDone. Report at {output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
