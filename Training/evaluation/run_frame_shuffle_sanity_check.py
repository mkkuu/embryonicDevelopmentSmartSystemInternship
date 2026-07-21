"""
Frame-shuffle sanity experiment.

Retrains and re-evaluates all four registered models on the SAME
trajectories with window order scrambled within each trajectory (see
evaluation.trajectory.shuffle_trajectory) — train AND test, so
Persistence/LinearSSM's transition matrix and residual calibration are
both fit and evaluated under destroyed temporal order, not just evaluated
under it.

Purpose: if persistence/linear_ssm's advantage over identity_dynamics
(established, if it exists, by run_e1_full_analysis.py) SURVIVES this
shuffle, that is direct evidence they are not actually using genuine
temporal order — something else is driving the apparent advantage, and
the E1 conclusion as originally stated would not hold.

Built-in validity check: base_rate and identity_dynamics are, by
construction, invariant to window order — see
test_predictions_are_invariant_to_window_order in
Tests/evaluation/models/test_identity_dynamics.py for the property this
relies on. If shuffling changes THEIR results at all (beyond floating
point noise), the shuffle implementation itself is broken, and nothing
else this script reports should be trusted until that is fixed. This
script checks that condition explicitly and reports it first.

This script does NOT interpret its own output — see the accompanying
execution checklist.

Usage
-----
    cd Training
    python -m evaluation.run_frame_shuffle_sanity_check \\
        --cache_root ../Embeddings --embedding_model_name resnet18 \\
        --n_bootstrap 1000 --bootstrap_seed 0 --shuffle_seed 0 \\
        --output_dir ../Results/evaluation/e1_frame_shuffle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

from embeddings.dataset import EmbeddingDataset

from evaluation.metrics import bootstrap_trajectory_metrics
from evaluation.model import get_model_class
from evaluation.trajectory import group_into_trajectories, shuffle_trajectory

import evaluation.models.base_rate  # noqa: F401
import evaluation.models.identity_dynamics  # noqa: F401
import evaluation.models.linear_ssm  # noqa: F401
import evaluation.models.persistence  # noqa: F401

MODEL_ORDER = ["base_rate", "identity_dynamics", "persistence", "linear_ssm"]
ORDER_INVARIANT_MODELS = {"base_rate", "identity_dynamics"}  # must show zero change — the built-in check
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
    p.add_argument("--shuffle_seed", type=int, default=0)
    p.add_argument("--output_dir", default="../Results/evaluation/e1_frame_shuffle")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_root = Path(args.cache_root) / args.embedding_model_name
    train_real = group_into_trajectories(EmbeddingDataset(cache_root, "train"))
    val_real = group_into_trajectories(EmbeddingDataset(cache_root, "val"))
    test_real = group_into_trajectories(EmbeddingDataset(cache_root, "test"))

    shuffle_rng = np.random.RandomState(args.shuffle_seed)
    train_shuffled = [shuffle_trajectory(t, shuffle_rng) for t in train_real]
    val_shuffled = [shuffle_trajectory(t, shuffle_rng) for t in val_real]
    test_shuffled = [shuffle_trajectory(t, shuffle_rng) for t in test_real]

    report: Dict[str, dict] = {}
    for name in MODEL_ORDER:
        model_real = get_model_class(name)(**MODEL_KWARGS[name])
        model_real.fit(train_real, val_real)
        agg_real, _ = bootstrap_trajectory_metrics(
            model_real, test_real, n_bootstrap=args.n_bootstrap, rng_seed=args.bootstrap_seed
        )

        model_shuffled = get_model_class(name)(**MODEL_KWARGS[name])
        model_shuffled.fit(train_shuffled, val_shuffled)
        agg_shuffled, _ = bootstrap_trajectory_metrics(
            model_shuffled, test_shuffled, n_bootstrap=args.n_bootstrap, rng_seed=args.bootstrap_seed
        )

        report[name] = {
            metric: {
                "real_mean": agg_real[metric].mean,
                "shuffled_mean": agg_shuffled[metric].mean,
                "difference": agg_shuffled[metric].mean - agg_real[metric].mean,
            }
            for metric in PRIMARY_METRICS
        }
        print(f"[{name}] " + ", ".join(
            f"{m}: real={report[name][m]['real_mean']:.4f} shuffled={report[name][m]['shuffled_mean']:.4f} "
            f"diff={report[name][m]['difference']:+.4f}" for m in PRIMARY_METRICS
        ))

    # --- built-in validity check, reported first and separately ---
    validity_problems = []
    for name in ORDER_INVARIANT_MODELS:
        for metric in PRIMARY_METRICS:
            diff = abs(report[name][metric]["difference"])
            if diff > 1e-6:
                validity_problems.append(
                    f"{name}.{metric} changed by {diff:.6g} under shuffling, but this model is "
                    f"order-invariant BY CONSTRUCTION — the shuffle implementation itself is broken, "
                    f"not this model. Do not trust persistence/linear_ssm's results below until fixed."
                )

    output = {
        "validity_check_passed": len(validity_problems) == 0,
        "validity_problems": validity_problems,
        "results": report,
    }
    (output_dir / "frame_shuffle_report.json").write_text(json.dumps(output, indent=2))

    print()
    if validity_problems:
        print("VALIDITY CHECK FAILED:")
        for p in validity_problems:
            print(f"  - {p}")
    else:
        print("Validity check passed: base_rate and identity_dynamics are unaffected by shuffling, as required.")
    print(f"\nFull report at {output_dir / 'frame_shuffle_report.json'}")


if __name__ == "__main__":
    main()
