"""
STEP 6C — GRU frame-shuffle sanity check. Extends the same validity
control already run for the four E1 models (run_frame_shuffle_sanity_check.py)
to the GRU: if a GRU/identity_dynamics advantage measured in STEP 6B
survives training AND evaluating on window-order-scrambled trajectories,
that is evidence the GRU isn't actually exploiting genuine temporal order.

Scope decision: only the seed=0 checkpoint/training config is used here
(not all 3 STEP 6B seeds) — this is a validity CONTROL on the mechanism,
not a re-estimate of the point estimate, and GRU training is cheap enough
(~1-4 min/formulation measured in STEP 6A) that this is a time/scope choice,
not a compute constraint.

Unlike E1's frame-shuffle script, GRU has no order-invariant sibling model
in this script to serve as a built-in "must show zero change" validity
check. Validity here is instead judged by requiring this script's own
REAL (unshuffled) bootstrap numbers to match STEP 6B's REPORT.md numbers
for the same seed — computed independently, from a freshly-loaded frozen
checkpoint, exactly as run_frame_shuffle_sanity_check.py redundantly
recomputes the "real" condition rather than trusting run_e1_full_analysis.py's
numbers blindly (a known, accepted design choice already used there).

Usage
-----
    cd Training
    python -m experiments.gru.run_gru_frame_shuffle_sanity_check \\
        --cache_root ../Embeddings --embedding_model_name resnet18 \\
        --gru_real_checkpoints_root ../Results/evaluation/e1_gru_step6a_full \\
        --n_bootstrap 1000 --bootstrap_seed 0 --shuffle_seed 0 --device cuda \\
        --output_dir ../Results/evaluation/e1_gru_step6c_frame_shuffle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

from embeddings.dataset import EmbeddingDataset

from evaluation.metrics import bootstrap_trajectory_metrics
from evaluation.models.gru import GRUDynamicsModel
from evaluation.trajectory import group_into_trajectories, shuffle_trajectory

GRU_FORMULATIONS = ["forecast", "classification"]
PREDICT_MODE = {"forecast": "residual", "classification": "direct"}
PRIMARY_METRICS = ["accuracy", "auroc"]

# Exactly the STEP 6A/6B hyperparameters — a shuffle control must use the
# SAME configuration as the real run, or a difference could be attributed
# to the hyperparameter change rather than to shuffling itself.
GRU_KWARGS = dict(
    hidden_dim=32, num_layers=1, dropout=0.1, weight_decay=1e-5,
    learning_rate=1e-3, max_epochs=50, patience=8, grad_clip_norm=1.0, seed=0,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_root", default="../Embeddings")
    p.add_argument("--embedding_model_name", default="resnet18", choices=["resnet18", "timesformer"])
    p.add_argument("--gru_real_checkpoints_root", default="../Results/evaluation/e1_gru_step6a_full")
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--bootstrap_seed", type=int, default=0)
    p.add_argument("--shuffle_seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="../Results/evaluation/e1_gru_step6c_frame_shuffle")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_root) / args.embedding_model_name

    train_real = group_into_trajectories(EmbeddingDataset(cache_root, "train"))
    val_real = group_into_trajectories(EmbeddingDataset(cache_root, "val"))
    test_real = group_into_trajectories(EmbeddingDataset(cache_root, "test"))
    print(f"Loaded {len(train_real)} train / {len(val_real)} val / {len(test_real)} test trajectories.")

    shuffle_rng = np.random.RandomState(args.shuffle_seed)
    train_shuffled = [shuffle_trajectory(t, shuffle_rng) for t in train_real]
    val_shuffled = [shuffle_trajectory(t, shuffle_rng) for t in val_real]
    test_shuffled = [shuffle_trajectory(t, shuffle_rng) for t in test_real]

    report: Dict[str, dict] = {}
    for formulation in GRU_FORMULATIONS:
        real_ckpt = Path(args.gru_real_checkpoints_root) / formulation / "checkpoint"
        model_real = GRUDynamicsModel.load(real_ckpt, device=args.device)
        agg_real, _ = bootstrap_trajectory_metrics(
            model_real, test_real, n_bootstrap=args.n_bootstrap, rng_seed=args.bootstrap_seed
        )

        print(f"Fitting GRU({formulation}) on SHUFFLED train/val...")
        model_shuffled = GRUDynamicsModel(
            predict_mode=PREDICT_MODE[formulation], device=args.device, **GRU_KWARGS
        )
        formulation_dir = output_dir / formulation
        model_shuffled.fit(
            train_shuffled, val_shuffled,
            checkpoint_dir=formulation_dir / "checkpoint_inprogress",
            verbose=True,
        )
        model_shuffled.save(formulation_dir / "checkpoint")
        agg_shuffled, _ = bootstrap_trajectory_metrics(
            model_shuffled, test_shuffled, n_bootstrap=args.n_bootstrap, rng_seed=args.bootstrap_seed
        )

        report[formulation] = {
            metric: {
                "real_mean": agg_real[metric].mean,
                "real_ci": [agg_real[metric].ci_low, agg_real[metric].ci_high],
                "shuffled_mean": agg_shuffled[metric].mean,
                "shuffled_ci": [agg_shuffled[metric].ci_low, agg_shuffled[metric].ci_high],
                "difference": agg_shuffled[metric].mean - agg_real[metric].mean,
            }
            for metric in PRIMARY_METRICS
        }
        print(
            f"[{formulation}] "
            + ", ".join(
                f"{m}: real={report[formulation][m]['real_mean']:.4f} "
                f"shuffled={report[formulation][m]['shuffled_mean']:.4f} "
                f"diff={report[formulation][m]['difference']:+.4f}"
                for m in PRIMARY_METRICS
            )
        )

    output = {
        "note": (
            "GRU has no order-invariant sibling model in this script to serve as a "
            "built-in validity check the way E1's base_rate/identity_dynamics did. "
            "Cross-check the 'real_mean'/'real_ci' values above against STEP 6B's "
            "REPORT.md (gru_<formulation>_seed0 row) before trusting the shuffled "
            "numbers — they should match (same checkpoint, same bootstrap seed)."
        ),
        "results": report,
    }
    (output_dir / "frame_shuffle_report.json").write_text(json.dumps(output, indent=2))
    print(f"\nFull report at {output_dir / 'frame_shuffle_report.json'}")


if __name__ == "__main__":
    main()
