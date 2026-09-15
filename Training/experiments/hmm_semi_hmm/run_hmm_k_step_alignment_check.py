"""
Diagnostic CLI, additive and Val-only: scores an already-frozen HMMModel
checkpoint (loaded via HMMModel.load(), never refit) against `consistency_flag`
using both the old lag-1 event (predict(), k=1) and the k=(window_size-1)
event predict_k_step_transition_probability() was built for (see
Training/evaluation/models/hmm.py's own docstrings for the full derivation of
why k=window_size-1 is the event consistency_flag actually measures).

This exists to PERSIST the measurement as a reviewable artifact -- the
2026-08-21 session that implemented predict_k_step_transition_probability
reported these numbers in docs/PROJECT_STATE.md but never wrote them to a
file under Results/, so they could not be independently re-checked without
re-running the diagnostic from scratch. This script is that persistence,
nothing more: it does not retrain, does not touch Train (the loaded model's
own `_consistency_base_rate` already captures everything from Train that this
diagnostic needs), and structurally cannot touch Test (no Test-loading code
path exists here, mirroring run_hmm_evaluation.py's own structural guard).

Usage
-----
    cd Training
    python -m experiments.hmm_semi_hmm.run_hmm_k_step_alignment_check \\
        --cache_root ../Embeddings --embedding_model_name resnet18 \\
        --model_dir ../Results/evaluation/e1_hmm_step1_train_val_sweep/best_model \\
        --k 7 \\
        --output_dir ../Results/evaluation/e1_hmm_step2_k_step_alignment_check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

from embeddings.dataset import EmbeddingDataset

from evaluation.calibration import score_probabilistic
from evaluation.models.hmm import HMMModel
from evaluation.trajectory import Trajectory, group_into_trajectories


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_root", default="../Embeddings")
    p.add_argument("--embedding_model_name", default="resnet18", choices=["resnet18", "timesformer"])
    p.add_argument("--model_dir", required=True, help="Path to a directory saved by HMMModel.save().")
    p.add_argument("--k", type=int, default=7, help="k for predict_k_step_transition_probability (default 7 = window_size-1 at real Train/Val scale).")
    p.add_argument("--n_bins", type=int, default=10, help="Number of equal-width reliability/ECE bins.")
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def _flatten_flag(trajectories: List[Trajectory]) -> np.ndarray:
    flags = []
    for traj in trajectories:
        flags.extend(traj.consistency_flag.tolist())
    return np.asarray(flags, dtype=np.int64)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_root = Path(args.cache_root) / args.embedding_model_name
    val_trajectories = group_into_trajectories(EmbeddingDataset(cache_root, "val"))

    model = HMMModel.load(Path(args.model_dir))

    y_flag = _flatten_flag(val_trajectories)

    # Coverage is tracked explicitly per-window (via window_starts value
    # matching, mirroring predict_k_step_transition_probability's own logic)
    # rather than inferred from "prob == base_rate", which would misfire
    # whenever a genuinely computed value happens to coincide with the base
    # rate.
    k1_probs, k1_coverage = [], []
    kk_probs, kk_coverage = [], []
    for traj in val_trajectories:
        T = len(traj)
        start_to_index = {s: i for i, s in enumerate(traj.window_starts)}
        p1 = model.predict_k_step_transition_probability(traj, k=1)
        pk = model.predict_k_step_transition_probability(traj, k=args.k)
        k1_probs.extend(p1.tolist())
        kk_probs.extend(pk.tolist())
        for t in range(T):
            t0_1 = start_to_index.get(traj.window_starts[t] - 1)
            k1_coverage.append(bool(t0_1 is not None and t0_1 + 1 == t))
            t0_k = start_to_index.get(traj.window_starts[t] - args.k)
            kk_coverage.append(bool(t0_k is not None and t0_k + args.k == t))
    k1_probs = np.asarray(k1_probs, dtype=np.float64)
    kk_probs = np.asarray(kk_probs, dtype=np.float64)
    k1_coverage = np.asarray(k1_coverage, dtype=bool)
    kk_coverage = np.asarray(kk_coverage, dtype=bool)

    report = {
        "model_dir": str(args.model_dir),
        "cache_root": str(cache_root),
        "n_val_trajectories": len(val_trajectories),
        "n_val_windows_total": int(len(y_flag)),
        "consistency_base_rate_from_model": model._consistency_base_rate,
        "k_used_for_alignment": args.k,
        "coverage": {
            "k1_covered_fraction": float(k1_coverage.mean()) if len(k1_coverage) else float("nan"),
            "k_covered_fraction": float(kk_coverage.mean()) if len(kk_coverage) else float("nan"),
        },
        "k1_vs_consistency_flag_all_windows": score_probabilistic(y_flag, k1_probs, args.n_bins),
        "k_vs_consistency_flag_all_windows": score_probabilistic(y_flag, kk_probs, args.n_bins),
        "k1_vs_consistency_flag_covered_only": score_probabilistic(y_flag[k1_coverage], k1_probs[k1_coverage], args.n_bins),
        "k_vs_consistency_flag_covered_only": score_probabilistic(y_flag[kk_coverage], kk_probs[kk_coverage], args.n_bins),
    }
    (output_dir / "k_step_alignment_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps({
        "k1_all": {k: report["k1_vs_consistency_flag_all_windows"][k] for k in ("auroc", "brier", "brier_baseline_constant_rate", "ece")},
        f"k{args.k}_all": {k: report["k_vs_consistency_flag_all_windows"][k] for k in ("auroc", "brier", "brier_baseline_constant_rate", "ece")},
        "coverage": report["coverage"],
    }, indent=2))
    print("\nFull report written to", output_dir / "k_step_alignment_report.json")


if __name__ == "__main__":
    main()
