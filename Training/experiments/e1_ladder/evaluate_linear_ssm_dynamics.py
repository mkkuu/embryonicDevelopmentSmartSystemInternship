"""
Evaluates LinearStateSpaceModel's dynamics-specific capabilities — future
prediction and missing-frame imputation — which are NOT part of the
standard classification-metrics pipeline (evaluation.runner /
evaluation.metrics), since forecast/imputation quality is a regression
problem (embedding reconstruction error), not a classification one. This
is a separate evaluation modality, not an omission from run_e1_baselines.

Reports, on the test split:
  - k-step-ahead forecast MSE, for k = 1..max_steps, against a
    Persistence-style naive baseline (forecast = the last observed
    embedding, unchanged) — so the numbers are interpretable relative to
    something, not just absolute.
  - single-window imputation MSE (each window in turn withheld and
    reconstructed from its neighbors), against the same naive baseline.

Usage
-----
    cd Training
    python -m experiments.e1_ladder.evaluate_linear_ssm_dynamics \\
        --cache_root ../Embeddings --embedding_model_name resnet18 \\
        --latent_dim 16 --max_steps 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from embeddings.dataset import EmbeddingDataset
from evaluation.models.linear_ssm import LinearStateSpaceModel
from evaluation.trajectory import Trajectory, group_into_trajectories


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_root", default="../Embeddings")
    p.add_argument("--embedding_model_name", default="resnet18", choices=["resnet18", "timesformer"])
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--ridge_alpha", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=5)
    p.add_argument("--output_path", default="../Results/evaluation/linear_ssm_dynamics_report.json")
    return p.parse_args()


def evaluate_forecast(model: LinearStateSpaceModel, trajectories: List[Trajectory], max_steps: int) -> Dict[int, Dict[str, float]]:
    model_errors: Dict[int, List[float]] = {k: [] for k in range(1, max_steps + 1)}
    naive_errors: Dict[int, List[float]] = {k: [] for k in range(1, max_steps + 1)}

    for traj in trajectories:
        T = len(traj)
        for k in range(1, max_steps + 1):
            for i in range(T - k):
                start_embedding = traj.embeddings[i]
                actual = traj.embeddings[i + k]

                forecasted = model.forecast_from_embedding(start_embedding, k)[-1]
                model_errors[k].append(float(((forecasted - actual) ** 2).mean()))

                naive_forecast = start_embedding  # persistence-style: unchanged
                naive_errors[k].append(float(((naive_forecast - actual) ** 2).mean()))

    return {
        k: {
            "model_mse": float(np.mean(model_errors[k])) if model_errors[k] else float("nan"),
            "naive_mse": float(np.mean(naive_errors[k])) if naive_errors[k] else float("nan"),
            "n_pairs": len(model_errors[k]),
        }
        for k in range(1, max_steps + 1)
    }


def evaluate_imputation(model: LinearStateSpaceModel, trajectories: List[Trajectory]) -> Dict[str, float]:
    model_errors: List[float] = []
    naive_errors: List[float] = []

    for traj in trajectories:
        T = len(traj)
        if T < 2:
            continue
        for i in range(T):
            true_value = traj.embeddings[i]
            try:
                imputed = model.impute_missing_window(traj, missing_index=i)
            except RuntimeError:
                continue  # e.g. last window with a non-invertible A — genuinely not imputable here, skip and move on
            model_errors.append(float(((imputed - true_value) ** 2).mean()))

            # Naive baseline: nearest observed neighbor's value, whichever side exists.
            if i > 0:
                naive = traj.embeddings[i - 1]
            else:
                naive = traj.embeddings[i + 1]
            naive_errors.append(float(((naive - true_value) ** 2).mean()))

    return {
        "model_mse": float(np.mean(model_errors)) if model_errors else float("nan"),
        "naive_mse": float(np.mean(naive_errors)) if naive_errors else float("nan"),
        "n_windows": len(model_errors),
    }


def main() -> None:
    args = parse_args()

    cache_root = Path(args.cache_root) / args.embedding_model_name
    train_trajectories = group_into_trajectories(EmbeddingDataset(cache_root, "train"))
    test_trajectories = group_into_trajectories(EmbeddingDataset(cache_root, "test"))

    model = LinearStateSpaceModel(latent_dim=args.latent_dim, ridge_alpha=args.ridge_alpha)
    model.fit(train_trajectories)

    forecast_report = evaluate_forecast(model, test_trajectories, args.max_steps)
    imputation_report = evaluate_imputation(model, test_trajectories)

    report = {
        "embedding_model_name": args.embedding_model_name,
        "latent_dim": args.latent_dim,
        "ridge_alpha": args.ridge_alpha,
        "forecast": forecast_report,
        "imputation": imputation_report,
    }

    for k, stats in forecast_report.items():
        print(f"[forecast k={k}] model_mse={stats['model_mse']:.6f}  naive_mse={stats['naive_mse']:.6f}  n={stats['n_pairs']}")
    print(f"[imputation] model_mse={imputation_report['model_mse']:.6f}  "
          f"naive_mse={imputation_report['naive_mse']:.6f}  n={imputation_report['n_windows']}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"\nFull report written to {output_path}")


if __name__ == "__main__":
    main()
