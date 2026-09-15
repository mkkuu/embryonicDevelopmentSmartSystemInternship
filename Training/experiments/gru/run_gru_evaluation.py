"""
CLI entry point for GRUDynamicsModel — separate from, and never touching,
run_e1_full_analysis.py / evaluate_linear_ssm_dynamics.py / the artifacts
they already produced. Mirrors their structure (argparse, load cached
embeddings via embeddings.dataset.EmbeddingDataset + evaluation.trajectory.
group_into_trajectories, evaluate via evaluation.metrics) but additionally
owns the training loop that the four closed-form E1 models never needed
(GRUDynamicsModel.fit() itself contains that loop — this script is
orchestration around it, not a second copy of it).

--formulation maps directly onto GRUDynamicsModel's predict_mode:
  --formulation forecast        -> predict_mode="residual" (mirrors
                                    PersistenceModel/LinearStateSpaceModel)
  --formulation classification  -> predict_mode="direct"

TEST-split discipline (STEP 1 design, see the migration study's Section 4)
---------------------------------------------------------------------------
This script NEVER loads the test split unless --final_eval is passed
explicitly. This is a structural guard, not just a discipline: the smoke
test / hyperparameter-exploration code path below has no reference to
EmbeddingDataset(cache_root, "test") anywhere unless that flag is set, so
touching test by accident during Phases 1-5 is not just discouraged, it is
absent from the code path.

--max_trajectories truncates to the first N trajectories in
group_into_trajectories()'s own (video first-appearance) order — this is a
debug/smoke-test convenience ONLY, explicitly NOT a representative random
sample (unlike DataSet.py's NumberOfVideos, which IS a seeded sample) —
never use --max_trajectories for anything other than Phase 3/4 smoke
testing; the real Phase 5/6 run must leave it at its default (0 = use the
entire split).

Usage (smoke test, CPU, Phase 3 — see the migration study)
------------------------------------------------------------
    cd Training
    python -m experiments.gru.run_gru_evaluation \\
        --cache_root ../Embeddings --embedding_model_name resnet18 \\
        --formulation forecast \\
        --hidden_dim 32 --num_layers 1 --dropout 0.1 --weight_decay 1e-5 \\
        --max_trajectories 5 --epochs 2 --device cpu --seed 0 \\
        --output_dir ../Results/evaluation/e1_gru_smoke_forecast

Not executed as part of writing this file — Phase 3 authorization is
separate (see the migration study's Section 14 / STEP 3).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

from embeddings.dataset import EmbeddingDataset

from evaluation.metrics import compute_metrics
from evaluation.models.gru import GRUDynamicsModel
from evaluation.trajectory import Trajectory, group_into_trajectories

FORMULATION_TO_PREDICT_MODE = {"forecast": "residual", "classification": "direct"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_root", default="../Embeddings")
    p.add_argument("--embedding_model_name", default="resnet18", choices=["resnet18", "timesformer"])
    p.add_argument("--formulation", required=True, choices=["forecast", "classification"])
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=50, help="Maximum epochs; early stopping may end training sooner.")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="Deliberately defaults to CPU — pass --device cuda explicitly for GPU phases.")
    p.add_argument(
        "--max_trajectories",
        type=int,
        default=0,
        help="0 (default) = use the entire split. >0 truncates TRAIN and VAL to "
        "the first N trajectories each, for Phase 3/4 smoke testing ONLY — see "
        "module docstring. Never use for a real Phase 5/6 run.",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--final_eval",
        action="store_true",
        help="Loads and evaluates the TEST split. Off by default — see module "
        "docstring's TEST-split discipline section. Only intended for Phase 6.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an in-progress checkpoint at "
        "<output_dir>/checkpoint_inprogress/train_state.pt if one exists. "
        "Without this flag, the script refuses to run if such a checkpoint "
        "is already present (mirrors embeddings/build_cache.py's convention) "
        "rather than silently overwriting in-progress training state.",
    )
    return p.parse_args()


def _load_split(cache_root: Path, split: str, max_trajectories: int) -> List[Trajectory]:
    trajectories = group_into_trajectories(EmbeddingDataset(cache_root, split))
    if max_trajectories > 0:
        trajectories = trajectories[:max_trajectories]
    return trajectories


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_root = Path(args.cache_root) / args.embedding_model_name
    train_trajectories = _load_split(cache_root, "train", args.max_trajectories)
    val_trajectories = _load_split(cache_root, "val", args.max_trajectories)
    print(f"Loaded {len(train_trajectories)} train / {len(val_trajectories)} val trajectories.")
    if args.max_trajectories > 0:
        print(
            f"NOTE: --max_trajectories={args.max_trajectories} is active — this is a "
            f"smoke-test subset (first N in cache order, NOT a representative sample), "
            f"never comparable to a real E1 result."
        )

    model = GRUDynamicsModel(
        predict_mode=FORMULATION_TO_PREDICT_MODE[args.formulation],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        max_epochs=args.epochs,
        patience=args.patience,
        grad_clip_norm=args.grad_clip_norm,
        seed=args.seed,
        device=args.device,
    )

    inprogress_dir = output_dir / "checkpoint_inprogress"
    inprogress_state = inprogress_dir / "train_state.pt"
    if inprogress_state.exists() and not args.resume:
        raise SystemExit(
            f"An in-progress checkpoint already exists at {inprogress_state} but --resume "
            f"was not passed. Refusing to run (would silently discard it). Pass --resume "
            f"to continue from it, or remove it explicitly if you intend to start over."
        )

    print(f"Fitting GRUDynamicsModel(predict_mode={model._predict_mode!r})...")
    fit_t0 = time.time()
    model.fit(
        train_trajectories,
        val_trajectories,
        checkpoint_dir=inprogress_dir,
        resume=args.resume,
        verbose=True,
    )
    total_training_seconds = time.time() - fit_t0
    print(f"Training finished after {len(model._train_history)} epoch(s) in {total_training_seconds:.1f}s.")
    if model._train_history:
        last = model._train_history[-1]
        print(f"Last epoch: train_loss={last['train_loss']:.6f}  val_loss={last['val_loss']:.6f}")

    best_epoch = min(range(len(model._train_history)), key=lambda i: model._train_history[i]["val_loss"])
    best_val_loss = model._train_history[best_epoch]["val_loss"]
    early_stopped = len(model._train_history) < args.epochs
    mean_epoch_seconds = (
        sum(h["epoch_seconds"] for h in model._train_history) / len(model._train_history)
        if model._train_history
        else 0.0
    )
    print(
        f"Best epoch: {best_epoch} (val_loss={best_val_loss:.6f}) | "
        f"early_stopped={early_stopped} | mean {mean_epoch_seconds:.1f}s/epoch"
    )

    checkpoint_dir = output_dir / "checkpoint"
    model.save(checkpoint_dir)
    print(f"Final (best-val) checkpoint written to {checkpoint_dir}")

    report = {
        "formulation": args.formulation,
        "predict_mode": model._predict_mode,
        "hyperparameters": {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "learning_rate": args.learning_rate,
            "epochs_max": args.epochs,
            "patience": args.patience,
            "grad_clip_norm": args.grad_clip_norm,
            "seed": args.seed,
        },
        "max_trajectories": args.max_trajectories,
        "n_train_trajectories": len(train_trajectories),
        "n_val_trajectories": len(val_trajectories),
        "train_history": model._train_history,
        "epochs_run": len(model._train_history),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "early_stopped": early_stopped,
        "mean_epoch_seconds": mean_epoch_seconds,
        "total_training_seconds": total_training_seconds,
        "resumed": args.resume,
    }

    # VAL-only classification metrics, computed here purely so Phase 3/4 smoke
    # tests have something concrete to inspect — NOT a substitute for the real,
    # bootstrap-based Phase 6/7 evaluation against E1 (evaluation.metrics.
    # bootstrap_trajectory_metrics, reused unmodified, is the correct tool for
    # that and is deliberately NOT invoked here).
    if val_trajectories:
        val_predictions = model.predict(val_trajectories)
        report["val_metrics"] = {
            k: v for k, v in compute_metrics(val_trajectories, val_predictions).items() if isinstance(v, float)
        }
        print("Val metrics:", report["val_metrics"])

    if args.final_eval:
        test_trajectories = _load_split(cache_root, "test", args.max_trajectories)
        print(f"--final_eval passed: loaded {len(test_trajectories)} test trajectories.")
        test_predictions = model.predict(test_trajectories)
        report["n_test_trajectories"] = len(test_trajectories)
        report["test_metrics"] = {
            k: v for k, v in compute_metrics(test_trajectories, test_predictions).items() if isinstance(v, float)
        }
        print("Test metrics (point estimate only — NOT the bootstrap comparison to E1, see Phase 7):", report["test_metrics"])

    (output_dir / "run_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {output_dir / 'run_report.json'}")


if __name__ == "__main__":
    main()
