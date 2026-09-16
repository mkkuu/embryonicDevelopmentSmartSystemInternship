"""
Shared calibration-metric helpers (Brier score, reliability bins, ECE),
factored out so `run_hmm_evaluation.py`'s k=7 sweep and
`run_hmm_k_step_alignment_check.py`'s diagnostic never silently compute
these numbers two different ways. `evaluation.metrics.compute_metrics`
deliberately does not grow a Brier/ECE dependency here — it stays the
generic, model-agnostic scorer every ladder model already relies on;
this module is specific to the probabilistic-calibration questions the
HMM branch introduced.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from .trajectory import Trajectory, TrajectoryPrediction


def flatten_predictions(
    trajectories: List[Trajectory], predictions: List[TrajectoryPrediction]
) -> Tuple[np.ndarray, np.ndarray]:
    """Same position + window_starts alignment contract as
    evaluation.metrics._flatten (duplicated rather than imported: that
    name is underscore-private to metrics.py, and this module is meant to
    be usable independently of it)."""
    if len(trajectories) != len(predictions):
        raise ValueError(
            f"Got {len(trajectories)} trajectories but {len(predictions)} predictions — "
            f"predict()/predict_k_step() must return exactly one TrajectoryPrediction per "
            f"input Trajectory, in the same order."
        )
    y_true, y_prob = [], []
    for traj, pred in zip(trajectories, predictions):
        if pred.window_starts != traj.window_starts:
            raise ValueError(
                f"Prediction for video '{traj.video_name}' has window_starts "
                f"{pred.window_starts}, but ground truth has {traj.window_starts} — "
                f"predictions must be window-aligned with their input, in the same order."
            )
        y_true.extend(traj.consistency_flag.tolist())
        y_prob.extend(pred.consistency_flag_prob.tolist())
    return np.asarray(y_true), np.asarray(y_prob)


def calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> List[Dict]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi) if hi < 1.0 else (y_prob >= lo) & (y_prob <= hi)
        n = int(mask.sum())
        bins.append({
            "bin_lo": float(lo),
            "bin_hi": float(hi),
            "n": n,
            "mean_predicted": float(y_prob[mask].mean()) if n else None,
            "observed_frequency": float(y_true[mask].mean()) if n else None,
        })
    return bins


def expected_calibration_error(bins: List[Dict], n_total: int) -> float:
    if n_total == 0:
        return float("nan")
    return sum(
        (b["n"] / n_total) * abs(b["mean_predicted"] - b["observed_frequency"])
        for b in bins if b["n"] > 0
    )


def score_probabilistic(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict:
    """Brier/AUROC/ECE + reliability bins for one (y_true, y_prob) pair —
    the calibration-specific counterpart to evaluation.metrics.compute_metrics
    (which covers accuracy/precision/recall/f1/AUROC/log-loss but not
    Brier/ECE/bins)."""
    n = len(y_true)
    bins = calibration_bins(y_true, y_prob, n_bins)
    result = {
        "n_windows": n,
        "observed_positive_rate": float(y_true.mean()) if n else float("nan"),
        "brier": float(brier_score_loss(y_true, y_prob)) if n else float("nan"),
        "brier_baseline_constant_rate": float(brier_score_loss(y_true, np.full(n, y_true.mean()))) if n else float("nan"),
        "ece": expected_calibration_error(bins, n),
        "calibration_bins": bins,
    }
    if n and len(np.unique(y_true)) == 2:
        result["auroc"] = float(roc_auc_score(y_true, y_prob))
    else:
        result["auroc"] = float("nan")
        result["_auroc_note"] = "undefined: only one true class present"
    return result
