"""
Shared metric computation and cross-seed aggregation. Every model's
predictions are scored by the same code, so comparisons across models are
apples-to-apples by construction, not by discipline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .trajectory import Trajectory, TrajectoryPrediction


def _flatten(
    trajectories: List[Trajectory], predictions: List[TrajectoryPrediction]
) -> Tuple[np.ndarray, np.ndarray]:
    # Matched by POSITION, not by video_name. This was a video_name-keyed
    # dict lookup originally; that silently corrupts results the moment
    # the same video_name appears more than once in `trajectories` —
    # which patient-level bootstrap resampling (sampling with
    # replacement) does routinely and by design. Model.predict()'s
    # contract requires order-preserving output for exactly this reason —
    # see model.py.
    if len(trajectories) != len(predictions):
        raise ValueError(
            f"Got {len(trajectories)} trajectories but {len(predictions)} predictions — "
            f"predict() must return exactly one TrajectoryPrediction per input "
            f"Trajectory, in the same order."
        )
    y_true, y_prob = [], []
    for traj, pred in zip(trajectories, predictions):
        if pred.window_starts != traj.window_starts:
            raise ValueError(
                f"Prediction for video '{traj.video_name}' has window_starts "
                f"{pred.window_starts}, but ground truth has {traj.window_starts} "
                f"— predictions must be window-aligned with their input, in the "
                f"same order."
            )
        y_true.extend(traj.consistency_flag.tolist())
        y_prob.extend(pred.consistency_flag_prob.tolist())
    return np.asarray(y_true), np.asarray(y_prob)


def compute_metrics(trajectories: List[Trajectory], predictions: List[TrajectoryPrediction]) -> Dict[str, float]:
    """
    Flattens all windows across all trajectories and computes standard
    binary classification metrics on the consistency_flag task — the
    minimal metric set matching what E1 actually needs (see
    RESEARCH_BLUEPRINT.md Part VII): accuracy/precision/recall/F1 plus
    AUROC and log-loss, so probability quality is always reported
    alongside hard-label accuracy, never only the more flattering number.
    """
    y_true, y_prob = _flatten(trajectories, predictions)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "n_samples": len(y_true),
        "positive_rate_true": float(y_true.mean()) if len(y_true) else float("nan"),
        "positive_rate_predicted": float(y_pred.mean()) if len(y_pred) else float("nan"),
    }

    # AUROC/log-loss are undefined with only one true class present — report
    # NaN with an explicit reason rather than letting sklearn raise and
    # killing the whole metrics computation for what is often a legitimate
    # (if inconvenient) small/imbalanced evaluation set.
    if len(np.unique(y_true)) < 2:
        metrics["auroc"] = float("nan")
        metrics["log_loss"] = float("nan")
        metrics["_note"] = "AUROC/log_loss undefined: only one class present in y_true for this evaluation set."
    else:
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
        eps = 1e-7
        y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
        metrics["log_loss"] = log_loss(y_true, y_prob_clipped, labels=[0, 1])

    return metrics


@dataclass
class AggregatedMetric:
    mean: float
    std: float
    ci_low: float
    ci_high: float
    n_seeds: int
    per_seed_values: List[float] = field(default_factory=list)


def aggregate_across_seeds(
    per_seed_metrics: Dict[int, Dict[str, float]], confidence_level: float = 0.95
) -> Dict[str, AggregatedMetric]:
    """
    Aggregates per-seed metric dicts into mean/std/confidence-interval per
    metric name.

    Uses a Student's t critical value, not a normal approximation — with
    the seed counts this project realistically uses (3-10, per the
    reproduction protocol established earlier), a normal approximation
    understates the true interval width. Even with the t-correction, a
    3-seed interval is wide and should be read as a rough sense of
    spread, not a tight bound — reported honestly as such, not dressed up
    to look more precise than 3 samples can support. A single seed
    produces a point value with ci_low == ci_high == mean, not a
    fabricated interval.
    """
    seeds = sorted(per_seed_metrics.keys())
    if not seeds:
        raise ValueError("No per-seed metrics to aggregate.")

    metric_names = sorted(
        {
            k
            for s in seeds
            for k in per_seed_metrics[s].keys()
            if not k.startswith("_") and k != "n_samples" and isinstance(per_seed_metrics[s][k], (int, float))
        }
    )

    aggregated: Dict[str, AggregatedMetric] = {}
    for name in metric_names:
        values = [per_seed_metrics[s][name] for s in seeds if not np.isnan(per_seed_metrics[s].get(name, np.nan))]
        n = len(values)
        if n == 0:
            continue
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if n > 1 else 0.0
        if n > 1:
            t_crit = stats.t.ppf((1 + confidence_level) / 2, df=n - 1)
            margin = t_crit * std / np.sqrt(n)
            ci_low, ci_high = mean - margin, mean + margin
        else:
            ci_low = ci_high = mean
        aggregated[name] = AggregatedMetric(mean=mean, std=std, ci_low=ci_low, ci_high=ci_high, n_seeds=n, per_seed_values=values)

    return aggregated


def bootstrap_trajectory_metrics(
    model,
    trajectories: List[Trajectory],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    rng_seed: int = 0,
) -> Tuple[Dict[str, AggregatedMetric], Dict[str, np.ndarray]]:
    """
    The statistically appropriate confidence-interval method for a model
    whose fit() has no meaningful stochasticity — true of all four models
    in this package, see the E1 experiment design's "sanity checks"
    section. Resamples whole TRAJECTORIES (patients) with replacement,
    n_bootstrap times, and recomputes metrics on an ALREADY-FITTED model
    each time — never refits. Resampling whole trajectories, not
    individual windows, respects the actual unit of statistical
    independence: windows within one video are heavily correlated
    (sliding-window overlap), so window-level resampling would
    understate true uncertainty.

    Returns both the aggregated (mean/std/percentile-CI) summary AND the
    raw per-resample arrays — the raw arrays are required by
    paired_bootstrap_comparison, which needs the same resample indices
    (same rng_seed) used for two models to make a valid paired
    comparison.
    """
    rng = np.random.RandomState(rng_seed)
    n = len(trajectories)
    per_resample_metrics: Dict[str, List[float]] = {}

    for _ in range(n_bootstrap):
        resample_indices = rng.randint(0, n, size=n)
        resampled = [trajectories[i] for i in resample_indices]
        predictions = model.predict(resampled)
        m = compute_metrics(resampled, predictions)
        for k, v in m.items():
            if isinstance(v, float) and not k.startswith("_") and not np.isnan(v):
                per_resample_metrics.setdefault(k, []).append(v)

    raw = {k: np.asarray(v) for k, v in per_resample_metrics.items()}
    aggregated: Dict[str, AggregatedMetric] = {}
    alpha = 1 - confidence_level
    for name, values in raw.items():
        if len(values) == 0:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        ci_low = float(np.percentile(values, 100 * alpha / 2))
        ci_high = float(np.percentile(values, 100 * (1 - alpha / 2)))
        aggregated[name] = AggregatedMetric(
            mean=mean, std=std, ci_low=ci_low, ci_high=ci_high, n_seeds=len(values), per_seed_values=values.tolist()
        )

    return aggregated, raw


def paired_bootstrap_comparison(
    raw_a: Dict[str, np.ndarray],
    raw_b: Dict[str, np.ndarray],
    metric_name: str,
    confidence_level: float = 0.95,
) -> Dict[str, float]:
    """
    Compares two models' bootstrap distributions for one metric, PAIRED
    by resample index. `raw_a` and `raw_b` must come from
    bootstrap_trajectory_metrics() calls using the SAME n_bootstrap and
    SAME rng_seed over the SAME trajectory list, so resample k for model
    A and resample k for model B used the identical resampled patient
    set — that shared randomness is what makes this paired rather than
    an independent-samples comparison, and is the whole reason
    bootstrap_trajectory_metrics takes an explicit rng_seed instead of
    drawing fresh randomness each call.

    Returns:
      mean_difference — mean(B - A) across paired resamples. The
        primary, most directly interpretable effect size for a bounded
        proportion-like metric (e.g. "accuracy improves by 3.2 points").
      ci_low / ci_high — percentile CI of the paired difference.
      cohens_d — mean_difference / std(B - A). Standard, for
        comparability with other reported effect sizes; assumes the
        paired-difference distribution is reasonably close to normal —
        worth checking (e.g. a histogram), not assuming, for small
        n_bootstrap.
      wilcoxon_p — two-sided Wilcoxon signed-rank test p-value on the
        paired differences. Distribution-free, the standard choice for
        paired samples without assuming normality.
    """
    a = raw_a[metric_name]
    b = raw_b[metric_name]
    if len(a) != len(b):
        raise ValueError(
            f"Bootstrap arrays have different lengths ({len(a)} vs {len(b)}) for "
            f"metric '{metric_name}' — they must come from the same n_bootstrap "
            f"and rng_seed to be validly paired."
        )

    diff = b - a
    mean_diff = float(diff.mean())
    std_diff = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    alpha = 1 - confidence_level
    ci_low = float(np.percentile(diff, 100 * alpha / 2))
    ci_high = float(np.percentile(diff, 100 * (1 - alpha / 2)))
    cohens_d = mean_diff / std_diff if std_diff > 0 else float("nan")

    if np.allclose(a, b):
        # scipy.stats.wilcoxon raises on all-zero differences rather than
        # returning a (correct) p=1.0 — handled explicitly so identical
        # bootstrap distributions don't crash the comparison.
        wilcoxon_p = 1.0
    else:
        _, wilcoxon_p = stats.wilcoxon(b, a)

    return {
        "mean_difference": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "cohens_d": cohens_d,
        "wilcoxon_p": float(wilcoxon_p),
        "n_bootstrap": len(diff),
    }


def holm_bonferroni(p_values: Dict[str, float], alpha: float = 0.05) -> Dict[str, bool]:
    """
    Holm-Bonferroni step-down correction for multiple comparisons.
    Returns, per comparison name, whether it remains significant at
    family-wise alpha after correction.

    Apply this across a PRE-REGISTERED, small set of comparisons decided
    before looking at results — for E1, the three adjacent ablation-ladder
    steps (identity_dynamics vs base_rate, persistence vs
    identity_dynamics, linear_ssm vs persistence), not all six possible
    pairwise comparisons among four models, which was never the
    pre-registered question and would just be multiple-testing fishing.
    """
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    significant: Dict[str, bool] = {}
    still_rejecting = True
    for k, (name, p) in enumerate(items):
        threshold = alpha / (m - k)
        if still_rejecting and p <= threshold:
            significant[name] = True
        else:
            still_rejecting = False
            significant[name] = False
    return significant

    return aggregated
