"""
Unit tests for the bootstrap/comparison/correction machinery added to
evaluation.metrics for the E1 experiment. This file's first test is a
direct regression test for a real bug found while designing patient-level
bootstrap: predictions used to be matched to trajectories by video_name,
which silently collides the moment the same video_name appears twice —
exactly what sampling with replacement does routinely.
"""

import numpy as np
import pytest
import torch

from evaluation.metrics import (
    bootstrap_trajectory_metrics,
    compute_metrics,
    holm_bonferroni,
    paired_bootstrap_comparison,
)
from evaluation.trajectory import Trajectory, TrajectoryPrediction


def make_trajectory(video_name, embeddings, consistency_flags):
    embeddings_t = torch.as_tensor(embeddings, dtype=torch.float32)
    flags_t = torch.as_tensor(consistency_flags, dtype=torch.long)
    T = embeddings_t.shape[0]
    return Trajectory(
        video_name=video_name,
        window_starts=list(range(T)),
        embeddings=embeddings_t,
        consistency_flag=flags_t,
        first_frame_phase=torch.zeros(T, dtype=torch.long),
        last_frame_phase=torch.zeros(T, dtype=torch.long),
    )


class _ConstantModel:
    """Minimal fake Model for testing bootstrap machinery in isolation —
    predicts a fixed probability regardless of input, so exactly what
    compute_metrics should produce for any resample is known."""

    def predict(self, trajectories):
        return [
            TrajectoryPrediction(
                video_name=t.video_name, window_starts=t.window_starts, consistency_flag_prob=torch.full((len(t),), 0.7)
            )
            for t in trajectories
        ]


# --------------------------------------------------------------------------
# regression test: the actual bug this whole turn's design work found
# --------------------------------------------------------------------------

def test_compute_metrics_handles_duplicate_video_names_from_bootstrap():
    # Two trajectories sharing a video_name (as bootstrap-with-replacement
    # produces routinely), with DIFFERENT true labels. Both must be
    # counted — the old video_name-keyed dict lookup would have silently
    # dropped one.
    traj_a = make_trajectory("v1", embeddings=[[0.0]], consistency_flags=[1])
    traj_b = make_trajectory("v1", embeddings=[[0.0]], consistency_flags=[0])
    pred_a = TrajectoryPrediction(video_name="v1", window_starts=traj_a.window_starts, consistency_flag_prob=torch.tensor([0.9]))
    pred_b = TrajectoryPrediction(video_name="v1", window_starts=traj_b.window_starts, consistency_flag_prob=torch.tensor([0.1]))

    metrics = compute_metrics([traj_a, traj_b], [pred_a, pred_b])
    assert metrics["n_samples"] == 2
    # accuracy: pred_a=0.9->1 vs true=1 (correct), pred_b=0.1->0 vs true=0 (correct) -> 100%
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_flatten_raises_on_length_mismatch_not_silent_truncation():
    traj = make_trajectory("v1", embeddings=[[0.0]], consistency_flags=[1])
    pred = TrajectoryPrediction(video_name="v1", window_starts=traj.window_starts, consistency_flag_prob=torch.tensor([0.5]))
    with pytest.raises(ValueError):
        compute_metrics([traj, traj], [pred])  # 2 trajectories, 1 prediction


# --------------------------------------------------------------------------
# bootstrap_trajectory_metrics
# --------------------------------------------------------------------------

def test_bootstrap_trajectory_metrics_runs_and_produces_sensible_output():
    trajectories = [
        make_trajectory("v1", [[0.0]], [1]),
        make_trajectory("v2", [[0.0]], [0]),
        make_trajectory("v3", [[0.0]], [1]),
    ]
    aggregated, raw = bootstrap_trajectory_metrics(_ConstantModel(), trajectories, n_bootstrap=200, rng_seed=0)

    assert "accuracy" in aggregated
    assert 0.0 <= aggregated["accuracy"].mean <= 1.0
    assert aggregated["accuracy"].n_seeds == len(raw["accuracy"])
    assert len(raw["accuracy"]) <= 200  # some resamples may be all-one-class and get skipped for AUROC/log_loss, but not accuracy


def test_bootstrap_trajectory_metrics_is_reproducible_given_same_seed():
    trajectories = [make_trajectory(f"v{i}", [[float(i)]], [i % 2]) for i in range(5)]
    _, raw1 = bootstrap_trajectory_metrics(_ConstantModel(), trajectories, n_bootstrap=100, rng_seed=42)
    _, raw2 = bootstrap_trajectory_metrics(_ConstantModel(), trajectories, n_bootstrap=100, rng_seed=42)
    np.testing.assert_array_equal(raw1["accuracy"], raw2["accuracy"])


def test_bootstrap_trajectory_metrics_differs_across_different_seeds():
    trajectories = [make_trajectory(f"v{i}", [[float(i)]], [i % 2]) for i in range(10)]
    _, raw1 = bootstrap_trajectory_metrics(_ConstantModel(), trajectories, n_bootstrap=100, rng_seed=1)
    _, raw2 = bootstrap_trajectory_metrics(_ConstantModel(), trajectories, n_bootstrap=100, rng_seed=2)
    assert not np.array_equal(raw1["accuracy"], raw2["accuracy"])


# --------------------------------------------------------------------------
# paired_bootstrap_comparison
# --------------------------------------------------------------------------

def test_paired_bootstrap_comparison_detects_known_shift():
    rng = np.random.RandomState(0)
    raw_a = {"accuracy": rng.normal(0.5, 0.02, size=500)}
    raw_b = {"accuracy": raw_a["accuracy"] + 0.1 + rng.normal(0, 0.005, size=500)}

    result = paired_bootstrap_comparison(raw_a, raw_b, "accuracy")

    assert result["mean_difference"] == pytest.approx(0.1, abs=0.01)
    assert result["ci_low"] > 0  # CI should exclude zero given a clear, consistent shift
    assert result["wilcoxon_p"] < 0.01
    assert result["cohens_d"] > 1.0  # a consistent 0.1 shift against ~0.005-0.02 scale noise is a large effect


def test_paired_bootstrap_comparison_identical_distributions_gives_no_effect():
    values = np.random.RandomState(0).normal(0.6, 0.05, size=200)
    result = paired_bootstrap_comparison({"accuracy": values}, {"accuracy": values}, "accuracy")

    assert result["mean_difference"] == pytest.approx(0.0)
    assert result["ci_low"] == pytest.approx(0.0)
    assert result["ci_high"] == pytest.approx(0.0)
    assert result["wilcoxon_p"] == pytest.approx(1.0)


def test_paired_bootstrap_comparison_raises_on_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap_comparison({"accuracy": np.zeros(10)}, {"accuracy": np.zeros(20)}, "accuracy")


# --------------------------------------------------------------------------
# holm_bonferroni
# --------------------------------------------------------------------------

def test_holm_bonferroni_known_case():
    # alpha=0.05, m=3: sorted p-values 0.001, 0.04, 0.5
    # thresholds: 0.05/3=0.0167, 0.05/2=0.025, 0.05/1=0.05
    # 0.001 <= 0.0167 -> reject; 0.04 > 0.025 -> fail, stop; 0.5 -> fail (already stopped)
    p_values = {"a": 0.001, "b": 0.04, "c": 0.5}
    result = holm_bonferroni(p_values, alpha=0.05)
    assert result == {"a": True, "b": False, "c": False}


def test_holm_bonferroni_all_significant():
    p_values = {"a": 0.0001, "b": 0.0002, "c": 0.0003}
    result = holm_bonferroni(p_values, alpha=0.05)
    assert all(result.values())


def test_holm_bonferroni_none_significant():
    p_values = {"a": 0.9, "b": 0.8, "c": 0.7}
    result = holm_bonferroni(p_values, alpha=0.05)
    assert not any(result.values())
