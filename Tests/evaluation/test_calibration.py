"""
Unit tests for evaluation.calibration — the Brier/ECE/reliability-bin
helpers factored out for the HMM k=7 selection protocol
(run_hmm_evaluation.py) and the k-step alignment diagnostic
(run_hmm_k_step_alignment_check.py). Synthetic arrays only.
"""

import numpy as np
import pytest
import torch

from evaluation.calibration import (
    calibration_bins,
    expected_calibration_error,
    flatten_predictions,
    score_probabilistic,
)
from evaluation.trajectory import Trajectory, TrajectoryPrediction


def make_trajectory(video_name, consistency_flags):
    flags_t = torch.as_tensor(consistency_flags, dtype=torch.long)
    T = flags_t.shape[0]
    return Trajectory(
        video_name=video_name,
        window_starts=list(range(T)),
        embeddings=torch.zeros((T, 2)),
        consistency_flag=flags_t,
        first_frame_phase=torch.zeros(T, dtype=torch.long),
        last_frame_phase=torch.zeros(T, dtype=torch.long),
    )


def make_prediction(video_name, window_starts, probs):
    return TrajectoryPrediction(
        video_name=video_name,
        window_starts=window_starts,
        consistency_flag_prob=torch.as_tensor(probs, dtype=torch.float32),
    )


def test_score_probabilistic_perfect_predictions_zero_brier():
    y_true = np.array([0, 1, 0, 1, 1])
    y_prob = y_true.astype(np.float64)
    result = score_probabilistic(y_true, y_prob, n_bins=5)
    assert result["brier"] == pytest.approx(0.0, abs=1e-12)
    assert result["auroc"] == pytest.approx(1.0)


def test_score_probabilistic_constant_prediction_matches_baseline_exactly():
    y_true = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 0])
    p = float(y_true.mean())
    y_prob = np.full(len(y_true), p)
    result = score_probabilistic(y_true, y_prob, n_bins=10)
    assert result["brier"] == pytest.approx(result["brier_baseline_constant_rate"], abs=1e-12)


def test_calibration_bins_cover_all_windows_exactly_once():
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, size=200)
    y_prob = rng.rand(200)
    bins = calibration_bins(y_true, y_prob, n_bins=10)
    assert sum(b["n"] for b in bins) == 200


def test_expected_calibration_error_zero_when_perfectly_calibrated():
    # Construct bins by hand where mean_predicted == observed_frequency in
    # every non-empty bin -- ECE must be exactly 0, independent of Brier.
    bins = [
        {"bin_lo": 0.0, "bin_hi": 0.5, "n": 4, "mean_predicted": 0.25, "observed_frequency": 0.25},
        {"bin_lo": 0.5, "bin_hi": 1.0, "n": 6, "mean_predicted": 0.75, "observed_frequency": 0.75},
    ]
    assert expected_calibration_error(bins, n_total=10) == pytest.approx(0.0, abs=1e-12)


def test_expected_calibration_error_matches_hand_computation():
    bins = [
        {"bin_lo": 0.0, "bin_hi": 0.5, "n": 4, "mean_predicted": 0.10, "observed_frequency": 0.30},
        {"bin_lo": 0.5, "bin_hi": 1.0, "n": 6, "mean_predicted": 0.90, "observed_frequency": 0.60},
    ]
    expected = (4 / 10) * abs(0.10 - 0.30) + (6 / 10) * abs(0.90 - 0.60)
    assert expected_calibration_error(bins, n_total=10) == pytest.approx(expected, abs=1e-12)


def test_score_probabilistic_auroc_nan_when_single_class_present():
    y_true = np.zeros(20, dtype=np.int64)
    y_prob = np.random.RandomState(0).rand(20)
    result = score_probabilistic(y_true, y_prob, n_bins=5)
    assert np.isnan(result["auroc"])
    assert "_auroc_note" in result


def test_flatten_predictions_raises_on_length_mismatch():
    trajectories = [make_trajectory("v0", [0, 1, 0])]
    predictions = []
    with pytest.raises(ValueError):
        flatten_predictions(trajectories, predictions)


def test_flatten_predictions_raises_on_window_starts_mismatch():
    trajectories = [make_trajectory("v0", [0, 1, 0])]
    predictions = [make_prediction("v0", [0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4])]
    with pytest.raises(ValueError):
        flatten_predictions(trajectories, predictions)


def test_flatten_predictions_matches_by_position_not_only_content():
    traj = make_trajectory("v0", [0, 1, 1, 0])
    pred = make_prediction("v0", [0, 1, 2, 3], [0.9, 0.1, 0.2, 0.8])
    y_true, y_prob = flatten_predictions([traj], [pred])
    np.testing.assert_array_equal(y_true, [0, 1, 1, 0])
    np.testing.assert_allclose(y_prob, [0.9, 0.1, 0.2, 0.8])
