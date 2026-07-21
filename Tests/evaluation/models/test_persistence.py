"""
Unit tests for evaluation.models.persistence.PersistenceModel.

Uses hand-constructed synthetic Trajectory objects only — no real
embedding cache, no GPU, no trained encoder. What's under test is the
model's own logic (distance computation, the fit/predict/save/load
contract, and its documented edge cases), not anything about what real
embeddings actually contain — that is what evaluation.explore and running
this against a real cache are for, separately.
"""

import numpy as np
import pytest
import torch

from evaluation.models.persistence import PersistenceModel
from evaluation.trajectory import Trajectory


def make_trajectory(video_name, embeddings, consistency_flags, window_starts=None):
    embeddings_t = torch.as_tensor(embeddings, dtype=torch.float32)
    flags_t = torch.as_tensor(consistency_flags, dtype=torch.long)
    T = embeddings_t.shape[0]
    return Trajectory(
        video_name=video_name,
        window_starts=window_starts if window_starts is not None else list(range(T)),
        embeddings=embeddings_t,
        consistency_flag=flags_t,
        first_frame_phase=torch.zeros(T, dtype=torch.long),
        last_frame_phase=torch.zeros(T, dtype=torch.long),
    )


# --------------------------------------------------------------------------
# distance computation: the actual mathematical claim of this model
# --------------------------------------------------------------------------

def test_consecutive_distances_hand_computed():
    traj = make_trajectory("v1", embeddings=[[0.0, 0.0], [3.0, 4.0], [3.0, 4.0]], consistency_flags=[0, 1, 0])
    d = PersistenceModel._consecutive_distances(traj)
    # ||(3,4) - (0,0)|| = 5.0 ; ||(3,4) - (3,4)|| = 0.0
    assert d.shape == (2,)
    np.testing.assert_allclose(d, [5.0, 0.0], atol=1e-5)


def test_single_window_trajectory_has_no_distances():
    traj = make_trajectory("v1", embeddings=[[1.0, 2.0]], consistency_flags=[0])
    d = PersistenceModel._consecutive_distances(traj)
    assert d.shape == (0,)


# --------------------------------------------------------------------------
# interface contract
# --------------------------------------------------------------------------

def test_predict_before_fit_raises():
    model = PersistenceModel()
    traj = make_trajectory("v1", embeddings=[[0.0], [1.0]], consistency_flags=[0, 1])
    with pytest.raises(RuntimeError):
        model.predict([traj])


def test_predict_output_is_aligned_and_bounded():
    trajectories = [make_trajectory("v1", embeddings=[[0.0], [5.0], [5.1]], consistency_flags=[0, 1, 0])]
    model = PersistenceModel()
    model.fit(trajectories)
    predictions = model.predict(trajectories)

    assert len(predictions) == 1
    pred = predictions[0]
    assert pred.video_name == "v1"
    assert pred.window_starts == trajectories[0].window_starts
    assert torch.all(pred.consistency_flag_prob >= 0.0)
    assert torch.all(pred.consistency_flag_prob <= 1.0)


# --------------------------------------------------------------------------
# the scientific claim: large jumps should predict transitions
# --------------------------------------------------------------------------

def test_fit_learns_separable_relationship():
    # Construct trajectories where large embedding jumps ALWAYS coincide
    # with a transition and small jumps NEVER do — an unambiguous
    # synthetic signal the logistic fit should recover with high
    # confidence. This is the actual claim of the model, made checkable.
    rng = np.random.RandomState(0)
    trajectories = []
    for v in range(20):
        embeddings, flags = [[0.0]], [0]
        value = 0.0
        for _ in range(1, 10):
            if rng.rand() < 0.3:
                value += 10.0  # big jump
                flags.append(1)
            else:
                value += 0.01  # tiny jump
                flags.append(0)
            embeddings.append([value])
        trajectories.append(make_trajectory(f"v{v}", embeddings, flags))

    model = PersistenceModel()
    model.fit(trajectories)
    predictions = model.predict(trajectories)

    y_true, y_prob = [], []
    for traj, pred in zip(trajectories, predictions):
        y_true.extend(traj.consistency_flag[1:].tolist())  # skip first-window fallback windows
        y_prob.extend(pred.consistency_flag_prob[1:].tolist())
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= 0.5).astype(int)
    accuracy = float((y_true == y_pred).mean())

    assert accuracy > 0.95, f"Expected near-perfect separation on an unambiguous synthetic signal, got {accuracy}"


# --------------------------------------------------------------------------
# documented edge cases
# --------------------------------------------------------------------------

def test_first_window_uses_base_rate_fallback():
    trajectories = [
        make_trajectory("v1", embeddings=[[0.0], [1.0], [2.0]], consistency_flags=[1, 0, 1]),
        make_trajectory("v2", embeddings=[[0.0], [1.0], [2.0]], consistency_flags=[0, 0, 1]),
    ]
    model = PersistenceModel()
    model.fit(trajectories)
    predictions = model.predict(trajectories)

    for pred in predictions:
        assert pred.consistency_flag_prob[0].item() == pytest.approx(model._base_rate)


def test_single_window_trajectory_predicts_base_rate_only():
    trajectories = [
        make_trajectory("v1", embeddings=[[0.0], [10.0]], consistency_flags=[0, 1]),  # gives fit() something to learn from
    ]
    model = PersistenceModel()
    model.fit(trajectories)

    lone_window = [make_trajectory("v2", embeddings=[[3.0]], consistency_flags=[0])]
    predictions = model.predict(lone_window)

    assert len(predictions[0].consistency_flag_prob) == 1
    assert predictions[0].consistency_flag_prob[0].item() == pytest.approx(model._base_rate)


def test_fit_with_single_class_falls_back_to_base_rate_without_crashing():
    trajectories = [
        make_trajectory("v1", embeddings=[[0.0], [1.0], [50.0]], consistency_flags=[0, 0, 0]),
        make_trajectory("v2", embeddings=[[0.0], [2.0], [60.0]], consistency_flags=[0, 0, 0]),
    ]
    model = PersistenceModel()
    model.fit(trajectories)  # must not raise

    assert model._logreg is None
    predictions = model.predict(trajectories)
    for pred in predictions:
        assert torch.all(pred.consistency_flag_prob == pytest.approx(0.0))


# --------------------------------------------------------------------------
# save/load round-trip
# --------------------------------------------------------------------------

def test_save_load_round_trip_gives_identical_predictions(tmp_path):
    trajectories = [
        make_trajectory(f"v{i}", embeddings=[[0.0], [1.0], [10.0], [10.1]], consistency_flags=[0, 0, 1, 0])
        for i in range(5)
    ]

    model = PersistenceModel()
    model.fit(trajectories)
    before = model.predict(trajectories)

    model.save(tmp_path)
    reloaded = PersistenceModel.load(tmp_path)
    after = reloaded.predict(trajectories)

    for pred_before, pred_after in zip(before, after):
        torch.testing.assert_close(pred_before.consistency_flag_prob, pred_after.consistency_flag_prob)


def test_save_load_round_trip_without_logreg(tmp_path):
    # Covers the branch where fit() never trained a logreg at all
    # (single-class training data) — save/load must still round-trip.
    trajectories = [make_trajectory("v1", embeddings=[[0.0], [1.0]], consistency_flags=[0, 0])]
    model = PersistenceModel()
    model.fit(trajectories)
    assert model._logreg is None

    model.save(tmp_path)
    reloaded = PersistenceModel.load(tmp_path)
    assert reloaded._logreg is None
    assert reloaded._base_rate == pytest.approx(model._base_rate)
