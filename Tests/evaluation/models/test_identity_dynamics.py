"""
Unit tests for evaluation.models.identity_dynamics.IdentityDynamicsModel.

Synthetic Trajectory objects only — no real embedding cache, no GPU.
Beyond the standard interface/edge-case tests (mirroring
test_persistence.py's structure), this file's most important test is
test_predictions_are_invariant_to_window_order: the direct, executable
version of this model's central design claim — that it uses only each
window's own embedding, never any other window's.
"""

import numpy as np
import pytest
import torch

from evaluation.models.identity_dynamics import IdentityDynamicsModel
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
# interface contract
# --------------------------------------------------------------------------

def test_predict_before_fit_raises():
    model = IdentityDynamicsModel()
    traj = make_trajectory("v1", embeddings=[[0.0], [1.0]], consistency_flags=[0, 1])
    with pytest.raises(RuntimeError):
        model.predict([traj])


def test_predict_output_is_aligned_and_bounded():
    trajectories = [make_trajectory("v1", embeddings=[[0.0], [5.0], [5.1]], consistency_flags=[0, 1, 0])]
    model = IdentityDynamicsModel()
    model.fit(trajectories)
    predictions = model.predict(trajectories)

    assert len(predictions) == 1
    pred = predictions[0]
    assert pred.video_name == "v1"
    assert pred.window_starts == trajectories[0].window_starts
    assert torch.all(pred.consistency_flag_prob >= 0.0)
    assert torch.all(pred.consistency_flag_prob <= 1.0)


# --------------------------------------------------------------------------
# the central design claim: no cross-window information is used, at all
# --------------------------------------------------------------------------

def test_predictions_are_invariant_to_window_order():
    rng = np.random.RandomState(0)
    embeddings = rng.randn(10, 4).tolist()
    flags = (rng.rand(10) < 0.4).astype(int).tolist()
    window_starts = list(range(10))

    model = IdentityDynamicsModel()
    fit_traj = make_trajectory("v1", embeddings, flags, window_starts)
    model.fit([fit_traj] * 5)  # repeated so the classifier has something non-trivial to learn

    original = make_trajectory("v1", embeddings, flags, window_starts)
    permutation = list(range(10))
    rng.shuffle(permutation)
    shuffled = make_trajectory(
        "v1",
        [embeddings[i] for i in permutation],
        [flags[i] for i in permutation],
        [window_starts[i] for i in permutation],
    )

    pred_original = model.predict([original])[0]
    pred_shuffled = model.predict([shuffled])[0]

    original_by_ws = dict(zip(pred_original.window_starts, pred_original.consistency_flag_prob.tolist()))
    shuffled_by_ws = dict(zip(pred_shuffled.window_starts, pred_shuffled.consistency_flag_prob.tolist()))

    for ws in window_starts:
        assert shuffled_by_ws[ws] == pytest.approx(original_by_ws[ws], abs=1e-5), (
            f"window_start={ws} got a different prediction depending on its position in the "
            f"trajectory — IdentityDynamicsModel must be invariant to window order by design."
        )


# --------------------------------------------------------------------------
# the scientific claim: content, not change, should be usable
# --------------------------------------------------------------------------

def test_fit_learns_content_based_separation():
    # Unlike PersistenceModel's synthetic test (a large JUMP predicts a
    # transition), here the ABSOLUTE embedding value determines the
    # label, with no relationship to the previous window at all — a
    # signal PersistenceModel structurally cannot use well, and
    # IdentityDynamics should have no trouble with, since it is designed
    # to use content.
    rng = np.random.RandomState(0)
    trajectories = []
    for v in range(20):
        embeddings, flags = [], []
        for _ in range(10):
            value = rng.uniform(-1, 1)
            embeddings.append([value])
            flags.append(1 if value > 0 else 0)
        trajectories.append(make_trajectory(f"v{v}", embeddings, flags))

    model = IdentityDynamicsModel()
    model.fit(trajectories)
    predictions = model.predict(trajectories)

    y_true, y_prob = [], []
    for traj, pred in zip(trajectories, predictions):
        y_true.extend(traj.consistency_flag.tolist())
        y_prob.extend(pred.consistency_flag_prob.tolist())
    y_pred = (np.asarray(y_prob) >= 0.5).astype(int)
    accuracy = float((np.asarray(y_true) == y_pred).mean())

    assert accuracy > 0.9, f"Expected near-perfect separation on an unambiguous content-based signal, got {accuracy}"


# --------------------------------------------------------------------------
# documented edge cases
# --------------------------------------------------------------------------

def test_fit_with_single_class_falls_back_to_base_rate_without_crashing():
    trajectories = [
        make_trajectory("v1", embeddings=[[0.0], [1.0], [50.0]], consistency_flags=[0, 0, 0]),
        make_trajectory("v2", embeddings=[[0.0], [2.0], [60.0]], consistency_flags=[0, 0, 0]),
    ]
    model = IdentityDynamicsModel()
    model.fit(trajectories)  # must not raise

    assert model._logreg is None
    predictions = model.predict(trajectories)
    for pred in predictions:
        assert pred.consistency_flag_prob.tolist() == pytest.approx([0.0] * len(pred.consistency_flag_prob))


def test_fit_with_no_trajectories_falls_back_to_half():
    model = IdentityDynamicsModel()
    model.fit([])
    assert model._base_rate == pytest.approx(0.5)
    predictions = model.predict([make_trajectory("v1", embeddings=[[0.0]], consistency_flags=[0])])
    assert predictions[0].consistency_flag_prob[0].item() == pytest.approx(0.5)


# --------------------------------------------------------------------------
# save/load round-trip
# --------------------------------------------------------------------------

def test_save_load_round_trip_gives_identical_predictions(tmp_path):
    rng = np.random.RandomState(1)
    trajectories = []
    for v in range(10):
        embeddings, flags = [], []
        for _ in range(6):
            value = rng.uniform(-2, 2)
            embeddings.append([value, -value])
            flags.append(1 if value > 0 else 0)
        trajectories.append(make_trajectory(f"v{v}", embeddings, flags))

    model = IdentityDynamicsModel()
    model.fit(trajectories)
    before = model.predict(trajectories)

    model.save(tmp_path)
    reloaded = IdentityDynamicsModel.load(tmp_path)
    after = reloaded.predict(trajectories)

    for pred_before, pred_after in zip(before, after):
        torch.testing.assert_close(pred_before.consistency_flag_prob, pred_after.consistency_flag_prob)


def test_save_load_round_trip_without_classifier(tmp_path):
    trajectories = [make_trajectory("v1", embeddings=[[0.0], [1.0]], consistency_flags=[0, 0])]
    model = IdentityDynamicsModel()
    model.fit(trajectories)
    assert model._logreg is None

    model.save(tmp_path)
    reloaded = IdentityDynamicsModel.load(tmp_path)
    assert reloaded._logreg is None
    assert reloaded._base_rate == pytest.approx(model._base_rate)
