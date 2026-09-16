"""
Unit tests for evaluation.models.gru.GRUDynamicsModel.

Synthetic Trajectory objects only — no real embedding cache, no GPU, no
"real" training (only tiny synthetic fits, 1-2 epochs, hidden_dim=8, exactly
the same kind of quick .fit() call the other three models' test files
already make freely). Embeddings here must be 512-dim, since (unlike
persistence.py/linear_ssm.py's tests, which use toy 1-2D vectors for
hand-computable arithmetic) GRUDynamicsModel hardcodes input_dim=512 to
match the real ResNet18 cache — there is nothing to hand-verify for a
learned non-linear transform, so this file's emphasis is different from the
other three: causality, determinism, shape/finiteness, and the save/load
contract, per the migration study's Section 9 test plan.

This file's two most important tests are test_gru_is_causal and
test_gru_no_future_information_in_forecast — direct, executable proof that
neither predict_mode ever lets a future window influence an earlier
prediction, mirroring the same testing philosophy already used for
IdentityDynamicsModel's test_predictions_are_invariant_to_window_order.
"""

import numpy as np
import pytest
import torch

from evaluation.models.gru import GRUDynamicsModel
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


def _random_trajectory(video_name, T, seed, D=512):
    rng = np.random.RandomState(seed)
    embeddings = rng.randn(T, D).tolist()
    flags = (rng.rand(T) < 0.4).astype(int).tolist()
    return make_trajectory(video_name, embeddings, flags)


# --------------------------------------------------------------------------
# interface contract
# --------------------------------------------------------------------------

def test_invalid_predict_mode_raises():
    with pytest.raises(ValueError):
        GRUDynamicsModel(predict_mode="bogus")


def test_predict_before_fit_raises():
    model = GRUDynamicsModel(predict_mode="residual")
    traj = make_trajectory("v1", embeddings=[[0.0] * 512, [1.0] * 512], consistency_flags=[0, 1])
    with pytest.raises(RuntimeError):
        model.predict([traj])


def test_forecast_from_embedding_before_fit_raises():
    model = GRUDynamicsModel(predict_mode="residual")
    with pytest.raises(RuntimeError):
        model.forecast_from_embedding(torch.zeros(512), steps=1)


@pytest.mark.parametrize("mode", ["residual", "direct"])
def test_predict_output_is_aligned_and_bounded(mode):
    trajectories = [_random_trajectory("v1", T=5, seed=0)]
    model = GRUDynamicsModel(predict_mode=mode, hidden_dim=8, max_epochs=1, seed=0)
    model.fit(trajectories)
    predictions = model.predict(trajectories)

    assert len(predictions) == 1
    pred = predictions[0]
    assert pred.video_name == "v1"
    assert pred.window_starts == trajectories[0].window_starts
    assert pred.consistency_flag_prob.shape == (5,)
    assert torch.all(pred.consistency_flag_prob >= 0.0)
    assert torch.all(pred.consistency_flag_prob <= 1.0)
    assert torch.isfinite(pred.consistency_flag_prob).all()


# --------------------------------------------------------------------------
# causality: the central design claim for BOTH predict_modes
# --------------------------------------------------------------------------

def test_gru_is_causal():
    # Two trajectories sharing an identical prefix, diverging afterward.
    # h_t for t within the shared prefix must be IDENTICAL between the two —
    # any difference would prove information from the diverging suffix
    # leaked backward into an earlier hidden state (impossible for a
    # correctly-unidirectional GRU, but exactly the kind of bug a
    # bidirectional=True typo would introduce silently).
    rng = np.random.RandomState(0)
    prefix = rng.randn(5, 512).tolist()
    flags_prefix = (rng.rand(5) < 0.4).astype(int).tolist()
    suffix_a = rng.randn(4, 512).tolist()
    suffix_b = rng.randn(4, 512).tolist()
    flags_a = (rng.rand(4) < 0.4).astype(int).tolist()
    flags_b = (rng.rand(4) < 0.4).astype(int).tolist()

    traj_a = make_trajectory("va", prefix + suffix_a, flags_prefix + flags_a)
    traj_b = make_trajectory("vb", prefix + suffix_b, flags_prefix + flags_b)

    model = GRUDynamicsModel(predict_mode="direct", hidden_dim=8, max_epochs=1, seed=0)
    model.fit([traj_a, traj_b])
    model._net.eval()

    with torch.no_grad():
        h_a, _, _ = model._net(model._standardize(traj_a.embeddings))
        h_b, _, _ = model._net(model._standardize(traj_b.embeddings))

    torch.testing.assert_close(h_a[:, :5, :], h_b[:, :5, :], atol=1e-6, rtol=1e-6)


def test_gru_no_future_information_in_forecast():
    rng = np.random.RandomState(1)
    prefix = rng.randn(6, 512).tolist()
    flags = (rng.rand(9) < 0.4).astype(int).tolist()
    tail_a = rng.randn(3, 512).tolist()
    tail_b = rng.randn(3, 512).tolist()

    traj_a = make_trajectory("va", prefix + tail_a, flags)
    traj_b = make_trajectory("vb", prefix + tail_b, flags)

    model = GRUDynamicsModel(predict_mode="residual", hidden_dim=8, max_epochs=1, seed=0)
    model.fit([traj_a, traj_b])
    model._net.eval()

    with torch.no_grad():
        _, forecast_a, _ = model._net(model._standardize(traj_a.embeddings))
        _, forecast_b, _ = model._net(model._standardize(traj_b.embeddings))

    # forecast[:, t] predicts z_{t+1} from h_t alone, which depends only on
    # z_0..z_t. The shared prefix covers indices 0..5 (6 windows), so
    # forecast[:, t] for t=0..4 (predicting z_1..z_5) must be identical
    # between the two trajectories regardless of what follows at index 6+.
    torch.testing.assert_close(forecast_a[:, :5, :], forecast_b[:, :5, :], atol=1e-6, rtol=1e-6)


# --------------------------------------------------------------------------
# determinism — a genuinely new requirement for this package: none of the
# four closed-form E1 models needed a dedicated test for this (their
# determinism was verified empirically against real data in a separate
# session, not via pytest) — a gradient-trained model needs it explicitly.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["residual", "direct"])
def test_gru_forward_deterministic_given_seed(mode):
    traj = _random_trajectory("v1", T=6, seed=0)
    model = GRUDynamicsModel(predict_mode=mode, hidden_dim=8, max_epochs=1, seed=0)
    model.fit([traj])

    pred1 = model.predict([traj])[0]
    pred2 = model.predict([traj])[0]
    torch.testing.assert_close(pred1.consistency_flag_prob, pred2.consistency_flag_prob, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("mode", ["residual", "direct"])
def test_gru_fit_is_reproducible_given_same_seed(mode):
    trajectories = [_random_trajectory(f"v{i}", T=6, seed=i) for i in range(3)]

    model1 = GRUDynamicsModel(predict_mode=mode, hidden_dim=8, max_epochs=2, seed=42)
    model1.fit(trajectories)
    pred1 = model1.predict(trajectories)

    model2 = GRUDynamicsModel(predict_mode=mode, hidden_dim=8, max_epochs=2, seed=42)
    model2.fit(trajectories)
    pred2 = model2.predict(trajectories)

    for p1, p2 in zip(pred1, pred2):
        torch.testing.assert_close(p1.consistency_flag_prob, p2.consistency_flag_prob, atol=1e-5, rtol=1e-4)


# --------------------------------------------------------------------------
# shapes, finiteness, variable length
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["residual", "direct"])
def test_handles_variable_length_trajectories_in_one_batch(mode):
    trajectories = [_random_trajectory(f"v{i}", T=T, seed=i) for i, T in enumerate([2, 15, 4, 30])]
    model = GRUDynamicsModel(predict_mode=mode, hidden_dim=8, max_epochs=1, seed=0)
    model.fit(trajectories)
    predictions = model.predict(trajectories)
    for traj, pred in zip(trajectories, predictions):
        assert pred.consistency_flag_prob.shape == (len(traj),)
        assert torch.isfinite(pred.consistency_flag_prob).all()


@pytest.mark.parametrize("mode", ["residual", "direct"])
def test_no_nan_or_inf_with_extreme_input_values(mode):
    # Large-magnitude embeddings stress standardization and gate saturation
    # (sigmoid/tanh) — exactly the failure mode standardization exists to
    # prevent (see module docstring's "Normalization" section).
    rng = np.random.RandomState(2)
    embeddings = (rng.randn(10, 512) * 1000).tolist()
    flags = (rng.rand(10) < 0.4).astype(int).tolist()
    trajectories = [make_trajectory("v1", embeddings, flags)]

    model = GRUDynamicsModel(predict_mode=mode, hidden_dim=8, max_epochs=2, seed=0)
    model.fit(trajectories)
    predictions = model.predict(trajectories)
    assert torch.isfinite(predictions[0].consistency_flag_prob).all()


# --------------------------------------------------------------------------
# documented edge cases
# --------------------------------------------------------------------------

def test_length_one_trajectory_residual_mode_uses_base_rate_fallback():
    trajectories = [_random_trajectory("v1", T=6, seed=0)]
    model = GRUDynamicsModel(predict_mode="residual", hidden_dim=8, max_epochs=1, seed=0)
    model.fit(trajectories)

    lone = [_random_trajectory("v2", T=1, seed=1)]
    predictions = model.predict(lone)
    assert len(predictions[0].consistency_flag_prob) == 1
    assert predictions[0].consistency_flag_prob[0].item() == pytest.approx(model._base_rate)


def test_length_one_trajectory_direct_mode_does_not_crash():
    trajectories = [_random_trajectory("v1", T=4, seed=0)]
    model = GRUDynamicsModel(predict_mode="direct", hidden_dim=8, max_epochs=1, seed=0)
    model.fit(trajectories)

    lone = [_random_trajectory("v2", T=1, seed=1)]
    predictions = model.predict(lone)
    assert len(predictions[0].consistency_flag_prob) == 1
    assert torch.isfinite(predictions[0].consistency_flag_prob).all()
    prob = predictions[0].consistency_flag_prob[0].item()
    assert 0.0 <= prob <= 1.0


def test_residual_mode_single_class_falls_back_without_crashing():
    trajectories = [
        make_trajectory("v1", np.random.RandomState(3).randn(6, 512).tolist(), [0, 0, 0, 0, 0, 0]),
        make_trajectory("v2", np.random.RandomState(4).randn(6, 512).tolist(), [0, 0, 0, 0, 0, 0]),
    ]
    model = GRUDynamicsModel(predict_mode="residual", hidden_dim=8, max_epochs=1, seed=0)
    model.fit(trajectories)  # must not raise

    assert model._residual_logreg is None
    predictions = model.predict(trajectories)
    for pred in predictions:
        assert pred.consistency_flag_prob.tolist() == pytest.approx([0.0] * len(pred.consistency_flag_prob))


def test_fit_with_no_trajectories_falls_back_to_half():
    model = GRUDynamicsModel(predict_mode="residual")
    model.fit([])
    assert model._base_rate == pytest.approx(0.5)
    predictions = model.predict([make_trajectory("v1", embeddings=[[0.0] * 512], consistency_flags=[0])])
    assert predictions[0].consistency_flag_prob[0].item() == pytest.approx(0.5)


# --------------------------------------------------------------------------
# forecast_from_embedding — extra capability, residual mode only
# --------------------------------------------------------------------------

def test_forecast_from_embedding_shape_and_finiteness():
    trajectories = [_random_trajectory(f"v{i}", T=8, seed=i) for i in range(3)]
    model = GRUDynamicsModel(predict_mode="residual", hidden_dim=8, max_epochs=1, seed=0)
    model.fit(trajectories)

    forecast = model.forecast_from_embedding(trajectories[0].embeddings[0], steps=4)
    assert forecast.shape == (4, 512)
    assert torch.isfinite(forecast).all()


def test_forecast_from_embedding_raises_for_direct_mode():
    trajectories = [_random_trajectory("v1", T=5, seed=0)]
    model = GRUDynamicsModel(predict_mode="direct", hidden_dim=8, max_epochs=1, seed=0)
    model.fit(trajectories)

    with pytest.raises(RuntimeError):
        model.forecast_from_embedding(trajectories[0].embeddings[0], steps=2)


# --------------------------------------------------------------------------
# save/load round-trip
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["residual", "direct"])
def test_save_load_round_trip_gives_identical_predictions(tmp_path, mode):
    trajectories = [_random_trajectory(f"v{i}", T=8, seed=i) for i in range(4)]
    model = GRUDynamicsModel(predict_mode=mode, hidden_dim=8, max_epochs=2, seed=0)
    model.fit(trajectories)
    before = model.predict(trajectories)

    save_dir = tmp_path / mode
    model.save(save_dir)
    reloaded = GRUDynamicsModel.load(save_dir)
    after = reloaded.predict(trajectories)

    for pred_before, pred_after in zip(before, after):
        torch.testing.assert_close(
            pred_before.consistency_flag_prob, pred_after.consistency_flag_prob, atol=1e-6, rtol=1e-6
        )


def test_save_load_round_trip_without_residual_logreg(tmp_path):
    # Covers the branch where fit() never trained a residual logreg at all
    # (single-class training data) — save/load must still round-trip.
    trajectories = [
        make_trajectory("v1", np.random.RandomState(5).randn(6, 512).tolist(), [0, 0, 0, 0, 0, 0]),
    ]
    model = GRUDynamicsModel(predict_mode="residual", hidden_dim=8, max_epochs=1, seed=0)
    model.fit(trajectories)
    assert model._residual_logreg is None

    model.save(tmp_path)
    reloaded = GRUDynamicsModel.load(tmp_path)
    assert reloaded._residual_logreg is None
    assert reloaded._base_rate == pytest.approx(model._base_rate)


def test_scaler_save_load_round_trip_matches_exactly(tmp_path):
    trajectories = [_random_trajectory(f"v{i}", T=6, seed=i) for i in range(3)]
    model = GRUDynamicsModel(predict_mode="direct", hidden_dim=8, max_epochs=1, seed=0)
    model.fit(trajectories)

    sample = trajectories[0].embeddings.numpy()
    transformed_before = model._scaler.transform(sample)

    model.save(tmp_path)
    reloaded = GRUDynamicsModel.load(tmp_path)
    transformed_after = reloaded._scaler.transform(sample)

    np.testing.assert_allclose(transformed_before, transformed_after, atol=1e-8)
    np.testing.assert_allclose(model._scaler.mean_, reloaded._scaler.mean_, atol=1e-8)
    np.testing.assert_allclose(model._scaler.scale_, reloaded._scaler.scale_, atol=1e-8)


def test_save_load_round_trip_forecast_from_embedding(tmp_path):
    trajectories = [_random_trajectory(f"v{i}", T=8, seed=i) for i in range(3)]
    model = GRUDynamicsModel(predict_mode="residual", hidden_dim=8, max_epochs=2, seed=0)
    model.fit(trajectories)
    forecast_before = model.forecast_from_embedding(trajectories[0].embeddings[0], steps=3)

    model.save(tmp_path)
    reloaded = GRUDynamicsModel.load(tmp_path)
    forecast_after = reloaded.forecast_from_embedding(trajectories[0].embeddings[0], steps=3)

    torch.testing.assert_close(forecast_before, forecast_after, atol=1e-6, rtol=1e-6)
