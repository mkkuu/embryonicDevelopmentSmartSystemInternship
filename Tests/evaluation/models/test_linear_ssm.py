"""
Unit tests for evaluation.models.linear_ssm.LinearStateSpaceModel.

Synthetic Trajectory objects only — no real embedding cache, no GPU.
Beyond the standard interface tests, this file's two most important
tests are test_apply_dynamics_hand_computed (catches matrix-orientation
bugs a symmetric test matrix would hide) and
test_fit_never_pairs_across_trajectory_boundaries (a correctness property
that would be very easy to silently violate and very hard to notice by
inspection alone).
"""

import numpy as np
import pytest
import torch

from evaluation.models.linear_ssm import LinearStateSpaceModel
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


def make_linear_trajectory(video_name, A, b, z0, n_steps):
    """Generates a trajectory that exactly follows z_{i+1} = A z_i + b,
    with no noise — used to check that fit() recovers a known dynamics
    rule and that forecast/impute reproduce it."""
    A = np.asarray(A)
    b = np.asarray(b)
    values = [np.asarray(z0)]
    for _ in range(n_steps):
        values.append(A @ values[-1] + b)
    return make_trajectory(video_name, [v.tolist() for v in values], [0] * len(values))


# --------------------------------------------------------------------------
# the actual matrix arithmetic — deliberately non-symmetric A, which a
# transpose-direction bug would fail on but a symmetric A would hide
# --------------------------------------------------------------------------

def test_apply_dynamics_hand_computed():
    model = LinearStateSpaceModel(latent_dim=None)
    model._A = torch.tensor([[2.0, 1.0], [0.0, 3.0]])
    model._b = torch.tensor([1.0, 1.0])

    z = torch.tensor([[1.0, 0.0]])
    result = model._apply_dynamics(z)
    # A @ [1,0] = [2,0]; + b = [3,1]
    torch.testing.assert_close(result, torch.tensor([[3.0, 1.0]]))


def test_forecast_multistep_matches_closed_form():
    model = LinearStateSpaceModel(latent_dim=None)
    model._pca = None
    model._A = torch.tensor([[2.0, 0.0], [0.0, 0.5]])
    model._b = torch.tensor([1.0, 1.0])

    traj = make_trajectory("v1", embeddings=[[1.0, 1.0]], consistency_flags=[0])
    forecasted = model.forecast(traj, steps=2)

    # step 1: A@[1,1] + b = [2,0.5] + [1,1] = [3, 1.5]
    # step 2: A@[3,1.5] + b = [6,0.75] + [1,1] = [7, 1.75]
    expected = torch.tensor([[3.0, 1.5], [7.0, 1.75]])
    torch.testing.assert_close(forecasted, expected, atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------
# "learn transition matrix A" — the actual scientific claim of this model
# --------------------------------------------------------------------------

def test_fit_recovers_known_linear_dynamics():
    rng = np.random.RandomState(0)
    true_A = np.array([[0.9, 0.0], [0.0, 0.8]])
    true_b = np.array([0.1, -0.05])

    trajectories = [
        make_linear_trajectory(f"v{v}", true_A, true_b, rng.uniform(-1, 1, size=2), n_steps=20)
        for v in range(15)
    ]

    model = LinearStateSpaceModel(latent_dim=None, ridge_alpha=1e-6)  # near-zero regularization: clean, noise-free data
    model.fit(trajectories)

    np.testing.assert_allclose(model._A.numpy(), true_A, atol=0.05)
    np.testing.assert_allclose(model._b.numpy(), true_b, atol=0.05)


def test_fit_never_pairs_across_trajectory_boundaries():
    true_A = np.array([[0.5]])
    true_b = np.array([0.0])
    # Wildly different starting points — if fit() incorrectly paired the
    # last window of traj1 with the first window of traj2, this
    # artificial jump would corrupt the fitted A.
    traj1 = make_linear_trajectory("v1", true_A, true_b, [1.0], n_steps=5)
    traj2 = make_linear_trajectory("v2", true_A, true_b, [-100.0], n_steps=5)

    model = LinearStateSpaceModel(latent_dim=None, ridge_alpha=1e-6)
    model.fit([traj1, traj2])

    np.testing.assert_allclose(model._A.numpy(), true_A, atol=0.05)


# --------------------------------------------------------------------------
# "missing-frame prediction"
# --------------------------------------------------------------------------

def test_impute_missing_window_recovers_clean_synthetic_value():
    true_A = np.array([[0.9]])
    true_b = np.array([0.05])
    traj = make_linear_trajectory("v1", true_A, true_b, [1.0], n_steps=6)

    model = LinearStateSpaceModel(latent_dim=None, ridge_alpha=1e-6)
    model.fit([traj] * 3)  # repeated so ridge has enough pairs to fit cleanly

    true_value = traj.embeddings[3].clone()
    imputed = model.impute_missing_window(traj, missing_index=3)

    torch.testing.assert_close(imputed, true_value, atol=0.05, rtol=0.05)


def test_impute_last_window_uses_forward_only_when_uninvertible():
    model = LinearStateSpaceModel(latent_dim=None)
    model._pca = None
    model._A = torch.zeros((1, 1))  # deliberately singular: not invertible
    model._b = torch.tensor([1.0])
    model._A_inv = model._safe_invert(model._A)
    assert model._A_inv is None  # confirms the fixture itself is set up correctly

    traj = make_trajectory("v1", embeddings=[[5.0], [10.0], [10.0]], consistency_flags=[0, 0, 0])
    imputed = model.impute_missing_window(traj, missing_index=2)  # last window, no forward-only fallback possible via backward
    # only the forward estimate (from window 1) is available: A@10 + b = 0 + 1 = 1
    torch.testing.assert_close(imputed, torch.tensor([1.0]), atol=1e-4, rtol=1e-4)


def test_impute_only_window_raises():
    model = LinearStateSpaceModel(latent_dim=None)
    model._A = torch.eye(1)
    model._b = torch.zeros(1)
    model._A_inv = torch.eye(1)
    traj = make_trajectory("v1", embeddings=[[1.0]], consistency_flags=[0])
    with pytest.raises(ValueError):
        model.impute_missing_window(traj, missing_index=0)


def test_fit_with_state_dimension_one_produces_2d_A():
    """Regression test for a shape bug: sklearn's Ridge.coef_ degenerates to
    a 1D array of shape (n_features,) instead of (n_targets, n_features)
    whenever n_targets == 1, i.e. whenever the latent/state dimension is 1
    (scalar embeddings, or latent_dim=1). fit() must normalize this back to
    the (d, d) = (1, 1) matrix contract the rest of the class assumes, so
    that _safe_invert (which requires a >=2D array) and downstream
    predict()/impute_missing_window() don't crash."""
    true_A = np.array([[0.5]])
    true_b = np.array([0.0])
    traj = make_linear_trajectory("v1", true_A, true_b, [1.0], n_steps=5)

    model = LinearStateSpaceModel(latent_dim=None, ridge_alpha=1e-6)
    model.fit([traj])

    assert model._A.shape == (1, 1)
    assert model._b.shape == (1,)
    # _safe_invert must not raise on a well-conditioned 1x1 matrix.
    assert model._A_inv is not None
    assert model._A_inv.shape == (1, 1)


# --------------------------------------------------------------------------
# "optional observation matrix C"
# --------------------------------------------------------------------------

def test_latent_dim_reduces_A_shape():
    rng = np.random.RandomState(0)
    trajectories = []
    for v in range(10):
        embeddings = rng.randn(8, 6).tolist()  # D=6
        flags = (rng.rand(8) < 0.4).astype(int).tolist()
        trajectories.append(make_trajectory(f"v{v}", embeddings, flags))

    model = LinearStateSpaceModel(latent_dim=3)
    model.fit(trajectories)
    assert model._A.shape == (3, 3)


def test_latent_dim_none_uses_full_embedding_dimension():
    rng = np.random.RandomState(0)
    trajectories = []
    for v in range(10):
        embeddings = rng.randn(8, 6).tolist()  # D=6
        flags = (rng.rand(8) < 0.4).astype(int).tolist()
        trajectories.append(make_trajectory(f"v{v}", embeddings, flags))

    model = LinearStateSpaceModel(latent_dim=None)
    model.fit(trajectories)
    assert model._pca is None
    assert model._A.shape == (6, 6)


# --------------------------------------------------------------------------
# safe inversion
# --------------------------------------------------------------------------

def test_safe_invert_returns_none_for_singular_matrix():
    singular = torch.tensor([[1.0, 2.0], [2.0, 4.0]])  # rank 1
    assert LinearStateSpaceModel._safe_invert(singular) is None


def test_safe_invert_returns_correct_inverse_for_well_conditioned_matrix():
    A = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
    A_inv = LinearStateSpaceModel._safe_invert(A)
    assert A_inv is not None
    torch.testing.assert_close(A_inv, torch.tensor([[0.5, 0.0], [0.0, 0.25]]), atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------
# interface contract and edge cases
# --------------------------------------------------------------------------

def test_predict_before_fit_raises():
    model = LinearStateSpaceModel()
    traj = make_trajectory("v1", embeddings=[[0.0], [1.0]], consistency_flags=[0, 1])
    with pytest.raises(RuntimeError):
        model.predict([traj])


def test_forecast_before_fit_raises():
    model = LinearStateSpaceModel()
    traj = make_trajectory("v1", embeddings=[[0.0]], consistency_flags=[0])
    with pytest.raises(RuntimeError):
        model.forecast(traj, steps=1)


def test_predict_output_is_aligned_and_bounded():
    trajectories = [make_trajectory("v1", embeddings=[[0.0], [5.0], [5.1]], consistency_flags=[0, 1, 0])]
    model = LinearStateSpaceModel(latent_dim=None)
    model.fit(trajectories)
    predictions = model.predict(trajectories)

    pred = predictions[0]
    assert pred.video_name == "v1"
    assert pred.window_starts == trajectories[0].window_starts
    assert torch.all(pred.consistency_flag_prob >= 0.0)
    assert torch.all(pred.consistency_flag_prob <= 1.0)


def test_fit_with_single_class_falls_back_to_base_rate_without_crashing():
    trajectories = [
        make_trajectory("v1", embeddings=[[0.0], [1.0], [2.0]], consistency_flags=[0, 0, 0]),
        make_trajectory("v2", embeddings=[[0.0], [1.0], [2.0]], consistency_flags=[0, 0, 0]),
    ]
    model = LinearStateSpaceModel(latent_dim=None)
    model.fit(trajectories)  # must not raise
    assert model._logreg is None
    predictions = model.predict(trajectories)
    for pred in predictions:
        assert pred.consistency_flag_prob.tolist() == pytest.approx([0.0] * len(pred.consistency_flag_prob))


# --------------------------------------------------------------------------
# save/load round-trip
# --------------------------------------------------------------------------

def test_save_load_round_trip_gives_identical_predictions_and_forecasts(tmp_path):
    rng = np.random.RandomState(2)
    true_A = np.array([[0.7, 0.1], [0.0, 0.6]])
    true_b = np.array([0.2, 0.0])
    trajectories = [
        make_linear_trajectory(f"v{v}", true_A, true_b, rng.uniform(-1, 1, size=2), n_steps=10)
        for v in range(10)
    ]

    model = LinearStateSpaceModel(latent_dim=None, ridge_alpha=1e-3)
    model.fit(trajectories)
    predictions_before = model.predict(trajectories)
    forecast_before = model.forecast(trajectories[0], steps=3)

    model.save(tmp_path)
    reloaded = LinearStateSpaceModel.load(tmp_path)
    predictions_after = reloaded.predict(trajectories)
    forecast_after = reloaded.forecast(trajectories[0], steps=3)

    for pred_before, pred_after in zip(predictions_before, predictions_after):
        torch.testing.assert_close(pred_before.consistency_flag_prob, pred_after.consistency_flag_prob)
    torch.testing.assert_close(forecast_before, forecast_after)


def test_save_load_round_trip_with_pca(tmp_path):
    rng = np.random.RandomState(3)
    trajectories = []
    for v in range(10):
        embeddings = rng.randn(8, 5).tolist()
        flags = (rng.rand(8) < 0.4).astype(int).tolist()
        trajectories.append(make_trajectory(f"v{v}", embeddings, flags))

    model = LinearStateSpaceModel(latent_dim=2)
    model.fit(trajectories)
    predictions_before = model.predict(trajectories)
    # forecast() round-trips through _to_embedding -> pca.inverse_transform,
    # the other PCA code path besides predict()'s _to_latent -> pca.transform
    # — both must survive save/load, not just one.
    forecast_before = model.forecast(trajectories[0], steps=2)

    model.save(tmp_path)
    reloaded = LinearStateSpaceModel.load(tmp_path)
    assert reloaded._pca is not None
    assert reloaded._A.shape == (2, 2)
    predictions_after = reloaded.predict(trajectories)
    forecast_after = reloaded.forecast(trajectories[0], steps=2)

    for pred_before, pred_after in zip(predictions_before, predictions_after):
        torch.testing.assert_close(pred_before.consistency_flag_prob, pred_after.consistency_flag_prob, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(forecast_before, forecast_after, atol=1e-4, rtol=1e-4)
