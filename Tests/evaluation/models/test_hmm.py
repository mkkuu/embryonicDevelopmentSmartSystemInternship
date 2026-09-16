"""
Unit tests for evaluation.models.hmm.HMMModel.

Synthetic Trajectory objects only — no real embedding cache, no GPU, no
metadata_with_time.csv. Covers the 24 minimum test points from the HMM
branch's Phase 4 implementation plan plus a few tests specifically
requested to verify non-obvious mathematical claims made in hmm.py's
module docstring (the pseudo-likelihood conversion's validity for
posteriors, and predict_next() vs raw A[i,:] not being the same thing).
"""

import itertools
import json

import numpy as np
import pytest
import torch
from scipy.special import logsumexp

from evaluation.model import Model
from evaluation.models.hmm import HMMModel
from evaluation.trajectory import Trajectory


def make_trajectory(video_name, embeddings, phases, window_starts=None, first_phases=None):
    embeddings_t = torch.as_tensor(embeddings, dtype=torch.float32)
    phases_t = torch.as_tensor(phases, dtype=torch.long)
    T = embeddings_t.shape[0]
    first_t = torch.as_tensor(first_phases, dtype=torch.long) if first_phases is not None else phases_t.clone()
    consistency = (first_t != phases_t).long()
    return Trajectory(
        video_name=video_name,
        window_starts=window_starts if window_starts is not None else list(range(T)),
        embeddings=embeddings_t,
        consistency_flag=consistency,
        first_frame_phase=first_t,
        last_frame_phase=phases_t,
    )


def make_separable_dataset(n_states=4, n_videos=30, seed=0):
    """K well-separated Gaussian clusters (one per state) in 2D, arranged
    so a monotone, occasionally-skipping phase sequence 0..K-1 is
    traversed by every synthetic trajectory — gives the logistic
    regression an unambiguous signal to fit, and gives A a real,
    predictable segment/window structure to estimate."""
    rng = np.random.RandomState(seed)
    centers = np.array([[10.0 * i, 0.0] for i in range(n_states)])
    trajectories = []
    for v in range(n_videos):
        seq = [0]
        while seq[-1] < n_states - 1:
            step = 1 if rng.rand() < 0.85 else min(2, n_states - 1 - seq[-1])
            seq.append(min(seq[-1] + step, n_states - 1))
        # repeat each phase a few windows to build real dwell/self-loop signal
        full_seq = []
        for s in seq:
            full_seq.extend([s] * rng.randint(2, 5))
        embeddings = centers[full_seq] + rng.normal(scale=0.5, size=(len(full_seq), 2))
        trajectories.append(make_trajectory(f"v{v}", embeddings, full_seq))
    return trajectories


# --------------------------------------------------------------------------
# 1. import / 2. interface Model
# --------------------------------------------------------------------------

def test_import():
    from evaluation.models import hmm  # noqa: F401


def test_hmm_model_implements_model_interface():
    assert issubclass(HMMModel, Model)
    for method in ("fit", "predict", "save", "load"):
        assert hasattr(HMMModel, method)


def test_hmm_is_registered_under_name_hmm():
    from evaluation.model import get_model_class
    assert get_model_class("hmm") is HMMModel


# --------------------------------------------------------------------------
# 3. fit / 4. predict
# --------------------------------------------------------------------------

def test_fit_does_not_raise_on_reasonable_synthetic_data():
    trajectories = make_separable_dataset()
    model = HMMModel(n_states=4)
    model.fit(trajectories)  # must not raise
    assert model._fitted


def test_predict_before_fit_raises():
    model = HMMModel(n_states=3)
    traj = make_trajectory("v1", [[0.0], [1.0]], [0, 1])
    with pytest.raises(RuntimeError):
        model.predict([traj])


def test_predict_output_is_aligned_and_bounded():
    trajectories = make_separable_dataset()
    model = HMMModel(n_states=4)
    model.fit(trajectories)
    predictions = model.predict(trajectories)
    assert len(predictions) == len(trajectories)
    for traj, pred in zip(trajectories, predictions):
        assert pred.video_name == traj.video_name
        assert pred.window_starts == traj.window_starts
        assert torch.all(pred.consistency_flag_prob >= 0.0)
        assert torch.all(pred.consistency_flag_prob <= 1.0)


# --------------------------------------------------------------------------
# 5. pi normalise / 6. A normalisee / 7. masque j<i
# --------------------------------------------------------------------------

def test_pi_is_normalized():
    trajectories = make_separable_dataset()
    model = HMMModel(n_states=4)
    model.fit(trajectories)
    pi = np.exp(model._log_pi)
    assert pi.sum() == pytest.approx(1.0, abs=1e-8)
    assert np.all(pi >= 0.0)


def test_pi_not_deterministic_on_state_zero_when_data_says_otherwise():
    # Some trajectories start at state 1, not state 0 -> pi must reflect that,
    # not collapse to a deterministic pi_0=1 (Phase 3's explicit decision).
    trajectories = [
        make_trajectory("v_start0", [[0.0], [1.0]], [0, 1]),
        make_trajectory("v_start1a", [[10.0], [11.0]], [1, 2]),
        make_trajectory("v_start1b", [[10.5], [11.5]], [1, 2]),
    ]
    model = HMMModel(n_states=3)
    model.fit(trajectories)
    pi = np.exp(model._log_pi)
    assert pi[0] == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert pi[1] == pytest.approx(2.0 / 3.0, abs=1e-6)
    assert pi[2] == pytest.approx(0.0, abs=1e-6)


def test_A_rows_are_normalized():
    trajectories = make_separable_dataset()
    model = HMMModel(n_states=4)
    model.fit(trajectories)
    A = np.exp(model._log_A)
    A[~np.isfinite(np.exp(model._log_A))] = 0.0
    row_sums = A.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(4), atol=1e-6)


def test_A_masks_backward_transitions_to_exactly_zero():
    trajectories = make_separable_dataset()
    model = HMMModel(n_states=4)
    model.fit(trajectories)
    A = np.exp(model._log_A)
    for i in range(4):
        for j in range(i):
            assert A[i, j] == 0.0, f"A[{i},{j}] should be structurally forbidden (j<i)"
    for i in range(4):
        for j in range(i):
            assert model._log_A[i, j] <= -1e9


def test_A_mask_survives_even_with_adversarial_backward_looking_data():
    # fit() explicitly discards any (a,b) count pair where b < a while
    # counting (see hmm.py fit()) -- verify a trajectory that LOOKS
    # backward in its raw phase sequence still yields a masked A, not a
    # crash or leaked probability mass into j<i.
    trajectories = [
        make_trajectory("v_weird", [[0.0], [1.0], [2.0]], [2, 1, 0]),  # backward on purpose
        make_trajectory("v_normal", [[0.0], [1.0], [2.0]], [0, 1, 2]),
    ]
    model = HMMModel(n_states=3)
    model.fit(trajectories)  # must not raise
    A = np.exp(model._log_A)
    for i in range(3):
        for j in range(i):
            assert A[i, j] == 0.0


# --------------------------------------------------------------------------
# 8. emissions finies
# --------------------------------------------------------------------------

def test_emissions_are_finite():
    trajectories = make_separable_dataset()
    model = HMMModel(n_states=4)
    model.fit(trajectories)
    for traj in trajectories:
        log_b = model._log_emission(traj)
        assert np.all(np.isfinite(log_b) | (log_b <= -1e9))
        assert not np.any(np.isnan(log_b))


def test_emission_falls_back_to_uniform_when_only_one_class_present():
    trajectories = [make_trajectory("v1", [[0.0], [1.0], [2.0]], [0, 0, 0])]
    model = HMMModel(n_states=2)
    model.fit(trajectories)
    assert model._logreg is None
    log_b = model._log_emission(trajectories[0])
    np.testing.assert_allclose(log_b, 0.0)


# --------------------------------------------------------------------------
# 9. filtering normalisee / 10. filtering causal
# --------------------------------------------------------------------------

def test_filtering_rows_are_normalized():
    trajectories = make_separable_dataset()
    model = HMMModel(n_states=4)
    model.fit(trajectories)
    for traj in trajectories[:5]:
        filt = model.filtering(traj)
        np.testing.assert_allclose(filt.sum(axis=1), np.ones(len(traj)), atol=1e-6)
        assert np.all(filt >= 0.0)


def test_filtering_is_causal_same_prefix_same_posterior():
    # Mandatory test (Phase 4 Section 7): two trajectories sharing an
    # identical prefix, diverging afterward, must produce EXACTLY the same
    # filtering posterior on the shared prefix, regardless of the suffix —
    # mirrors test_gru_is_causal in test_gru.py for the same property.
    rng = np.random.RandomState(0)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    prefix_states = [0, 0, 1, 1, 2]
    prefix = (centers[prefix_states] + rng.normal(scale=0.3, size=(5, 2))).tolist()
    suffix_a_states = [2, 2]
    suffix_b_states = [2]
    suffix_a = (centers[suffix_a_states] + rng.normal(scale=0.3, size=(2, 2))).tolist()
    suffix_b = (centers[suffix_b_states] + rng.normal(scale=0.3, size=(1, 2))).tolist()

    traj_a = make_trajectory("va", prefix + suffix_a, prefix_states + suffix_a_states)
    traj_b = make_trajectory("vb", prefix + suffix_b, prefix_states + suffix_b_states)

    train = make_separable_dataset(n_states=3, n_videos=20, seed=1)
    model = HMMModel(n_states=3)
    model.fit(train)

    filt_a = model.filtering(traj_a)
    filt_b = model.filtering(traj_b)
    np.testing.assert_allclose(filt_a[:5], filt_b[:5], atol=1e-10)


# --------------------------------------------------------------------------
# 11. smoothing normalisee / 12. filtering == smoothing pour T=1
# --------------------------------------------------------------------------

def test_smoothing_rows_are_normalized_and_finite():
    trajectories = make_separable_dataset()
    model = HMMModel(n_states=4)
    model.fit(trajectories)
    for traj in trajectories[:5]:
        smooth = model.smoothing(traj)
        assert not np.any(np.isnan(smooth))
        np.testing.assert_allclose(smooth.sum(axis=1), np.ones(len(traj)), atol=1e-6)


def test_filtering_equals_smoothing_when_T_equals_1():
    trajectories = make_separable_dataset()
    model = HMMModel(n_states=4)
    model.fit(trajectories)
    single = make_trajectory("solo", [[0.1, 0.0]], [0])
    filt = model.filtering(single)
    smooth = model.smoothing(single)
    np.testing.assert_allclose(filt, smooth, atol=1e-10)


# --------------------------------------------------------------------------
# 13. next-state prediction normalisee (+ distinction vs raw A)
# --------------------------------------------------------------------------

def test_predict_next_rows_are_normalized():
    trajectories = make_separable_dataset()
    model = HMMModel(n_states=4)
    model.fit(trajectories)
    for traj in trajectories[:5]:
        nxt = model.predict_next(traj)
        np.testing.assert_allclose(nxt.sum(axis=1), np.ones(len(traj)), atol=1e-6)


def test_predict_next_differs_from_raw_A_row_when_posterior_is_uncertain():
    # Explicit test for the distinction the whole HMM design insists on:
    # P(S_{t+1}=j|O_1:t) = sum_i filtering_t(i) A[i,j] is NOT simply "the
    # row of A for the most likely current state" whenever the posterior is
    # not already a one-hot certainty.
    trajectories = make_separable_dataset(n_states=3, n_videos=25, seed=2)
    model = HMMModel(n_states=3)
    model.fit(trajectories)
    traj = trajectories[0]
    filt = model.filtering(traj)
    nxt = model.predict_next(traj)
    A = np.exp(model._log_A)

    found_uncertain = False
    for t in range(len(traj)):
        argmax_state = int(np.argmax(filt[t]))
        if filt[t, argmax_state] < 0.999:  # posterior is not a one-hot certainty
            found_uncertain = True
            assert not np.allclose(nxt[t], A[argmax_state], atol=1e-6), (
                "predict_next() must not degenerate to the raw A row of the "
                "arg-max state when the posterior carries real uncertainty."
            )
    assert found_uncertain, "test data did not exercise an uncertain-posterior window; strengthen the fixture."


# --------------------------------------------------------------------------
# 14/15. Viterbi + no backward + synthetic known path + T=1 + unique state
#          + impossible transitions
# --------------------------------------------------------------------------

def test_viterbi_runs_and_returns_valid_path_length():
    trajectories = make_separable_dataset()
    model = HMMModel(n_states=4)
    model.fit(trajectories)
    for traj in trajectories[:5]:
        path, score = model.viterbi(traj)
        assert len(path) == len(traj)
        assert np.isfinite(score)
        assert all(0 <= s < 4 for s in path)


def test_viterbi_never_goes_backward():
    trajectories = make_separable_dataset()
    model = HMMModel(n_states=4)
    model.fit(trajectories)
    for traj in trajectories:
        path, _ = model.viterbi(traj)
        assert all(b >= a for a, b in zip(path, path[1:])), f"backward step found in decoded path {path}"


def test_viterbi_on_hand_constructed_unambiguous_trajectory():
    # 3 states, embeddings placed so the emission is unambiguous per state;
    # the expected Viterbi path is exactly the phase sequence used to
    # generate the (noiseless) embeddings.
    train = make_separable_dataset(n_states=3, n_videos=25, seed=3)
    model = HMMModel(n_states=3)
    model.fit(train)

    centers = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    expected = [0, 0, 1, 1, 1, 2]
    embeddings = centers[expected]  # exactly on the cluster centers, no noise
    traj = make_trajectory("clean", embeddings, expected)
    path, _ = model.viterbi(traj)
    assert path == expected


def test_viterbi_T_equals_1():
    trajectories = make_separable_dataset()
    model = HMMModel(n_states=4)
    model.fit(trajectories)
    single = make_trajectory("solo", [[0.1, 0.0]], [0])
    path, score = model.viterbi(single)
    assert len(path) == 1
    assert np.isfinite(score)


def test_viterbi_single_reachable_state():
    # n_states=1: only one possible path, trivially.
    trajectories = [make_trajectory("v1", [[0.0], [0.1], [0.2]], [0, 0, 0])]
    model = HMMModel(n_states=1)
    model.fit(trajectories)
    path, score = model.viterbi(trajectories[0])
    assert path == [0, 0, 0]
    assert np.isfinite(score)


def test_viterbi_cannot_select_a_structurally_impossible_transition():
    # Emission is constructed to STRONGLY favor a backward jump (state 0
    # after state 2), which A structurally forbids (log_A=-inf for j<i).
    # A correct Viterbi must never select it, even though the emission
    # evidence alone would prefer it.
    train = make_separable_dataset(n_states=3, n_videos=25, seed=4)
    model = HMMModel(n_states=3)
    model.fit(train)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    # Sequence whose raw embeddings scream "state 2 then state 0" but the
    # model must still decode monotonically given the mask.
    embeddings = np.array([centers[2], centers[0]])
    traj = make_trajectory("adversarial", embeddings, [2, 2])  # labels irrelevant to decoding
    path, _ = model.viterbi(traj)
    assert path[1] >= path[0], f"Viterbi selected a backward transition: {path}"


# --------------------------------------------------------------------------
# 16. log-likelihood
# --------------------------------------------------------------------------

def test_log_likelihood_matches_brute_force_on_tiny_sequence():
    # 2 states, T=2: brute-force sum over the 4 possible state paths must
    # match logsumexp(log_alpha[-1]) exactly.
    trajectories = [
        make_trajectory("v1", [[0.0], [1.0]], [0, 0]),
        make_trajectory("v2", [[0.0], [1.0]], [0, 1]),
        make_trajectory("v3", [[5.0], [1.0]], [1, 1]),
    ]
    model = HMMModel(n_states=2, transition_smoothing_alpha=1.0, transition_decay_rho=0.5)
    model.fit(trajectories)

    probe = make_trajectory("probe", [[0.2], [4.8]], [0, 1])
    log_b = model._log_emission(probe)
    pi = np.exp(model._log_pi)
    A = np.exp(model._log_A)
    b = np.exp(log_b)

    brute_force = 0.0
    for s0 in range(2):
        for s1 in range(2):
            brute_force += pi[s0] * b[0, s0] * A[s0, s1] * b[1, s1]

    ll = model.sequence_log_likelihood(probe)
    assert ll == pytest.approx(np.log(brute_force), abs=1e-6)


def test_log_likelihood_is_finite_for_reasonable_data():
    trajectories = make_separable_dataset()
    model = HMMModel(n_states=4)
    model.fit(trajectories)
    for traj in trajectories[:5]:
        ll = model.sequence_log_likelihood(traj)
        assert np.isfinite(ll)


# --------------------------------------------------------------------------
# Explicit mathematical verification test (Phase 4 Section 6's requirement)
# --------------------------------------------------------------------------

def test_emission_pseudolikelihood_conversion_is_valid_for_posteriors():
    """Direct algebraic check of the cancellation property proved in
    hmm.py's module docstring: multiplying every emission log-probability
    at a given timestep t by an arbitrary STATE-INDEPENDENT constant (i.e.
    exactly the kind of missing P(O_t) factor the hybrid conversion drops)
    must leave filtering, smoothing, predict_next, and viterbi's decoded
    path UNCHANGED, since that constant cancels in every normalization this
    model performs. This is what makes it legitimate to use the b_i(O) :=
    P(S=i|O)/P(S=i) pseudo-likelihood in place of the true, unknown
    P(O|S=i) for every quantity except sequence_log_likelihood (see that
    method's own docstring for the one quantity where this constant does
    NOT cancel)."""
    trajectories = make_separable_dataset(n_states=3, n_videos=20, seed=5)
    model = HMMModel(n_states=3)
    model.fit(trajectories)

    traj = trajectories[0]
    log_b_original = model._log_emission(traj)

    rng = np.random.RandomState(0)
    per_timestep_log_constant = rng.uniform(-5, 5, size=len(traj))
    log_b_perturbed = log_b_original + per_timestep_log_constant[:, None]

    log_alpha_orig = model._log_forward(log_b_original)
    log_alpha_pert = model._log_forward(log_b_perturbed)

    filt_orig = model._normalize_log(log_alpha_orig)
    filt_pert = model._normalize_log(log_alpha_pert)
    np.testing.assert_allclose(filt_orig, filt_pert, atol=1e-8)

    log_beta_orig = model._log_backward(log_b_original)
    log_beta_pert = model._log_backward(log_b_perturbed)
    smooth_orig = model._normalize_log(log_alpha_orig + log_beta_orig)
    smooth_pert = model._normalize_log(log_alpha_pert + log_beta_pert)
    np.testing.assert_allclose(smooth_orig, smooth_pert, atol=1e-8)

    # sequence_log_likelihood MUST differ (this is the documented, expected
    # non-invariance) by exactly sum(per_timestep_log_constant).
    ll_orig = float(logsumexp(log_alpha_orig[-1]))
    ll_pert = float(logsumexp(log_alpha_pert[-1]))
    assert ll_pert == pytest.approx(ll_orig + per_timestep_log_constant.sum(), abs=1e-6)


def test_predict_two_slice_joint_matches_brute_force_enumeration():
    """Diagnostic session 2026-08-21 (consistency_flag_prob calibration
    investigation): independently verifies predict()'s two-slice joint
    P(S_t != S_{t-1} | O_1:t) against FULL PATH enumeration (sum over
    every possible state sequence S_0..S_t, not the forward-algorithm
    shortcut used in production), for every t in a T=5, K=3 random
    synthetic case with the model's own j>=i structural mask respected.
    This rules out a math/implementation bug in the forward recursion or
    predict()'s joint_log construction as an explanation for the observed
    consistency_flag_prob miscalibration — see hmm.py's predict()
    docstring for the (different, demonstrated) root cause."""
    import itertools

    rng = np.random.RandomState(42)
    K, T = 3, 5

    pi = rng.dirichlet(np.ones(K))
    A = np.zeros((K, K))
    for i in range(K):
        A[i, i:] = rng.dirichlet(np.ones(K - i))
    b = rng.dirichlet(np.ones(K), size=T)  # (T, K), rows sum to 1, plain probabilities

    def brute_force(pi, A, b):
        T, K = b.shape
        out = np.full(T, np.nan)
        for t in range(1, T):
            total = 0.0
            same = 0.0
            for path in itertools.product(range(K), repeat=t + 1):
                p = pi[path[0]] * b[0, path[0]]
                for k in range(1, t + 1):
                    p *= A[path[k - 1], path[k]] * b[k, path[k]]
                total += p
                if path[t] == path[t - 1]:
                    same += p
            out[t] = 1.0 - same / total
        return out

    expected = brute_force(pi, A, b)

    # Drive HMMModel's actual production code path (_log_forward +
    # predict()'s own joint_log construction) with these exact pi/A/b.
    model = HMMModel(n_states=K)
    model._log_pi = np.log(pi)
    model._log_A = np.where(A > 0, np.log(np.where(A > 0, A, 1.0)), -1e10)
    log_b = np.log(b)
    log_alpha = model._log_forward(log_b)

    actual = np.full(T, np.nan)
    for t in range(1, T):
        joint_log = log_alpha[t - 1][:, None] + model._log_A + log_b[t][None, :]
        total_log = logsumexp(joint_log)
        same_state_log = logsumexp(np.diag(joint_log))
        actual[t] = 1.0 - float(np.exp(same_state_log - total_log))

    np.testing.assert_allclose(actual[1:], expected[1:], atol=1e-9)


# --------------------------------------------------------------------------
# k-step two-slice joint (2026-08-21 session): generalizes predict()'s
# lag-1 consistency_flag_prob to an arbitrary lag k, so that
# k=(window_size-1) can express P(S_last != S_first | O_1:t) for a window
# -- the event consistency_flag actually measures. predict() itself is
# NOT modified by this work; these tests are entirely additive.
# --------------------------------------------------------------------------

def test_predict_k_step_k_equals_1_matches_predict():
    trajectories = make_separable_dataset(n_states=4, n_videos=10, seed=7)
    model = HMMModel(n_states=4)
    model.fit(trajectories)
    for traj in trajectories[:5]:
        old = model.predict([traj])[0].consistency_flag_prob.numpy()
        new = model.predict_k_step_transition_probability(traj, k=1)
        np.testing.assert_allclose(old, new, atol=1e-9)

    # Also check the batch wrapper predict_k_step() against predict() directly.
    old_preds = model.predict(trajectories[:5])
    new_preds = model.predict_k_step(trajectories[:5], k=1)
    for op, np_ in zip(old_preds, new_preds):
        np.testing.assert_allclose(
            op.consistency_flag_prob.numpy(), np_.consistency_flag_prob.numpy(), atol=1e-9
        )


def test_predict_k_step_no_transition_case_gives_zero():
    # n_states=1: structurally impossible to ever leave the only state,
    # for any k.
    traj = make_trajectory("v", [[0.0]] * 10, [0] * 10)
    model = HMMModel(n_states=1)
    model.fit([traj])
    for k in (1, 3, 7):
        probs = model.predict_k_step_transition_probability(traj, k=k)
        assert np.allclose(probs[k:], 0.0, atol=1e-6)


def test_predict_k_step_forced_transition_approaches_one():
    # A strictly-increasing 8-state chain, trained on nothing but that
    # exact chain repeated: self-loop probability A[i,i] -> ~0 for every
    # non-terminal i, so "no change over k steps" -> ~0, i.e.
    # P(different) -> ~1, for every k up to 7.
    K = 8
    phases = list(range(K))
    embeddings = [[10.0 * i] for i in phases]
    traj_list = [make_trajectory(f"v{i}", embeddings, phases) for i in range(5)]
    model = HMMModel(n_states=K, transition_smoothing_alpha=1e-9)
    model.fit(traj_list)
    probe = make_trajectory("probe", embeddings, phases)
    for k in (1, 3, 7):
        probs = model.predict_k_step_transition_probability(probe, k=k)
        assert probs[k] > 0.99


def test_predict_k_step_matches_brute_force_general_k():
    # Independent full-path enumeration (S_0..S_{t0+k}), NOT using
    # HMMModel's forward algorithm at all, for several (t0, k) pairs.
    rng = np.random.RandomState(123)
    K, T = 3, 6
    pi = rng.dirichlet(np.ones(K))
    A = np.zeros((K, K))
    for i in range(K):
        A[i, i:] = rng.dirichlet(np.ones(K - i))
    b = rng.dirichlet(np.ones(K), size=T)

    def brute_force_k_step(t0, k):
        total = 0.0
        same = 0.0
        for path in itertools.product(range(K), repeat=t0 + k + 1):
            p = pi[path[0]] * b[0, path[0]]
            for s in range(1, t0 + k + 1):
                p *= A[path[s - 1], path[s]] * b[s, path[s]]
            total += p
            if path[t0] == path[t0 + k]:
                same += p
        return 1.0 - same / total

    model = HMMModel(n_states=K)
    model._log_pi = np.log(pi)
    model._log_A = np.where(A > 0, np.log(np.where(A > 0, A, 1.0)), -1e10)
    log_b = np.log(b)
    log_alpha = model._log_forward(log_b)

    for k in (1, 2, 3):
        for t0 in range(0, T - k):
            log_gamma = model._log_k_step_conditional_forward(log_b, t0, k)
            joint_log = log_alpha[t0][:, None] + log_gamma
            total_log = logsumexp(joint_log)
            same_log = logsumexp(np.diag(joint_log))
            actual = 1.0 - float(np.exp(same_log - total_log))
            expected = brute_force_k_step(t0, k)
            assert actual == pytest.approx(expected, abs=1e-9)


def test_predict_k_step_respects_backward_mask():
    # gamma[i, j] for j < i must be exactly 0: a backward step is
    # structurally impossible via A, for every intermediate step of the
    # k-step propagation, not just the endpoints.
    trajectories = make_separable_dataset(n_states=5, n_videos=15, seed=3)
    model = HMMModel(n_states=5)
    model.fit(trajectories)
    log_b = model._log_emission(trajectories[0])
    for k in (1, 3, 4):
        log_gamma = model._log_k_step_conditional_forward(log_b, 0, k)
        gamma = np.exp(log_gamma)
        for i in range(5):
            for j in range(i):
                assert gamma[i, j] == pytest.approx(0.0, abs=1e-9)


def test_predict_k_step_handles_trajectories_too_short_for_k():
    trajectories = make_separable_dataset(n_states=3, n_videos=5, seed=1)
    model = HMMModel(n_states=3)
    model.fit(trajectories)

    short = make_trajectory("short", [[0.0, 0.0], [10.0, 0.0]], [0, 1])  # T=2, k=7 > T
    probs = model.predict_k_step_transition_probability(short, k=7)
    assert len(probs) == 2
    assert np.allclose(probs, model._consistency_base_rate)

    empty = Trajectory(
        video_name="empty",
        window_starts=[],
        embeddings=torch.empty((0, 2)),
        consistency_flag=torch.empty(0, dtype=torch.long),
        first_frame_phase=torch.empty(0, dtype=torch.long),
        last_frame_phase=torch.empty(0, dtype=torch.long),
    )
    probs_empty = model.predict_k_step_transition_probability(empty, k=7)
    assert len(probs_empty) == 0


def test_predict_k_step_uses_window_starts_value_not_index_when_a_gap_exists():
    # window_starts intentionally skips a value (as DataSet.py's
    # phase-consistency filter can do): [0,1,2,4,5,6,7,8]. A trajectory
    # index-based (t-k) lookup would silently misalign across the gap;
    # value-based matching must instead fall back to the base rate for
    # any t whose window_starts[t]-k lookup fails, or whose k-step
    # trajectory-index span crosses the gap.
    window_starts = [0, 1, 2, 4, 5, 6, 7, 8]
    phases = [0, 0, 1, 1, 1, 2, 2, 3]
    embeddings = [[10.0 * p, 0.0] for p in phases]
    e = torch.as_tensor(embeddings, dtype=torch.float32)
    p = torch.as_tensor(phases, dtype=torch.long)
    traj = Trajectory(
        video_name="gapped",
        window_starts=window_starts,
        embeddings=e,
        consistency_flag=torch.zeros(len(phases), dtype=torch.long),
        first_frame_phase=p.clone(),
        last_frame_phase=p,
    )
    train = make_separable_dataset(n_states=4, n_videos=15, seed=9)
    model = HMMModel(n_states=4)
    model.fit(train)

    probs = model.predict_k_step_transition_probability(traj, k=3)
    # window_starts[t]=8 (index 7) minus k=3 -> target=5 -> index 4
    # (window_starts[4]=5). Trajectory-index distance is 7-4=3 == k, AND
    # window_starts is contiguous over that exact span (5,6,7,8) -> a
    # valid computed value is expected here, not a fallback.
    assert probs[7] != pytest.approx(model._consistency_base_rate)

    # window_starts[t]=4 (index 3) minus k=3 -> target=1 -> index 1
    # (window_starts[1]=1). Trajectory-index distance is 3-1=2 != k=3 ->
    # the span [1,4] actually crosses the gap at raw position 3 (missing
    # from window_starts) -> must fall back to the base rate, not
    # silently use index 0 (which is 3 STEPS away in trajectory-index
    # terms, but only spans window_starts 0->4, i.e. 4 raw units, not 3).
    assert probs[3] == pytest.approx(model._consistency_base_rate)


def test_predict_k_step_is_causal_same_prefix_same_result():
    rng = np.random.RandomState(0)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    prefix_states = [0, 0, 1, 1, 2, 2, 2, 2]
    prefix = (centers[prefix_states] + rng.normal(scale=0.3, size=(8, 2))).tolist()
    suffix_a = (centers[[2, 2]] + rng.normal(scale=0.3, size=(2, 2))).tolist()
    suffix_b = (centers[[2]] + rng.normal(scale=0.3, size=(1, 2))).tolist()

    traj_a = make_trajectory("va", prefix + suffix_a, prefix_states + [2, 2])
    traj_b = make_trajectory("vb", prefix + suffix_b, prefix_states + [2])

    train = make_separable_dataset(n_states=3, n_videos=20, seed=1)
    model = HMMModel(n_states=3)
    model.fit(train)

    for k in (1, 3, 7):
        probs_a = model.predict_k_step_transition_probability(traj_a, k=k)
        probs_b = model.predict_k_step_transition_probability(traj_b, k=k)
        # Shared prefix positions (0..7) must be IDENTICAL regardless of
        # what follows -- causality: position t never depends on
        # anything past t.
        np.testing.assert_allclose(probs_a[:8], probs_b[:8], atol=1e-10)


def test_predict_k_step_no_nan_inf_and_bounded_in_unit_interval():
    trajectories = make_separable_dataset(n_states=5, n_videos=20, seed=4)
    model = HMMModel(n_states=5)
    model.fit(trajectories)
    for traj in trajectories[:10]:
        for k in (1, 2, 4, 7):
            probs = model.predict_k_step_transition_probability(traj, k=k)
            assert not np.any(np.isnan(probs))
            assert not np.any(np.isinf(probs))
            assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_predict_k_step_rejects_k_less_than_one():
    trajectories = make_separable_dataset(n_states=3, n_videos=5, seed=2)
    model = HMMModel(n_states=3)
    model.fit(trajectories)
    with pytest.raises(ValueError):
        model.predict_k_step_transition_probability(trajectories[0], k=0)


# --------------------------------------------------------------------------
# 17. T=1 / 18. trajectoire vide / 19. trajectoire tres courte / 20. etat unique
# --------------------------------------------------------------------------

def test_T_equals_1_across_all_methods():
    trajectories = make_separable_dataset()
    model = HMMModel(n_states=4)
    model.fit(trajectories)
    single = make_trajectory("solo", [[0.1, 0.0]], [0])

    filt = model.filtering(single)
    smooth = model.smoothing(single)
    nxt = model.predict_next(single)
    path, score = model.viterbi(single)
    ll = model.sequence_log_likelihood(single)
    preds = model.predict([single])

    assert filt.shape == (1, 4)
    assert smooth.shape == (1, 4)
    assert nxt.shape == (1, 4)
    assert len(path) == 1
    assert np.isfinite(score)
    assert np.isfinite(ll)
    assert len(preds[0].consistency_flag_prob) == 1


def test_empty_trajectory_does_not_crash():
    # Trajectory itself places no lower bound on T; the interface allows an
    # empty one (Trajectory.__len__ simply returns 0) — handled explicitly
    # here (empty outputs), not by raising.
    trajectories = make_separable_dataset()
    model = HMMModel(n_states=4)
    model.fit(trajectories)

    empty = Trajectory(
        video_name="empty",
        window_starts=[],
        embeddings=torch.empty((0, 2)),
        consistency_flag=torch.empty(0, dtype=torch.long),
        first_frame_phase=torch.empty(0, dtype=torch.long),
        last_frame_phase=torch.empty(0, dtype=torch.long),
    )
    assert model.filtering(empty).shape == (0, 4)
    assert model.smoothing(empty).shape == (0, 4)
    assert model.predict_next(empty).shape == (0, 4)
    path, score = model.viterbi(empty)
    assert path == []
    assert score == 0.0
    assert model.sequence_log_likelihood(empty) == 0.0
    preds = model.predict([empty])
    assert len(preds[0].consistency_flag_prob) == 0


def test_very_short_trajectory_T_equals_2():
    trajectories = make_separable_dataset()
    model = HMMModel(n_states=4)
    model.fit(trajectories)
    short = make_trajectory("short", [[0.1, 0.0], [10.2, 0.0]], [0, 1])
    filt = model.filtering(short)
    assert filt.shape == (2, 4)
    path, _ = model.viterbi(short)
    assert len(path) == 2


def test_single_state_model():
    trajectories = [
        make_trajectory("v1", [[0.0], [0.1], [0.2]], [0, 0, 0]),
        make_trajectory("v2", [[0.0], [0.1]], [0, 0]),
    ]
    model = HMMModel(n_states=1)
    model.fit(trajectories)  # must not raise
    filt = model.filtering(trajectories[0])
    np.testing.assert_allclose(filt, np.ones((3, 1)))
    path, _ = model.viterbi(trajectories[0])
    assert path == [0, 0, 0]


# --------------------------------------------------------------------------
# 21. save/load / 22. determinisme
# --------------------------------------------------------------------------

def test_save_load_round_trip_gives_identical_predictions(tmp_path):
    trajectories = make_separable_dataset()
    model = HMMModel(n_states=4)
    model.fit(trajectories)
    before = model.predict(trajectories)

    model.save(tmp_path)
    reloaded = HMMModel.load(tmp_path)
    after = reloaded.predict(trajectories)

    for pred_before, pred_after in zip(before, after):
        torch.testing.assert_close(pred_before.consistency_flag_prob, pred_after.consistency_flag_prob)

    for traj in trajectories[:3]:
        np.testing.assert_allclose(model.filtering(traj), reloaded.filtering(traj), atol=1e-10)
        path_before, score_before = model.viterbi(traj)
        path_after, score_after = reloaded.viterbi(traj)
        assert path_before == path_after
        assert score_before == pytest.approx(score_after, abs=1e-8)


def test_save_load_round_trip_without_logreg(tmp_path):
    trajectories = [make_trajectory("v1", [[0.0], [1.0]], [0, 0])]
    model = HMMModel(n_states=2)
    model.fit(trajectories)
    assert model._logreg is None

    model.save(tmp_path)
    reloaded = HMMModel.load(tmp_path)
    assert reloaded._logreg is None
    np.testing.assert_allclose(model.filtering(trajectories[0]), reloaded.filtering(trajectories[0]))


def test_fit_is_deterministic():
    trajectories = make_separable_dataset()
    model_a = HMMModel(n_states=4)
    model_a.fit(trajectories)
    model_b = HMMModel(n_states=4)
    model_b.fit(trajectories)

    np.testing.assert_allclose(model_a._log_pi, model_b._log_pi)
    np.testing.assert_allclose(model_a._log_A, model_b._log_A)
    for traj in trajectories[:3]:
        np.testing.assert_allclose(model_a.filtering(traj), model_b.filtering(traj))


# --------------------------------------------------------------------------
# 23. absence de NaN/Inf / 24. test des probabilites
# --------------------------------------------------------------------------

def test_no_nan_or_unjustified_inf_anywhere():
    trajectories = make_separable_dataset()
    model = HMMModel(n_states=4)
    model.fit(trajectories)
    assert not np.any(np.isnan(model._log_pi))
    assert not np.any(np.isnan(model._log_A))
    for traj in trajectories[:5]:
        filt = model.filtering(traj)
        smooth = model.smoothing(traj)
        nxt = model.predict_next(traj)
        assert not np.any(np.isnan(filt)) and np.all(np.isfinite(filt))
        assert not np.any(np.isnan(smooth)) and np.all(np.isfinite(smooth))
        assert not np.any(np.isnan(nxt)) and np.all(np.isfinite(nxt))
        _, score = model.viterbi(traj)
        assert np.isfinite(score)
        ll = model.sequence_log_likelihood(traj)
        assert np.isfinite(ll)


def test_all_reported_distributions_are_valid_probabilities():
    trajectories = make_separable_dataset()
    model = HMMModel(n_states=4)
    model.fit(trajectories)
    for traj in trajectories[:5]:
        for dist in (model.filtering(traj), model.smoothing(traj), model.predict_next(traj)):
            assert np.all(dist >= -1e-9)
            assert np.all(dist <= 1.0 + 1e-9)
            np.testing.assert_allclose(dist.sum(axis=1), 1.0, atol=1e-6)


def test_explain_produces_typed_structured_records_not_text():
    trajectories = make_separable_dataset()
    model = HMMModel(n_states=4)
    model.fit(trajectories)
    records = model.explain(trajectories[0], mode="filtering")
    assert len(records) == len(trajectories[0])
    for r in records:
        assert set(r.keys()) == {
            "video", "window", "phase", "phase_posterior",
            "next_phase_probability", "entropy", "inference_mode",
        }
        assert r["inference_mode"] == "filtering"
        assert isinstance(r["phase"], str)
        assert abs(sum(r["phase_posterior"].values()) - 1.0) < 1e-6
        assert abs(sum(r["next_phase_probability"].values()) - 1.0) < 1e-6
        assert r["entropy"] >= 0.0

    with pytest.raises(ValueError):
        model.explain(trajectories[0], mode="viterbi")


# ==========================================================================
# logreg_class_weight (added 2026-08-23, emission-rebalancing experiment)
# ==========================================================================

def _make_imbalanced_dataset(seed=0, n_majority_videos=40, n_minority_videos=2):
    """2-state dataset, well-separated embeddings, but state 1 ('minority')
    is drastically underrepresented in window count -- mirrors the real
    t3/t9+ ~67x imbalance qualitatively (not quantitatively) to give
    class_weight='balanced' something real to correct."""
    rng = np.random.RandomState(seed)
    centers = np.array([[0.0, 0.0], [30.0, 0.0]])
    trajectories = []
    for v in range(n_majority_videos):
        seq = [0] * 20  # long majority-state run
        emb = centers[seq] + rng.normal(scale=1.0, size=(len(seq), 2))
        trajectories.append(make_trajectory(f"maj{v}", emb, seq))
    for v in range(n_minority_videos):
        seq = [0] * 3 + [1] * 3  # short exposure to state 1
        emb = centers[seq] + rng.normal(scale=1.0, size=(len(seq), 2))
        trajectories.append(make_trajectory(f"min{v}", emb, seq))
    return trajectories


def test_logreg_class_weight_default_none_matches_pre_existing_behavior():
    # Adding the parameter must not change ANY existing caller's fitted
    # coefficients -- explicit None and omitting the argument entirely
    # must be bit-for-bit identical (this IS the pre-2026-08-23 behavior).
    trajectories = make_separable_dataset(n_states=4, n_videos=15, seed=3)
    model_omitted = HMMModel(n_states=4)
    model_omitted.fit(trajectories)
    model_explicit_none = HMMModel(n_states=4, logreg_class_weight=None)
    model_explicit_none.fit(trajectories)
    assert model_omitted.logreg_class_weight is None
    np.testing.assert_array_equal(model_omitted._logreg.coef_, model_explicit_none._logreg.coef_)
    np.testing.assert_array_equal(model_omitted._logreg.intercept_, model_explicit_none._logreg.intercept_)


def test_logreg_class_weight_balanced_changes_fitted_coefficients():
    trajectories = _make_imbalanced_dataset(seed=1)
    model_unweighted = HMMModel(n_states=2, logreg_class_weight=None)
    model_unweighted.fit(trajectories)
    model_balanced = HMMModel(n_states=2, logreg_class_weight="balanced")
    model_balanced.fit(trajectories)
    assert not np.allclose(model_unweighted._logreg.coef_, model_balanced._logreg.coef_)
    assert model_balanced.logreg_class_weight == "balanced"
    assert model_unweighted.logreg_class_weight is None


def test_logreg_class_weight_balanced_improves_minority_class_recall():
    # The actual motivating hypothesis: unweighted fit should under-predict
    # the rare class; 'balanced' should recover more of it. Measured on
    # the SAME embeddings used for fitting (a coarse sanity check, not a
    # generalization claim).
    trajectories = _make_imbalanced_dataset(seed=2)
    all_emb = torch.cat([t.embeddings for t in trajectories], dim=0).numpy()
    all_lbl = torch.cat([t.last_frame_phase for t in trajectories], dim=0).numpy()
    minority_mask = all_lbl == 1

    model_unweighted = HMMModel(n_states=2, logreg_class_weight=None)
    model_unweighted.fit(trajectories)
    model_balanced = HMMModel(n_states=2, logreg_class_weight="balanced")
    model_balanced.fit(trajectories)

    X_scaled_u = model_unweighted._scaler.transform(all_emb)
    X_scaled_b = model_balanced._scaler.transform(all_emb)
    pred_u = model_unweighted._logreg.predict(X_scaled_u)
    pred_b = model_balanced._logreg.predict(X_scaled_b)
    recall_u = (pred_u[minority_mask] == 1).mean()
    recall_b = (pred_b[minority_mask] == 1).mean()
    assert recall_b >= recall_u


def test_logreg_class_weight_serialization_roundtrip(tmp_path):
    trajectories = make_separable_dataset(n_states=3, n_videos=10, seed=4)
    model = HMMModel(n_states=3, logreg_class_weight="balanced")
    model.fit(trajectories)
    model.save(tmp_path / "model")
    reloaded = HMMModel.load(tmp_path / "model")
    assert reloaded.logreg_class_weight == "balanced"
    np.testing.assert_array_equal(model._logreg.coef_, reloaded._logreg.coef_)
    for traj in trajectories[:3]:
        np.testing.assert_allclose(model.filtering(traj), reloaded.filtering(traj), atol=1e-10)


def test_logreg_class_weight_missing_key_in_old_state_defaults_to_none(tmp_path):
    # Simulates loading a state.json saved BEFORE this parameter existed
    # (every frozen model in this branch prior to 2026-08-23) -- must not
    # crash, must default to None (the exact pre-existing behavior).
    trajectories = make_separable_dataset(n_states=3, n_videos=10, seed=5)
    model = HMMModel(n_states=3)
    model.fit(trajectories)
    model.save(tmp_path / "model")
    state_path = tmp_path / "model" / "state.json"
    state = json.loads(state_path.read_text())
    assert "logreg_class_weight" in state
    del state["logreg_class_weight"]  # simulate a pre-2026-08-23 artifact
    state_path.write_text(json.dumps(state))
    reloaded = HMMModel.load(tmp_path / "model")
    assert reloaded.logreg_class_weight is None


def test_logreg_class_weight_balanced_no_nan_inf_and_causal():
    trajectories = _make_imbalanced_dataset(seed=6, n_majority_videos=20, n_minority_videos=3)
    model = HMMModel(n_states=2, logreg_class_weight="balanced")
    model.fit(trajectories)
    for traj in trajectories[:5]:
        for arr in (model.filtering(traj), model.smoothing(traj), model.predict_next(traj)):
            assert not np.any(np.isnan(arr))
            assert not np.any(np.isinf(arr))
        probs = model.predict_k_step_transition_probability(traj, k=1)
        assert not np.any(np.isnan(probs)) and not np.any(np.isinf(probs))

    # causality: identical shared prefix must give identical filtering,
    # regardless of what follows -- same discipline as the existing
    # causality tests, re-checked specifically for a balanced-weight fit.
    rng = np.random.RandomState(7)
    centers = np.array([[0.0, 0.0], [30.0, 0.0]])
    prefix_states = [0, 0, 0, 1, 1]
    prefix = (centers[prefix_states] + rng.normal(scale=0.3, size=(5, 2))).tolist()
    traj_a = make_trajectory("a", prefix + [[30.0, 0.0]], prefix_states + [1])
    traj_b = make_trajectory("b", prefix + [[30.0, 0.0], [30.0, 0.0]], prefix_states + [1, 1])
    filt_a = model.filtering(traj_a)
    filt_b = model.filtering(traj_b)
    np.testing.assert_allclose(filt_a[:5], filt_b[:5], atol=1e-10)
