"""
Unit tests for evaluation.models.semi_hmm (SemiHMMModel, DurationModel
hierarchy, censoring, explicit-duration forward). Synthetic data only --
no real embedding cache, no GPU, no Test split anywhere in this file.
"""

import itertools
import json

import numpy as np
import pytest
import torch
from scipy.special import logsumexp

from evaluation.model import Model
from evaluation.models.hmm import HMMModel
from evaluation.models.semi_hmm import (
    CensoringType,
    DurationModel,
    EmpiricalDurationModel,
    NegativeBinomialDurationModel,
    SemiHMMModel,
    classify_segment_censoring,
    determine_dmax,
    duration_model_from_dict,
    extract_phase_durations,
)
from evaluation.trajectory import Trajectory

_NEG_INF = -1e10


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


def make_separable_dataset(n_states=4, n_videos=20, seed=0):
    rng = np.random.RandomState(seed)
    centers = np.array([[10.0 * i, 0.0] for i in range(n_states)])
    trajectories = []
    for v in range(n_videos):
        seq = [0]
        while seq[-1] < n_states - 1:
            step = 1 if rng.rand() < 0.85 else min(2, n_states - 1 - seq[-1])
            seq.append(min(seq[-1] + step, n_states - 1))
        full_seq = []
        for s in seq:
            full_seq.extend([s] * rng.randint(2, 5))
        embeddings = centers[full_seq] + rng.normal(scale=0.5, size=(len(full_seq), 2))
        trajectories.append(make_trajectory(f"v{v}", embeddings, full_seq))
    return trajectories


# ==========================================================================
# A. Segmentation / censoring helpers
# ==========================================================================

def test_import():
    from evaluation.models import semi_hmm  # noqa: F401


def test_semi_hmm_implements_model_interface():
    assert issubclass(SemiHMMModel, Model)
    for method in ("fit", "predict", "save", "load"):
        assert hasattr(SemiHMMModel, method)


def test_semi_hmm_is_registered():
    from evaluation.model import get_model_class
    assert get_model_class("semi_hmm") is SemiHMMModel


def test_extract_phase_durations_simple_segments():
    traj = make_trajectory("v", [[0.0]] * 6, [2, 2, 2, 3, 3, 4])
    durations = extract_phase_durations([traj])
    assert durations[2] == [(3, CensoringType.LEFT_CENSORED)]
    assert durations[3] == [(2, CensoringType.OBSERVED)]
    assert durations[4] == [(1, CensoringType.RIGHT_CENSORED)]


def test_extract_phase_durations_no_self_transition_created():
    traj = make_trajectory("v", [[0.0]] * 6, [2, 2, 2, 3, 3, 4])
    segs = HMMModel._segments(traj)
    phase_sequence = [p for p, _, _ in segs]
    assert phase_sequence == [2, 3, 4]
    for a, b in zip(phase_sequence[:-1], phase_sequence[1:]):
        assert b > a  # strictly increasing -- no i->i segment-to-segment pair


def test_classify_segment_censoring_single_segment_video():
    assert classify_segment_censoring(0, 1) == CensoringType.BOTH_CENSORED


def test_classify_segment_censoring_middle_is_observed():
    assert classify_segment_censoring(1, 3) == CensoringType.OBSERVED
    assert classify_segment_censoring(0, 3) == CensoringType.LEFT_CENSORED
    assert classify_segment_censoring(2, 3) == CensoringType.RIGHT_CENSORED


def test_determine_dmax_strategies():
    durations = {0: [(5, CensoringType.OBSERVED), (10, CensoringType.OBSERVED), (3, CensoringType.LEFT_CENSORED)]}
    assert determine_dmax(durations, strategy="observed_max") == 10
    assert determine_dmax(durations, strategy="explicit", explicit_value=7) == 7
    q = determine_dmax(durations, strategy="quantile", quantile=0.5)
    assert 1 <= q <= 10


def test_determine_dmax_rejects_empty():
    with pytest.raises(ValueError):
        determine_dmax({}, strategy="observed_max")


# ==========================================================================
# C/E/F. Duration models
# ==========================================================================

def test_duration_model_support_starts_at_one():
    dm = NegativeBinomialDurationModel(dmax=10, r=2.0, p=0.4)
    with pytest.raises(ValueError):
        dm.log_prob(0)
    with pytest.raises(ValueError):
        dm.log_prob(11)


def test_duration_model_probabilities_sum_to_one_negbinom():
    dm = NegativeBinomialDurationModel(dmax=20, r=2.0, p=0.4)
    total = sum(dm.probability(d) for d in range(1, dm.dmax + 1))
    assert total == pytest.approx(1.0, abs=1e-6)


def test_duration_model_probabilities_sum_to_one_empirical():
    dm = EmpiricalDurationModel(dmax=5, probs=np.array([0.4, 0.3, 0.1, 0.1, 0.1]))
    total = sum(dm.probability(d) for d in range(1, dm.dmax + 1))
    assert total == pytest.approx(1.0, abs=1e-9)


def test_duration_model_expected_duration_matches_manual_sum():
    dm = EmpiricalDurationModel(dmax=4, probs=np.array([0.25, 0.25, 0.25, 0.25]))
    expected = 1 * 0.25 + 2 * 0.25 + 3 * 0.25 + 4 * 0.25
    assert dm.expected_duration() == pytest.approx(expected)


def test_duration_model_quantiles_monotonic():
    dm = NegativeBinomialDurationModel(dmax=30, r=3.0, p=0.3)
    qs = dm.quantiles([0.1, 0.5, 0.9])
    assert qs[0.1] <= qs[0.5] <= qs[0.9]
    for v in qs.values():
        assert 1 <= v <= dm.dmax


def test_duration_model_survival_probability_at_one_is_one():
    for dm in (NegativeBinomialDurationModel(dmax=10, r=1.0, p=0.5),
               EmpiricalDurationModel(dmax=10)):
        assert dm.survival_probability(1) == pytest.approx(1.0, abs=1e-9)


def test_duration_model_survival_beyond_dmax_is_zero():
    dm = NegativeBinomialDurationModel(dmax=5, r=1.0, p=0.5)
    assert dm.survival_probability(6) == pytest.approx(0.0, abs=1e-12)


def test_duration_model_survival_is_non_increasing():
    dm = NegativeBinomialDurationModel(dmax=15, r=2.0, p=0.4)
    survs = [dm.survival_probability(d) for d in range(1, 16)]
    for a, b in zip(survs[:-1], survs[1:]):
        assert b <= a + 1e-9


def test_negative_binomial_fit_recovers_reasonable_parameters():
    rng = np.random.RandomState(0)
    true_r, true_p = 3.0, 0.3
    raw = rng.negative_binomial(true_r, true_p, size=500) + 1  # shift to support >=1
    durations = [int(min(d, 40)) for d in raw]
    censoring = [CensoringType.OBSERVED] * len(durations)
    dm = NegativeBinomialDurationModel(dmax=40)
    dm.fit(durations, censoring)
    assert np.isfinite(dm.r) and dm.r > 0
    assert 0 < dm.p < 1
    # Fitted mean should be in the right ballpark of the empirical mean.
    assert dm.expected_duration() == pytest.approx(np.mean(durations), rel=0.35)


def test_negative_binomial_fit_uses_censored_survival_contribution():
    # Two datasets differing ONLY in whether the large durations are
    # flagged OBSERVED or RIGHT_CENSORED must fit to different parameters
    # -- proves the censoring type actually changes the likelihood used.
    durations = [2, 3, 3, 4, 20, 20, 20]
    dm_observed = NegativeBinomialDurationModel(dmax=25)
    dm_observed.fit(durations, [CensoringType.OBSERVED] * len(durations))
    dm_censored = NegativeBinomialDurationModel(dmax=25)
    dm_censored.fit(durations, [CensoringType.OBSERVED] * 4 + [CensoringType.RIGHT_CENSORED] * 3)
    assert (dm_observed.r, dm_observed.p) != (dm_censored.r, dm_censored.p)


def test_negative_binomial_fit_is_deterministic():
    durations = [2, 3, 5, 8, 13, 21, 4, 6, 7, 9]
    censoring = [CensoringType.OBSERVED] * len(durations)
    dm1 = NegativeBinomialDurationModel(dmax=30)
    dm1.fit(durations, censoring)
    dm2 = NegativeBinomialDurationModel(dmax=30)
    dm2.fit(durations, censoring)
    assert dm1.r == pytest.approx(dm2.r, abs=1e-9)
    assert dm1.p == pytest.approx(dm2.p, abs=1e-9)


def test_empirical_duration_model_fit_uses_observed_only():
    durations = [2, 2, 3]
    censoring = [CensoringType.OBSERVED, CensoringType.OBSERVED, CensoringType.LEFT_CENSORED]
    dm = EmpiricalDurationModel(dmax=5)
    dm.fit(durations, censoring, epsilon=0.0)
    # Only the two OBSERVED duration=2 entries should shape the histogram;
    # the censored duration=3 must be excluded.
    assert dm.probability(2) > dm.probability(3)
    assert dm.fallback_uniform is False


def test_empirical_duration_model_zero_observed_falls_back_uniform():
    dm = EmpiricalDurationModel(dmax=4)
    dm.fit([5], [CensoringType.LEFT_CENSORED])
    assert dm.fallback_uniform is True
    for d in range(1, 5):
        assert dm.probability(d) == pytest.approx(0.25)


def test_empirical_duration_model_clips_durations_above_dmax():
    dm = EmpiricalDurationModel(dmax=3)
    dm.fit([10, 10, 1], [CensoringType.OBSERVED] * 3, epsilon=0.0)
    assert dm.probability(3) > dm.probability(2)


def test_duration_model_serialization_round_trip():
    for dm in (NegativeBinomialDurationModel(dmax=12, r=2.5, p=0.35),
               EmpiricalDurationModel(dmax=6, probs=np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.1]))):
        state = dm.to_dict()
        restored = duration_model_from_dict(state)
        for d in range(1, dm.dmax + 1):
            assert restored.probability(d) == pytest.approx(dm.probability(d), abs=1e-9)


# ==========================================================================
# B/D. Transition / SemiHMMModel fit basics
# ==========================================================================

def test_fit_does_not_raise_on_reasonable_synthetic_data():
    trajectories = make_separable_dataset(n_states=4, n_videos=15, seed=1)
    model = SemiHMMModel(n_states=4, dmax=15)
    model.fit(trajectories)
    assert model._fitted


def test_transition_matrix_j_greater_than_i_only():
    trajectories = make_separable_dataset(n_states=5, n_videos=15, seed=2)
    model = SemiHMMModel(n_states=5, dmax=15)
    model.fit(trajectories)
    A = np.exp(model._log_A)
    for i in range(5):
        for j in range(5):
            if j <= i:
                assert A[i, j] == pytest.approx(0.0, abs=1e-9)  # no self-loop, no backward


def test_transition_matrix_rows_sum_to_one_for_non_terminal_states():
    trajectories = make_separable_dataset(n_states=5, n_videos=15, seed=3)
    model = SemiHMMModel(n_states=5, dmax=15)
    model.fit(trajectories)
    A = np.exp(model._log_A)
    for i in range(4):  # non-terminal
        assert A[i, :].sum() == pytest.approx(1.0, abs=1e-6)


def test_transition_matrix_terminal_state_has_no_outgoing_row():
    trajectories = make_separable_dataset(n_states=4, n_videos=15, seed=4)
    model = SemiHMMModel(n_states=4, dmax=15)
    model.fit(trajectories)
    A = np.exp(model._log_A)
    assert A[3, :].sum() == pytest.approx(0.0, abs=1e-9)


def test_transitions_allow_skips_j_greater_than_i_plus_one():
    # Force a dataset with an explicit skip (0 -> 2 directly, never 0->1).
    trajectories = [make_trajectory(f"v{i}", [[0.0], [10.0], [10.0], [20.0]], [0, 2, 2, 2]) for i in range(10)]
    model = SemiHMMModel(n_states=3, dmax=10)
    model.fit(trajectories)
    A = np.exp(model._log_A)
    assert A[0, 2] > 0.0  # skip allowed and observed


# ==========================================================================
# G/H. Forward vs independent brute-force enumeration
# ==========================================================================

def _brute_force_filtering(model: SemiHMMModel, traj: Trajectory, t: int) -> np.ndarray:
    """Independent enumeration (NOT calling _explicit_duration_forward):
    sums, over every way to partition window range [0..t] into m>=1
    consecutive segments with strictly increasing states (first m-1
    segments 'closed' -- exact duration pmf; last segment 'open' at time
    t -- duration survival), the joint log-probability, grouped by the
    terminal (currently active) state."""
    log_b = model._log_emission(traj)
    K = model.n_states
    n = t + 1

    def rec(start, prev_state, log_acc, is_first):
        results = []
        for end in range(start, n):
            length = end - start + 1
            if length > model.dmax:
                continue
            state_range = range(K) if is_first else range(prev_state + 1, K)
            for s in state_range:
                trans_term = model._log_pi[s] if is_first else model._log_A[prev_state, s]
                if trans_term <= _NEG_INF / 2:
                    continue
                em = float(np.sum(log_b[start:end + 1, s]))
                if end == t:
                    dur_term = model.duration_models[s].log_survival(length)
                    results.append((s, log_acc + trans_term + dur_term + em))
                else:
                    dur_term = model.duration_models[s].log_prob(length)
                    results.extend(rec(end + 1, s, log_acc + trans_term + dur_term + em, False))
        return results

    log_joint = np.full(K, _NEG_INF)
    for s, val in rec(0, None, 0.0, True):
        log_joint[s] = float(logsumexp([log_joint[s], val]))
    total = logsumexp(log_joint)
    return np.exp(log_joint - total)


def test_forward_matches_brute_force_T1():
    traj = make_trajectory("v", [[0.0, 0.0]], [0])
    model = SemiHMMModel(n_states=3, dmax=5)
    model.fit(make_separable_dataset(n_states=3, n_videos=10, seed=5))
    filt = model.filtering(traj)
    bf = _brute_force_filtering(model, traj, 0)
    np.testing.assert_allclose(filt[0], bf, atol=1e-8)


def test_forward_matches_brute_force_T2_two_phases():
    traj = make_trajectory("v", [[0.0, 0.0], [10.0, 0.0]], [0, 1])
    model = SemiHMMModel(n_states=3, dmax=5)
    model.fit(make_separable_dataset(n_states=3, n_videos=10, seed=6))
    filt = model.filtering(traj)
    for t in range(2):
        bf = _brute_force_filtering(model, traj, t)
        np.testing.assert_allclose(filt[t], bf, atol=1e-8)


def test_forward_matches_brute_force_T4_one_phase():
    traj = make_trajectory("v", [[0.0, 0.0]] * 4, [1, 1, 1, 1])
    model = SemiHMMModel(n_states=3, dmax=6)
    model.fit(make_separable_dataset(n_states=3, n_videos=10, seed=7))
    filt = model.filtering(traj)
    for t in range(4):
        bf = _brute_force_filtering(model, traj, t)
        np.testing.assert_allclose(filt[t], bf, atol=1e-7)


def test_forward_matches_brute_force_T4_two_segments_known_duration():
    traj = make_trajectory("v", [[0.0, 0.0]] * 4, [0, 0, 1, 1])
    model = SemiHMMModel(n_states=3, dmax=6)
    model.fit(make_separable_dataset(n_states=3, n_videos=12, seed=8))
    filt = model.filtering(traj)
    for t in range(4):
        bf = _brute_force_filtering(model, traj, t)
        np.testing.assert_allclose(filt[t], bf, atol=1e-7)


def test_forward_matches_brute_force_T4_known_transition_three_phases():
    traj = make_trajectory("v", [[0.0, 0.0]] * 4, [0, 1, 1, 2])
    model = SemiHMMModel(n_states=4, dmax=6)
    model.fit(make_separable_dataset(n_states=4, n_videos=12, seed=9))
    filt = model.filtering(traj)
    for t in range(4):
        bf = _brute_force_filtering(model, traj, t)
        np.testing.assert_allclose(filt[t], bf, atol=1e-7)


# ==========================================================================
# I. Causality
# ==========================================================================

def test_filtering_is_causal_same_prefix_same_result():
    model = SemiHMMModel(n_states=4, dmax=8)
    model.fit(make_separable_dataset(n_states=4, n_videos=15, seed=10))
    embeddings_prefix = [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]]
    phases_prefix = [0, 0, 1]
    traj_a = make_trajectory("a", embeddings_prefix + [[20.0, 0.0]], phases_prefix + [2])
    traj_b = make_trajectory("b", embeddings_prefix + [[0.5, 0.0]], phases_prefix + [1])
    filt_a = model.filtering(traj_a)
    filt_b = model.filtering(traj_b)
    np.testing.assert_allclose(filt_a[:3], filt_b[:3], atol=1e-9)


def test_predict_is_causal():
    model = SemiHMMModel(n_states=4, dmax=8)
    model.fit(make_separable_dataset(n_states=4, n_videos=15, seed=11))
    embeddings_prefix = [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]]
    phases_prefix = [0, 0, 1]
    traj_a = make_trajectory("a", embeddings_prefix + [[20.0, 0.0]], phases_prefix + [2])
    traj_b = make_trajectory("b", embeddings_prefix + [[0.5, 0.0]], phases_prefix + [1])
    pred_a = model.predict([traj_a])[0].consistency_flag_prob.numpy()
    pred_b = model.predict([traj_b])[0].consistency_flag_prob.numpy()
    np.testing.assert_allclose(pred_a[:3], pred_b[:3], atol=1e-6)


# ==========================================================================
# J. Determinism
# ==========================================================================

def test_fit_is_deterministic():
    trajectories = make_separable_dataset(n_states=4, n_videos=15, seed=12)
    model1 = SemiHMMModel(n_states=4, dmax=15)
    model1.fit(trajectories)
    model2 = SemiHMMModel(n_states=4, dmax=15)
    model2.fit(trajectories)
    np.testing.assert_allclose(model1._log_A, model2._log_A, atol=1e-9)
    np.testing.assert_allclose(model1._log_pi, model2._log_pi, atol=1e-9)
    for dm1, dm2 in zip(model1.duration_models, model2.duration_models):
        for d in range(1, dm1.dmax + 1):
            assert dm1.probability(d) == pytest.approx(dm2.probability(d), abs=1e-9)


# ==========================================================================
# K. Serialization
# ==========================================================================

def test_save_load_round_trip_predictions_identical(tmp_path):
    trajectories = make_separable_dataset(n_states=4, n_videos=15, seed=13)
    model = SemiHMMModel(n_states=4, dmax=15, duration_family="negative_binomial")
    model.fit(trajectories)
    model.save(tmp_path / "semi_hmm_model")
    reloaded = SemiHMMModel.load(tmp_path / "semi_hmm_model")

    for traj in trajectories[:3]:
        orig = model.predict([traj])[0].consistency_flag_prob.numpy()
        loaded = reloaded.predict([traj])[0].consistency_flag_prob.numpy()
        np.testing.assert_allclose(orig, loaded, atol=1e-6)
        np.testing.assert_allclose(model.filtering(traj), reloaded.filtering(traj), atol=1e-6)


def test_save_load_round_trip_empirical_family(tmp_path):
    trajectories = make_separable_dataset(n_states=3, n_videos=12, seed=14)
    model = SemiHMMModel(n_states=3, dmax=10, duration_family="empirical")
    model.fit(trajectories)
    model.save(tmp_path / "semi_hmm_empirical")
    reloaded = SemiHMMModel.load(tmp_path / "semi_hmm_empirical")
    for traj in trajectories[:3]:
        np.testing.assert_allclose(model.filtering(traj), reloaded.filtering(traj), atol=1e-6)


# ==========================================================================
# L. Edge cases
# ==========================================================================

def test_phase_never_observed_as_segment_falls_back_explicitly():
    # n_states=5 but state 3 never appears in training data at all.
    trajectories = [make_trajectory(f"v{i}", [[0.0], [10.0], [40.0]], [0, 1, 4]) for i in range(10)]
    model = SemiHMMModel(n_states=5, dmax=10, duration_family="empirical")
    model.fit(trajectories)
    assert model.duration_models[3].fallback_uniform is True


def test_transition_never_observed_still_gets_smoothed_nonzero_mass():
    trajectories = [make_trajectory(f"v{i}", [[0.0], [10.0]], [0, 1]) for i in range(10)]
    model = SemiHMMModel(n_states=4, dmax=10, transition_smoothing_alpha=1.0)
    model.fit(trajectories)
    A = np.exp(model._log_A)
    # 0->2 and 0->3 never observed, but must not be exactly zero (Dirichlet prior).
    assert A[0, 2] > 0.0
    assert A[0, 3] > 0.0


def test_rare_long_duration_does_not_crash():
    trajectories = make_separable_dataset(n_states=3, n_videos=10, seed=15)
    long_traj = make_trajectory("long", [[0.0, 0.0]] * 25, [0] * 25)
    trajectories.append(long_traj)
    model = SemiHMMModel(n_states=3, dmax=30)
    model.fit(trajectories)
    filt = model.filtering(long_traj)
    assert filt.shape == (25, 3)
    assert np.all(np.isfinite(filt))


def test_T1_trajectory():
    model = SemiHMMModel(n_states=3, dmax=5)
    model.fit(make_separable_dataset(n_states=3, n_videos=10, seed=16))
    traj = make_trajectory("v", [[0.0, 0.0]], [0])
    filt = model.filtering(traj)
    assert filt.shape == (1, 3)
    assert filt.sum() == pytest.approx(1.0, abs=1e-6)
    pred = model.predict([traj])[0].consistency_flag_prob.numpy()
    assert len(pred) == 1


def test_dmax_equals_one():
    trajectories = [make_trajectory(f"v{i}", [[0.0], [10.0]], [0, 1]) for i in range(10)]
    model = SemiHMMModel(n_states=3, dmax=1, duration_family="negative_binomial")
    model.fit(trajectories)
    for dm in model.duration_models:
        assert dm.probability(1) == pytest.approx(1.0, abs=1e-6)
    filt = model.filtering(trajectories[0])
    assert np.all(np.isfinite(filt))


def test_dmax_smaller_than_an_observed_duration_does_not_crash():
    trajectories = [make_trajectory(f"v{i}", [[0.0]] * 8, [0] * 5 + [1] * 3) for i in range(10)]
    model = SemiHMMModel(n_states=3, dmax=3)  # much smaller than the true duration=5 for phase 0
    model.fit(trajectories)  # durations get clipped to dmax internally, must not raise
    filt = model.filtering(trajectories[0])
    assert np.all(np.isfinite(filt))
    assert filt.shape == (8, 3)


def test_empty_trajectory():
    empty = Trajectory(
        video_name="empty", window_starts=[], embeddings=torch.empty((0, 2)),
        consistency_flag=torch.empty(0, dtype=torch.long),
        first_frame_phase=torch.empty(0, dtype=torch.long),
        last_frame_phase=torch.empty(0, dtype=torch.long),
    )
    model = SemiHMMModel(n_states=3, dmax=5)
    model.fit(make_separable_dataset(n_states=3, n_videos=10, seed=17))
    filt = model.filtering(empty)
    assert filt.shape == (0, 3)
    pred = model.predict([empty])[0]
    assert len(pred.consistency_flag_prob) == 0


# ==========================================================================
# Duration/transition/next-phase/chain outputs, mathematical validation
# ==========================================================================

def test_filtering_rows_sum_to_one():
    trajectories = make_separable_dataset(n_states=4, n_videos=15, seed=18)
    model = SemiHMMModel(n_states=4, dmax=15)
    model.fit(trajectories)
    for traj in trajectories[:5]:
        filt = model.filtering(traj)
        row_sums = filt.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(len(traj)), atol=1e-6)


def test_next_phase_distribution_rows_sum_to_one():
    trajectories = make_separable_dataset(n_states=4, n_videos=15, seed=19)
    model = SemiHMMModel(n_states=4, dmax=15)
    model.fit(trajectories)
    for traj in trajectories[:5]:
        nxt = model.next_phase_distribution(traj)
        row_sums = nxt.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(len(traj)), atol=1e-5)


def test_duration_distribution_sums_to_one_for_every_state():
    trajectories = make_separable_dataset(n_states=5, n_videos=15, seed=20)
    model = SemiHMMModel(n_states=5, dmax=20)
    model.fit(trajectories)
    for name in model.state_names:
        dist = model.duration_distribution(name)
        assert sum(dist.values()) == pytest.approx(1.0, abs=1e-6)


def test_transition_chain_structure():
    trajectories = make_separable_dataset(n_states=4, n_videos=10, seed=21)
    model = SemiHMMModel(n_states=4, dmax=15)
    model.fit(trajectories)
    chain = model.transition_chain(trajectories[0])
    assert len(chain) >= 1
    for entry in chain:
        assert "phase" in entry
        assert "observed_duration_windows" in entry
        assert entry["observed_duration_windows"] >= 1
        assert 0.0 <= entry["duration_probability_at_observed"] <= 1.0
        assert "next_phase_distribution" in entry


def test_no_nan_or_inf_anywhere():
    trajectories = make_separable_dataset(n_states=5, n_videos=15, seed=22)
    model = SemiHMMModel(n_states=5, dmax=15)
    model.fit(trajectories)
    for traj in trajectories[:5]:
        filt = model.filtering(traj)
        nxt = model.next_phase_distribution(traj)
        pred = model.predict([traj])[0].consistency_flag_prob.numpy()
        assert np.all(np.isfinite(filt))
        assert np.all(np.isfinite(nxt))
        assert np.all(np.isfinite(pred))
        assert np.all(pred >= 0.0) and np.all(pred <= 1.0)


def test_no_test_split_referenced_anywhere_in_semi_hmm_module():
    import inspect
    from evaluation.models import semi_hmm as module
    source = inspect.getsource(module)
    assert '"test"' not in source
    assert "'test'" not in source


# ==========================================================================
# Phase A1: numerical robustness fix (log_b clipped to a finite sentinel)
# ==========================================================================

def test_log_emission_clips_neg_inf_to_finite_sentinel(monkeypatch):
    # Simulates sklearn's predict_log_proba underflowing to literal -inf
    # for an extremely improbable class (the STEP 2 smoke test's actual
    # trigger: an out-of-distribution embedding) -- the fix must clip this
    # to the finite _NEG_INF sentinel so the forward recursion's prefix-sum
    # subtraction (cum[end]-cum[start]) never computes -inf-(-inf)=nan.
    trajectories = make_separable_dataset(n_states=3, n_videos=10, seed=30)
    model = SemiHMMModel(n_states=3, dmax=6)
    model.fit(trajectories)
    traj = trajectories[0]

    real_log_emission = model._emission_hmm._log_emission

    def extremely_improbable_log_emission(t):
        arr = real_log_emission(t).copy()
        arr[0, 0] = -np.inf  # extremely improbable emission: literal -inf
        if len(t) > 1:
            arr[-1, 1] = -np.inf
        return arr

    monkeypatch.setattr(model._emission_hmm, "_log_emission", extremely_improbable_log_emission)

    log_b = model._log_emission(traj)
    assert np.all(np.isfinite(log_b)), "log_b must be clipped to a finite sentinel, never literal -inf"
    assert not np.any(np.isposinf(log_b))
    assert np.all(log_b >= _NEG_INF - 1.0)

    filt = model.filtering(traj)
    assert np.all(np.isfinite(filt)), "filtering() must remain finite even with an extremely improbable emission"
    assert not np.any(np.isnan(filt))
    np.testing.assert_allclose(filt.sum(axis=1), np.ones(len(traj)), atol=1e-6)


def test_phase_with_zero_observed_but_nonzero_censored_forces_uniform_fallback():
    # A phase that appears only as a censored occurrence (never OBSERVED --
    # e.g. tPB2/tEB structurally, 0/400 and 0/265 on real Train, confirmed
    # this session) must NOT be handed to NegativeBinomialDurationModel's
    # censored-only MLE (which degenerates to the dmax boundary, per the
    # STEP 2 smoke test finding) -- it must be forced to a uniform
    # EmpiricalDurationModel fallback regardless of the configured
    # duration_family.
    trajectories = [make_trajectory(f"v{i}", [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], [0, 1, 2]) for i in range(10)]
    # state 0 is ALWAYS the first segment of its video here -> always
    # LEFT_CENSORED, never OBSERVED.
    model = SemiHMMModel(n_states=3, dmax=10, duration_family="negative_binomial")
    model.fit(trajectories)
    dm0 = model.duration_models[0]
    assert isinstance(dm0, EmpiricalDurationModel)
    assert dm0.fallback_uniform is True
    for d in range(1, model.dmax + 1):
        assert dm0.probability(d) == pytest.approx(1.0 / model.dmax)
    # state 1 (middle segment, OBSERVED) should still use the requested family.
    assert isinstance(model.duration_models[1], NegativeBinomialDurationModel)


# ==========================================================================
# Phase C: reference vs optimized forward, equivalence
# ==========================================================================

def _clip_to_sentinel(arr, floor=-1e9):
    # Both implementations use the SAME finite sentinel (_NEG_INF=-1e10) to
    # mean "impossible", but repeated sentinel-scale additions can land at
    # different magnitudes (-1e10 vs -2e10, say) depending on exactly which
    # terms got summed -- both still mean exactly zero probability. Clamp
    # anything already far below any realistic log-probability to a common
    # floor before comparing, so only REAL (non-degenerate) values are held
    # to numerical tolerance.
    return np.where(arr <= floor, floor, arr)


def _compare_reference_vs_optimized(model, traj, atol=1e-6):
    ref_end, ref_age = model._explicit_duration_forward_reference(traj)
    opt_end, opt_age = model._explicit_duration_forward_optimized(traj)
    np.testing.assert_allclose(_clip_to_sentinel(ref_end), _clip_to_sentinel(opt_end), atol=atol)
    np.testing.assert_allclose(_clip_to_sentinel(ref_age), _clip_to_sentinel(opt_age), atol=atol)


def test_reference_vs_optimized_various_T():
    model = SemiHMMModel(n_states=4, dmax=10)
    model.fit(make_separable_dataset(n_states=4, n_videos=15, seed=40))
    for T in (1, 2, 3, 5, 8, 15):
        phases = [min(t // 3, 3) for t in range(T)]
        traj = make_trajectory(f"v_T{T}", [[float(p) * 10.0, 0.0] for p in phases], phases)
        _compare_reference_vs_optimized(model, traj)


def test_reference_vs_optimized_various_dmax():
    trajectories = make_separable_dataset(n_states=4, n_videos=15, seed=41)
    traj = make_trajectory("v", [[0.0, 0.0]] * 3 + [[10.0, 0.0]] * 4 + [[20.0, 0.0]] * 2, [0] * 3 + [1] * 4 + [2] * 2)
    for dmax in (1, 3, 5, 10, 20):
        model = SemiHMMModel(n_states=4, dmax=dmax)
        model.fit(trajectories)
        _compare_reference_vs_optimized(model, traj)


def test_reference_vs_optimized_various_n_states():
    for n_states in (2, 3, 5, 6):
        trajectories = make_separable_dataset(n_states=n_states, n_videos=15, seed=42)
        model = SemiHMMModel(n_states=n_states, dmax=12)
        model.fit(trajectories)
        for traj in trajectories[:3]:
            _compare_reference_vs_optimized(model, traj)


def test_reference_vs_optimized_negative_binomial_and_empirical():
    trajectories = make_separable_dataset(n_states=4, n_videos=15, seed=43)
    for family in ("negative_binomial", "empirical"):
        model = SemiHMMModel(n_states=4, dmax=15, duration_family=family)
        model.fit(trajectories)
        for traj in trajectories[:3]:
            _compare_reference_vs_optimized(model, traj)


def test_reference_vs_optimized_various_transition_smoothing():
    trajectories = make_separable_dataset(n_states=5, n_videos=15, seed=44)
    for alpha, rho in [(0.1, 0.2), (1.0, 0.5), (5.0, 0.9)]:
        model = SemiHMMModel(n_states=5, dmax=12, transition_smoothing_alpha=alpha, transition_decay_rho=rho)
        model.fit(trajectories)
        for traj in trajectories[:3]:
            _compare_reference_vs_optimized(model, traj)


def test_reference_vs_optimized_filtering_and_next_phase_agree():
    # End-to-end: filtering()/next_phase_distribution() (which use the
    # optimized path by default) must match what the reference path would
    # produce if substituted directly.
    trajectories = make_separable_dataset(n_states=4, n_videos=15, seed=45)
    model = SemiHMMModel(n_states=4, dmax=10)
    model.fit(trajectories)
    for traj in trajectories[:3]:
        _, ref_age = model._explicit_duration_forward_reference(traj)
        _, opt_age = model._explicit_duration_forward_optimized(traj)
        ref_marginal = logsumexp(ref_age, axis=2)  # (T,K)
        ref_filt = np.exp(ref_marginal - logsumexp(ref_marginal, axis=1, keepdims=True))
        opt_filt = model.filtering(traj)
        np.testing.assert_allclose(ref_filt, opt_filt, atol=1e-6)


def _compare_next_phase_reference_vs_optimized(model, traj, atol=1e-6):
    ref = model._next_phase_distribution_reference(traj)
    opt = model._next_phase_distribution_optimized(traj)
    assert ref.shape == opt.shape
    np.testing.assert_allclose(ref, opt, atol=atol)


def test_next_phase_reference_vs_optimized_various_T():
    model = SemiHMMModel(n_states=4, dmax=10)
    model.fit(make_separable_dataset(n_states=4, n_videos=15, seed=50))
    for T in (1, 2, 3, 5, 8, 15):
        phases = [min(t // 3, 3) for t in range(T)]
        traj = make_trajectory(f"v_np_T{T}", [[float(p) * 10.0, 0.0] for p in phases], phases)
        _compare_next_phase_reference_vs_optimized(model, traj)


def test_next_phase_reference_vs_optimized_various_dmax():
    trajectories = make_separable_dataset(n_states=4, n_videos=15, seed=51)
    traj = make_trajectory("v_np", [[0.0, 0.0]] * 3 + [[10.0, 0.0]] * 4 + [[20.0, 0.0]] * 2, [0] * 3 + [1] * 4 + [2] * 2)
    for dmax in (1, 3, 5, 10, 20):
        model = SemiHMMModel(n_states=4, dmax=dmax)
        model.fit(trajectories)
        _compare_next_phase_reference_vs_optimized(model, traj)


def test_next_phase_reference_vs_optimized_various_n_states():
    for n_states in (2, 3, 5, 6):
        trajectories = make_separable_dataset(n_states=n_states, n_videos=15, seed=52)
        model = SemiHMMModel(n_states=n_states, dmax=12)
        model.fit(trajectories)
        for traj in trajectories[:3]:
            _compare_next_phase_reference_vs_optimized(model, traj)


def test_next_phase_reference_vs_optimized_negative_binomial_and_empirical():
    trajectories = make_separable_dataset(n_states=4, n_videos=15, seed=53)
    for family in ("negative_binomial", "empirical"):
        model = SemiHMMModel(n_states=4, dmax=15, duration_family=family)
        model.fit(trajectories)
        for traj in trajectories[:3]:
            _compare_next_phase_reference_vs_optimized(model, traj)


def test_next_phase_reference_vs_optimized_various_transition_smoothing():
    trajectories = make_separable_dataset(n_states=5, n_videos=15, seed=54)
    for alpha, rho in [(0.1, 0.2), (1.0, 0.5), (5.0, 0.9)]:
        model = SemiHMMModel(n_states=5, dmax=12, transition_smoothing_alpha=alpha, transition_decay_rho=rho)
        model.fit(trajectories)
        for traj in trajectories[:3]:
            _compare_next_phase_reference_vs_optimized(model, traj)


def test_next_phase_reference_vs_optimized_dmax_equals_one():
    # Edge case: Dmax=1 means max_a is always 1, and log_surv_a1 is always
    # the a+1<=Dmax branch's _NEG_INF fallback (a+1=2>Dmax=1) for every
    # (t,j) -- exercises the boundary of the log_surv_table lookup range.
    trajectories = make_separable_dataset(n_states=3, n_videos=10, seed=55)
    model = SemiHMMModel(n_states=3, dmax=1)
    model.fit(trajectories)
    for traj in trajectories[:5]:
        _compare_next_phase_reference_vs_optimized(model, traj)


def test_next_phase_reference_vs_optimized_zero_observed_phase_fallback():
    # Phases with zero observed occurrences (forced uniform
    # EmpiricalDurationModel fallback, see fit()'s own docstring) --
    # confirms the table-lookup path handles the fallback duration model
    # identically to the reference's direct calls.
    trajectories = make_separable_dataset(n_states=4, n_videos=15, seed=56)
    model = SemiHMMModel(n_states=4, dmax=10)
    model.fit(trajectories)
    assert any(dm.fallback_uniform for dm in model.duration_models if hasattr(dm, "fallback_uniform"))
    for traj in trajectories[:5]:
        _compare_next_phase_reference_vs_optimized(model, traj)


def test_next_phase_reference_vs_optimized_empty_trajectory():
    trajectories = make_separable_dataset(n_states=3, n_videos=10, seed=57)
    model = SemiHMMModel(n_states=3, dmax=8)
    model.fit(trajectories)
    empty = make_trajectory("v_np_empty", np.empty((0, 2)), [])
    ref = model._next_phase_distribution_reference(empty)
    opt = model._next_phase_distribution_optimized(empty)
    assert ref.shape == (0, 3)
    assert opt.shape == (0, 3)


def test_public_next_phase_distribution_uses_optimized_path_by_default():
    # The public API must dispatch to the optimized implementation, not the
    # reference oracle -- verified by construction (same result as calling
    # _next_phase_distribution_optimized directly, bit-identical since it
    # is literally the same call).
    trajectories = make_separable_dataset(n_states=4, n_videos=10, seed=58)
    model = SemiHMMModel(n_states=4, dmax=10)
    model.fit(trajectories)
    for traj in trajectories[:3]:
        public_result = model.next_phase_distribution(traj)
        optimized_result = model._next_phase_distribution_optimized(traj)
        assert np.array_equal(public_result, optimized_result)


def test_log_emission_normal_values_unaffected_by_clip():
    # The clip must be a strict no-op on any realistic log-probability --
    # normal-case behavior is unchanged, only literal -inf/degenerate
    # entries are affected.
    trajectories = make_separable_dataset(n_states=4, n_videos=10, seed=31)
    model = SemiHMMModel(n_states=4, dmax=8)
    model.fit(trajectories)
    traj = trajectories[0]
    clipped = model._log_emission(traj)
    unclipped = model._emission_hmm._log_emission(traj)
    np.testing.assert_allclose(clipped, unclipped, atol=1e-12)


# ==========================================================================
# J. k-step transition probability (P(S_t != S_{t-k} | O_1:t))
# ==========================================================================
# Mirrors Tests/evaluation/models/test_hmm.py's predict_k_step_* suite --
# same discipline (window_starts matching, base-rate fallback,
# causality), but the correctness oracle here is the age<=k closed form's
# OWN premise (no self-loop in A => consecutive segments differ), verified
# independently below via full-path brute-force enumeration, not assumed.

def test_predict_k_step_k_equals_1_matches_predict():
    trajectories = make_separable_dataset(n_states=4, n_videos=10, seed=7)
    model = SemiHMMModel(n_states=4, dmax=10)
    model.fit(trajectories)
    for traj in trajectories[:5]:
        old = model.predict([traj])[0].consistency_flag_prob.numpy()
        new = model.predict_k_step_transition_probability(traj, k=1)
        np.testing.assert_allclose(old, new, atol=1e-9)

    old_preds = model.predict(trajectories[:5])
    new_preds = model.predict_k_step(trajectories[:5], k=1)
    for op, np_ in zip(old_preds, new_preds):
        np.testing.assert_allclose(
            op.consistency_flag_prob.numpy(), np_.consistency_flag_prob.numpy(), atol=1e-9
        )


def _brute_force_k_step(model: SemiHMMModel, traj: Trajectory, t: int, k: int) -> float:
    """Independent full segmentation-path enumeration (NOT calling
    forward()/predict_k_step_transition_probability at all): sums, over
    every way to partition [0..t] into consecutive segments with strictly
    increasing states (last segment 'open' at t -- survival term), the
    joint log-probability of (state at t-k, state at t), then returns
    P(different) = 1 - P(same). Reuses _brute_force_filtering's
    recursion structure but tracks BOTH the state active at t-k and at t
    (not just the terminal one), so it is a genuinely independent check
    of the age<=k closed form used by predict_k_step_transition_probability
    -- if that identity were wrong, this brute force would disagree."""
    log_b = model._log_emission(traj)
    K = model.n_states
    n = t + 1

    def rec(start, prev_state, log_acc, is_first):
        results = []  # list of (state_at_tmk_or_None, state_at_t, log_prob)
        for end in range(start, n):
            length = end - start + 1
            if length > model.dmax:
                continue
            state_range = range(K) if is_first else range(prev_state + 1, K)
            for s in state_range:
                trans_term = model._log_pi[s] if is_first else model._log_A[prev_state, s]
                if trans_term <= _NEG_INF / 2:
                    continue
                em = float(np.sum(log_b[start:end + 1, s]))
                state_at_tmk = s if (start <= t - k <= end) else None
                if end == t:
                    dur_term = model.duration_models[s].log_survival(length)
                    results.append((state_at_tmk, s, log_acc + trans_term + dur_term + em))
                else:
                    dur_term = model.duration_models[s].log_prob(length)
                    for tmk_state, term_s, val in rec(end + 1, s, log_acc + trans_term + dur_term + em, False):
                        results.append((tmk_state if tmk_state is not None else state_at_tmk, term_s, val))
        return results

    all_paths = rec(0, None, 0.0, True)
    total = logsumexp([v for _, _, v in all_paths])
    same_terms = [v for tmk, s, v in all_paths if tmk is not None and tmk == s]
    if t - k < 0:
        return float("nan")  # not a well-defined event; caller must not query this
    same = logsumexp(same_terms) if same_terms else _NEG_INF
    return 1.0 - float(np.exp(same - total))


def test_predict_k_step_matches_brute_force_general_k():
    trajectories = [
        make_trajectory("v", [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0], [20.0, 0.0]],
                         [0, 0, 1, 1, 2]),
    ]
    model = SemiHMMModel(n_states=3, dmax=4)
    model.fit(make_separable_dataset(n_states=3, n_videos=12, seed=50))
    traj = trajectories[0]
    for k in (1, 2, 3):
        for t in range(k, len(traj)):
            expected = _brute_force_k_step(model, traj, t, k)
            actual = model.predict_k_step_transition_probability(traj, k=k)[t]
            assert actual == pytest.approx(expected, abs=1e-6)


def test_predict_k_step_handles_trajectories_too_short_for_k():
    trajectories = make_separable_dataset(n_states=3, n_videos=5, seed=1)
    model = SemiHMMModel(n_states=3, dmax=8)
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
    # Same gapped window_starts scenario as test_hmm.py's analogous test:
    # [0,1,2,4,5,6,7,8] -- a raw index-based (t-k) lookup would silently
    # misalign across the missing value 3.
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
    model = SemiHMMModel(n_states=4, dmax=10)
    model.fit(train)

    probs = model.predict_k_step_transition_probability(traj, k=3)
    # window_starts[7]=8, target=5 -> index 4 (window_starts[4]=5), 7-4=3==k,
    # AND window_starts is contiguous over [5,6,7,8] -> valid computed value.
    assert probs[7] != pytest.approx(model._consistency_base_rate)
    # window_starts[3]=4, target=1 -> index 1 (window_starts[1]=1), but
    # 3-1=2 != k=3 -> the span crosses the gap at raw position 3 -> fallback.
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
    model = SemiHMMModel(n_states=3, dmax=10)
    model.fit(train)

    for k in (1, 3, 7):
        probs_a = model.predict_k_step_transition_probability(traj_a, k=k)
        probs_b = model.predict_k_step_transition_probability(traj_b, k=k)
        np.testing.assert_allclose(probs_a[:8], probs_b[:8], atol=1e-10)


def test_predict_k_step_no_nan_inf_and_bounded_in_unit_interval():
    trajectories = make_separable_dataset(n_states=5, n_videos=20, seed=4)
    model = SemiHMMModel(n_states=5, dmax=12)
    model.fit(trajectories)
    for traj in trajectories[:10]:
        for k in (1, 2, 4, 7):
            probs = model.predict_k_step_transition_probability(traj, k=k)
            assert not np.any(np.isnan(probs))
            assert not np.any(np.isinf(probs))
            assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_predict_k_step_rejects_k_less_than_one():
    trajectories = make_separable_dataset(n_states=3, n_videos=5, seed=2)
    model = SemiHMMModel(n_states=3, dmax=8)
    model.fit(trajectories)
    with pytest.raises(ValueError):
        model.predict_k_step_transition_probability(trajectories[0], k=0)


def test_predict_k_step_batch_wrapper_matches_single():
    trajectories = make_separable_dataset(n_states=4, n_videos=10, seed=12)
    model = SemiHMMModel(n_states=4, dmax=10)
    model.fit(trajectories)
    preds = model.predict_k_step(trajectories[:4], k=3)
    for traj, pred in zip(trajectories[:4], preds):
        expected = model.predict_k_step_transition_probability(traj, k=3)
        np.testing.assert_allclose(pred.consistency_flag_prob.numpy(), expected, atol=1e-9)
        assert pred.video_name == traj.video_name
        assert pred.window_starts == traj.window_starts


# ==========================================================================
# K. logreg_class_weight threading (added 2026-08-23, emission-rebalancing)
# ==========================================================================

def _make_imbalanced_dataset(seed=0, n_majority_videos=40, n_minority_videos=2):
    rng = np.random.RandomState(seed)
    centers = np.array([[0.0, 0.0], [30.0, 0.0]])
    trajectories = []
    for v in range(n_majority_videos):
        seq = [0] * 20
        emb = centers[seq] + rng.normal(scale=1.0, size=(len(seq), 2))
        trajectories.append(make_trajectory(f"maj{v}", emb, seq))
    for v in range(n_minority_videos):
        seq = [0] * 3 + [1] * 3
        emb = centers[seq] + rng.normal(scale=1.0, size=(len(seq), 2))
        trajectories.append(make_trajectory(f"min{v}", emb, seq))
    return trajectories


def test_logreg_class_weight_default_none_matches_pre_existing_behavior():
    trajectories = make_separable_dataset(n_states=4, n_videos=15, seed=20)
    model_omitted = SemiHMMModel(n_states=4, dmax=10)
    model_omitted.fit(trajectories)
    model_explicit_none = SemiHMMModel(n_states=4, dmax=10, logreg_class_weight=None)
    model_explicit_none.fit(trajectories)
    assert model_omitted.logreg_class_weight is None
    np.testing.assert_array_equal(
        model_omitted._emission_hmm._logreg.coef_, model_explicit_none._emission_hmm._logreg.coef_
    )


def test_logreg_class_weight_threads_to_composed_emission_hmm():
    trajectories = _make_imbalanced_dataset(seed=21)
    model = SemiHMMModel(n_states=2, dmax=25, logreg_class_weight="balanced")
    model.fit(trajectories)
    assert model._emission_hmm.logreg_class_weight == "balanced"

    # Cross-check against a standalone HMMModel fit on the identical data
    # with the identical setting -- the composed emission must not merely
    # store the flag but actually produce the same fitted classifier.
    standalone = HMMModel(n_states=2, logreg_class_weight="balanced")
    standalone.fit(trajectories)
    np.testing.assert_array_equal(model._emission_hmm._logreg.coef_, standalone._logreg.coef_)

    unweighted = SemiHMMModel(n_states=2, dmax=25, logreg_class_weight=None)
    unweighted.fit(trajectories)
    assert not np.allclose(model._emission_hmm._logreg.coef_, unweighted._emission_hmm._logreg.coef_)


def test_logreg_class_weight_serialization_roundtrip(tmp_path):
    trajectories = make_separable_dataset(n_states=3, n_videos=10, seed=22)
    model = SemiHMMModel(n_states=3, dmax=8, logreg_class_weight="balanced")
    model.fit(trajectories)
    model.save(tmp_path / "model")
    reloaded = SemiHMMModel.load(tmp_path / "model")
    assert reloaded.logreg_class_weight == "balanced"
    assert reloaded._emission_hmm.logreg_class_weight == "balanced"
    np.testing.assert_array_equal(model._emission_hmm._logreg.coef_, reloaded._emission_hmm._logreg.coef_)
    for traj in trajectories[:3]:
        np.testing.assert_allclose(model.filtering(traj), reloaded.filtering(traj), atol=1e-10)


def test_logreg_class_weight_missing_key_in_old_state_defaults_to_none(tmp_path):
    trajectories = make_separable_dataset(n_states=3, n_videos=10, seed=23)
    model = SemiHMMModel(n_states=3, dmax=8)
    model.fit(trajectories)
    model.save(tmp_path / "model")
    state_path = tmp_path / "model" / "state.json"
    state = json.loads(state_path.read_text())
    assert "logreg_class_weight" in state
    del state["logreg_class_weight"]
    state_path.write_text(json.dumps(state))
    # emission_hmm's own nested state.json also needs the key removed to
    # fully simulate a pre-2026-08-23 artifact end to end.
    emission_state_path = tmp_path / "model" / "emission_hmm" / "state.json"
    emission_state = json.loads(emission_state_path.read_text())
    del emission_state["logreg_class_weight"]
    emission_state_path.write_text(json.dumps(emission_state))

    reloaded = SemiHMMModel.load(tmp_path / "model")
    assert reloaded.logreg_class_weight is None
    assert reloaded._emission_hmm.logreg_class_weight is None


def test_logreg_class_weight_balanced_no_nan_inf_and_causal():
    trajectories = _make_imbalanced_dataset(seed=24, n_majority_videos=20, n_minority_videos=3)
    model = SemiHMMModel(n_states=2, dmax=25, logreg_class_weight="balanced")
    model.fit(trajectories)
    for traj in trajectories[:5]:
        filt = model.filtering(traj)
        assert not np.any(np.isnan(filt)) and not np.any(np.isinf(filt))
        probs = model.predict_k_step_transition_probability(traj, k=1)
        assert not np.any(np.isnan(probs)) and not np.any(np.isinf(probs))

    rng = np.random.RandomState(25)
    centers = np.array([[0.0, 0.0], [30.0, 0.0]])
    prefix_states = [0, 0, 0, 1, 1]
    prefix = (centers[prefix_states] + rng.normal(scale=0.3, size=(5, 2))).tolist()
    traj_a = make_trajectory("a", prefix + [[30.0, 0.0]], prefix_states + [1])
    traj_b = make_trajectory("b", prefix + [[30.0, 0.0], [30.0, 0.0]], prefix_states + [1, 1])
    filt_a = model.filtering(traj_a)
    filt_b = model.filtering(traj_b)
    np.testing.assert_allclose(filt_a[:5], filt_b[:5], atol=1e-10)
