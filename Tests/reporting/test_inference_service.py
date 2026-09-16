"""
Synthetic, fully offline tests of the inference composition logic --
monkeypatches model_loader.get_model() and trajectory_service so infer()
runs its ENTIRE composition path (filtering -> current_state,
next_phase_distribution -> next_phase, duration_models -> duration,
ground truth, warnings) against a small synthetic model, with no real
embeddings cache or frozen checkpoint on disk. This is deliberately the
most heavily tested module: it is where every scientific method call is
composed, and where a wiring mistake (wrong index, wrong axis, wrong
argmax) would silently produce a plausible-looking but wrong record.
"""

import numpy as np
import pytest
import torch

from evaluation.models.semi_hmm import SemiHMMModel
from evaluation.trajectory import Trajectory
from reporting import inference_service, model_loader, trajectory_service


def make_trajectory(video_name, embeddings, phases, window_starts=None):
    embeddings_t = torch.as_tensor(embeddings, dtype=torch.float32)
    phases_t = torch.as_tensor(phases, dtype=torch.long)
    return Trajectory(
        video_name=video_name,
        window_starts=window_starts if window_starts is not None else list(range(len(phases))),
        embeddings=embeddings_t,
        consistency_flag=torch.cat([torch.zeros(1, dtype=torch.long),
                                     (phases_t[1:] != phases_t[:-1]).long()]) if len(phases) > 1
        else torch.zeros(len(phases), dtype=torch.long),
        first_frame_phase=phases_t.clone(),
        last_frame_phase=phases_t,
    )


@pytest.fixture
def tiny_fitted_model_and_traj():
    """4 states, every video runs 0->1->2->3 monotonically. State 0 is
    always the FIRST segment and state 3 always the LAST segment across
    every training video -- exactly mirrors tPB2/tEB's real structural
    censoring (0 observed interior segments), so duration_available=False
    is exercised for a genuine, not contrived, reason."""
    rng = np.random.RandomState(0)
    n_states = 4
    train = []
    for v in range(10):
        reps = []
        for s in range(n_states):
            reps.extend([s] * rng.randint(3, 6))
        embeddings = rng.normal(scale=0.3, size=(len(reps), 6)) + np.array(reps)[:, None] * 5.0
        train.append(make_trajectory(f"train{v}", embeddings, reps))
    model = SemiHMMModel(n_states=n_states, dmax=30, duration_family="negative_binomial")
    model.fit(train)

    val_reps = [0, 0, 0, 1, 1, 2, 2, 2, 3, 3]
    val_embeddings = rng.normal(scale=0.3, size=(len(val_reps), 6)) + np.array(val_reps)[:, None] * 5.0
    val_traj = make_trajectory("val0", val_embeddings, val_reps)
    return model, val_traj


@pytest.fixture(autouse=True)
def _reset_caches(tmp_path, monkeypatch):
    # Isolate the disk cache (Phase 1.5) to a throwaway directory so these
    # synthetic tests never write into the real Cache/ tree.
    from reporting import cache as reporting_cache
    monkeypatch.setattr(reporting_cache, "CACHE_ROOT", tmp_path / "cache_root")
    inference_service._POSTERIOR_CACHE.clear()
    inference_service._NEXT_PHASE_CACHE.clear()
    inference_service._DURATION_INFO_CACHE.clear()
    model_loader._MODEL_CACHE = None
    yield
    inference_service._POSTERIOR_CACHE.clear()
    inference_service._NEXT_PHASE_CACHE.clear()
    inference_service._DURATION_INFO_CACHE.clear()
    model_loader._MODEL_CACHE = None


def _wire_synthetic(monkeypatch, model, traj):
    monkeypatch.setattr(model_loader, "get_model", lambda: model)
    monkeypatch.setattr(trajectory_service, "get_trajectory", lambda video_name, split: traj)
    monkeypatch.setattr(trajectory_service, "get_window_time",
                         lambda video_name, split, window_start: {
                             "window_start_time": None, "window_end_time": None, "window_mid_time": None,
                             "window_duration": None, "time_available": False,
                             "time_unit": "unknown/unverified", "n_zero_time_diffs_in_window": None,
                         })
    monkeypatch.setattr(trajectory_service, "get_ground_truth",
                         lambda video_name, split, window_start: (
                             int(traj.last_frame_phase[traj.window_starts.index(window_start)]), 0
                         ))
    # model.state_names defaults to ["S0","S1","S2","S3"] for a 4-state model with no
    # explicit state_names (SemiHMMModel.__init__), which infer() now reads directly --
    # no monkeypatch needed here, unlike before the PHASE_NAMES hardcoding fix.


def test_infer_phase_probabilities_sum_to_one(monkeypatch, tiny_fitted_model_and_traj):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    record = inference_service.infer("val0", "val", 0)
    total = sum(record.current_state.phase_probabilities.values())
    assert abs(total - 1.0) < 1e-6


def test_infer_next_phase_distribution_sums_to_one(monkeypatch, tiny_fitted_model_and_traj):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    record = inference_service.infer("val0", "val", 3)
    total = sum(record.next_phase.next_phase_distribution.values())
    assert abs(total - 1.0) < 1e-6


def test_infer_current_phase_is_argmax_of_phase_probabilities(monkeypatch, tiny_fitted_model_and_traj):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    record = inference_service.infer("val0", "val", 5)
    best = max(record.current_state.phase_probabilities, key=record.current_state.phase_probabilities.get)
    assert record.current_state.current_phase == best
    assert record.current_state.phase_probability == record.current_state.phase_probabilities[best]


def test_infer_entropy_is_nonnegative(monkeypatch, tiny_fitted_model_and_traj):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    record = inference_service.infer("val0", "val", 0)
    assert record.current_state.entropy >= 0.0


def test_infer_duration_unavailable_for_structurally_censored_first_state(monkeypatch, tiny_fitted_model_and_traj):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    record = inference_service.infer("val0", "val", 0)  # window 0 -> phase S0, structurally censored
    assert record.duration.duration_available is False
    assert record.duration.expected_duration is None  # never fabricated as 0.0
    assert any("structural censoring" in w for w in record.warnings)


def test_infer_duration_available_for_interior_state(monkeypatch, tiny_fitted_model_and_traj):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    record = inference_service.infer("val0", "val", 5)  # window 5 -> phase S2, an interior state
    assert record.duration.duration_available is True
    assert record.duration.expected_duration is not None
    assert record.duration.duration_quantiles is not None


def test_infer_raises_on_unknown_window(monkeypatch, tiny_fitted_model_and_traj):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    with pytest.raises(inference_service.WindowNotFoundError):
        inference_service.infer("val0", "val", 9999)


# --- _duration_info memoization (docs/WEBAPP_TIMELINE_PERFORMANCE.md) --------
# _duration_info's result depends ONLY on (model, phase_index) -- these tests
# prove the memoization added for that report is both effective (avoids
# recomputation) and safe (never leaks a stale/wrong result across models).

def test_duration_info_is_computed_once_per_phase_and_then_cached(monkeypatch, tiny_fitted_model_and_traj):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    calls = {"n": 0}
    real_expected_duration = model.expected_duration

    def counting_expected_duration(phase_index):
        calls["n"] += 1
        return real_expected_duration(phase_index)

    monkeypatch.setattr(model, "expected_duration", counting_expected_duration)

    inference_service.infer("val0", "val", 5)  # phase S2 (interior, duration_available) -- first call, computes
    assert calls["n"] == 1
    inference_service.infer("val0", "val", 5)  # identical window/phase again -- must hit the cache, not recompute
    assert calls["n"] == 1


def test_duration_info_cache_is_process_global_like_the_existing_posterior_caches():
    # Documents a real, deliberate design characteristic, not a bug: this
    # cache is keyed by phase_index ALONE, exactly matching
    # _POSTERIOR_CACHE/_NEXT_PHASE_CACHE's own already-established "one
    # model per process" invariant (model_loader.get_model() loads once,
    # never swaps -- see _forward_pass's docstring). Two DIFFERENT models
    # sharing phase_index=0, with genuinely different fitted duration
    # distributions, prove the point concretely: without an explicit
    # clear() between them, the second model incorrectly receives the
    # first model's cached result -- exactly why every fixture in this
    # file (and test_integration_real_val.py) clears this cache alongside
    # the other two on every test boundary.
    inference_service._DURATION_INFO_CACHE.clear()
    try:
        rng = np.random.RandomState(1)
        # 3 states, phase_index=1 is an INTERIOR state (never first/last across every
        # training video), so it has an estimable duration -- state 0/2 would be
        # structurally censored regardless of scale, same reasoning
        # tiny_fitted_model_and_traj's own docstring already explains.

        def make_train(interior_duration_bias):
            trajs = []
            for v in range(8):
                reps = ([0] * rng.randint(2, 4) + [1] * (rng.randint(2, 5) + interior_duration_bias)
                        + [2] * rng.randint(2, 4))
                embeddings = rng.normal(scale=0.3, size=(len(reps), 4)) + np.array(reps)[:, None] * 5.0
                trajs.append(make_trajectory(f"train{v}", embeddings, reps))
            return trajs

        model_a = SemiHMMModel(n_states=3, dmax=20, duration_family="negative_binomial")
        model_a.fit(make_train(interior_duration_bias=0))
        model_b = SemiHMMModel(n_states=3, dmax=20, duration_family="negative_binomial")
        model_b.fit(make_train(interior_duration_bias=6))  # much longer state-1 segments -> real difference

        # Establish ground truth for each model independently first.
        fresh_a = inference_service._duration_info(model_a, 1)
        inference_service._DURATION_INFO_CACHE.clear()
        fresh_b = inference_service._duration_info(model_b, 1)
        assert fresh_a.duration_available and fresh_b.duration_available
        assert fresh_a.expected_duration != fresh_b.expected_duration, (
            "test setup invalid -- the two models must have genuinely different "
            "fitted durations for this test to demonstrate anything"
        )

        # Now show the actual cache-sharing behavior: querying model_a then
        # model_b WITHOUT clearing in between reuses model_a's entry.
        inference_service._DURATION_INFO_CACHE.clear()
        info_a = inference_service._duration_info(model_a, 1)
        info_b_uncleared = inference_service._duration_info(model_b, 1)
        assert info_b_uncleared.expected_duration == info_a.expected_duration
        assert info_b_uncleared.expected_duration != fresh_b.expected_duration
    finally:
        inference_service._DURATION_INFO_CACHE.clear()


def test_infer_ground_truth_matches_trajectory(monkeypatch, tiny_fitted_model_and_traj):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    record = inference_service.infer("val0", "val", 8)  # phase S3
    assert record.ground_truth.ground_truth_phase == "S3"


def test_infer_caches_forward_pass_across_repeated_calls(monkeypatch, tiny_fitted_model_and_traj):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    inference_service.infer("val0", "val", 0)
    assert ("val0", "val") in inference_service._POSTERIOR_CACHE
    cached = inference_service._POSTERIOR_CACHE[("val0", "val")]
    inference_service.infer("val0", "val", 1)
    assert inference_service._POSTERIOR_CACHE[("val0", "val")] is cached  # not recomputed


def test_infer_time_unavailable_produces_warning(monkeypatch, tiny_fitted_model_and_traj):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    record = inference_service.infer("val0", "val", 0)
    assert record.window.time_available is False
    assert any("time metadata unavailable" in w for w in record.warnings)


# ============================================================================
# infer_history -- the multi-window / G1 capability
# (docs/RAG_FIXED_QUESTION_BENCHMARK.md §7). Same synthetic wiring as above:
# no real embeddings cache / checkpoint, forward pass reused from the cache.
# ============================================================================


def test_infer_history_returns_a_band_of_multiple_windows(monkeypatch, tiny_fitted_model_and_traj):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    h = inference_service.infer_history("val0", "val", 5, before=2, after=2)
    assert h.n_windows == 5
    assert [e.window_start for e in h.entries] == [3, 4, 5, 6, 7]
    assert h.window_starts == [3, 4, 5, 6, 7]
    assert h.center_window == 5
    assert h.n_windows_before == 2 and h.n_windows_after == 2


def test_infer_history_entries_are_temporally_ordered_with_offsets(monkeypatch, tiny_fitted_model_and_traj):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    h = inference_service.infer_history("val0", "val", 5, before=3, after=1)
    starts = [e.window_start for e in h.entries]
    assert starts == sorted(starts)
    assert [e.offset_from_center for e in h.entries] == [-3, -2, -1, 0, 1]


def test_infer_history_exactly_one_center_entry_at_the_center_window(monkeypatch, tiny_fitted_model_and_traj):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    h = inference_service.infer_history("val0", "val", 4, before=2, after=2)
    centers = [e for e in h.entries if e.is_center]
    assert len(centers) == 1
    assert centers[0].window_start == 4
    assert centers[0].offset_from_center == 0


def test_infer_history_per_window_probabilities_sum_to_one_and_entropy_nonnegative(
    monkeypatch, tiny_fitted_model_and_traj
):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    h = inference_service.infer_history("val0", "val", 5, before=3, after=3)
    for e in h.entries:
        assert abs(sum(e.phase_probabilities.values()) - 1.0) < 1e-6
        assert abs(sum(e.next_phase_distribution.values()) - 1.0) < 1e-6
        assert e.entropy >= 0.0
        assert e.current_phase == max(e.phase_probabilities, key=e.phase_probabilities.get)


def test_infer_history_entry_matches_single_window_infer_for_the_same_window(
    monkeypatch, tiny_fitted_model_and_traj
):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    h = inference_service.infer_history("val0", "val", 5, before=2, after=2)
    for e in h.entries:
        rec = inference_service.infer("val0", "val", e.window_start)
        assert e.current_phase == rec.current_state.current_phase
        assert e.phase_probabilities == rec.current_state.phase_probabilities
        assert e.entropy == rec.current_state.entropy
        assert e.next_phase_distribution == rec.next_phase.next_phase_distribution
        assert e.ground_truth_phase == rec.ground_truth.ground_truth_phase
        assert e.consistency_flag == rec.ground_truth.consistency_flag


def test_infer_history_observed_and_model_derived_are_separated_in_to_dict(
    monkeypatch, tiny_fitted_model_and_traj
):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    d = inference_service.infer_history("val0", "val", 5, before=1, after=1).to_dict()
    entry = d["entries"][0]
    assert set(entry["model_derived"]) >= {"current_phase", "phase_probabilities", "entropy",
                                            "next_phase_distribution"}
    assert set(entry["observed"]) == {"ground_truth_phase", "consistency_flag"}
    # no model field leaks into the observed block and vice-versa
    assert "phase_probabilities" not in entry["observed"]
    assert "ground_truth_phase" not in entry["model_derived"]


def test_infer_history_clamps_at_start_of_trajectory_with_warning(monkeypatch, tiny_fitted_model_and_traj):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    h = inference_service.infer_history("val0", "val", 1, before=5, after=2)
    assert h.n_windows_before == 1  # only windows 0 and 1 exist before/at center 1 -> 1 before
    assert h.n_windows_after == 2
    assert [e.window_start for e in h.entries] == [0, 1, 2, 3]
    assert any("before" in w and "start of this video" in w for w in h.warnings)


def test_infer_history_clamps_at_end_of_trajectory_with_warning(monkeypatch, tiny_fitted_model_and_traj):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    h = inference_service.infer_history("val0", "val", 8, before=2, after=5)
    assert [e.window_start for e in h.entries] == [6, 7, 8, 9]
    assert h.n_windows_after == 1
    assert any("after" in w and "end of this video" in w for w in h.warnings)


def test_infer_history_never_leaks_windows_from_another_video(monkeypatch, tiny_fitted_model_and_traj):
    model, traj = tiny_fitted_model_and_traj
    # global (across-videos) window_starts: this video occupies 500..509 only
    shifted = make_trajectory("val0", traj.embeddings.numpy(),
                               traj.last_frame_phase.numpy(), window_starts=list(range(500, 510)))
    _wire_synthetic(monkeypatch, model, shifted)
    h = inference_service.infer_history("val0", "val", 505, before=1000, after=1000)
    assert all(500 <= w <= 509 for w in h.window_starts)
    assert h.window_starts == list(range(500, 510))  # clamped to this video's bounds, nothing outside it
    assert h.n_windows == 10
    assert any("per-side cap" in w for w in h.warnings)


def test_infer_history_unknown_center_window_raises(monkeypatch, tiny_fitted_model_and_traj):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    with pytest.raises(inference_service.WindowNotFoundError):
        inference_service.infer_history("val0", "val", 9999, before=2, after=2)


@pytest.mark.parametrize("before,after", [(-1, 2), (2, -1), (0, 0)])
def test_infer_history_rejects_a_nonsensical_band(monkeypatch, tiny_fitted_model_and_traj, before, after):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    with pytest.raises(ValueError):
        inference_service.infer_history("val0", "val", 5, before=before, after=after)


def test_infer_history_rejects_test_split(monkeypatch, tiny_fitted_model_and_traj):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    with pytest.raises(trajectory_service.TestSplitLockedError):
        inference_service.infer_history("val0", "test", 5, before=2, after=2)


def test_infer_history_reuses_the_cached_forward_pass_no_recompute(monkeypatch, tiny_fitted_model_and_traj):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    inference_service.infer("val0", "val", 5)  # warm the cache
    cached = inference_service._POSTERIOR_CACHE[("val0", "val")]
    h = inference_service.infer_history("val0", "val", 5, before=4, after=4)
    assert h.n_windows == 9
    assert inference_service._POSTERIOR_CACHE[("val0", "val")] is cached  # same array object, not recomputed


def test_infer_history_flags_time_unavailable_per_entry_and_globally(monkeypatch, tiny_fitted_model_and_traj):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)  # this wiring returns time_available=False for every window
    h = inference_service.infer_history("val0", "val", 5, before=2, after=2)
    assert all(e.time_available is False for e in h.entries)
    assert all(e.window_start_time is None for e in h.entries)  # never fabricated
    assert any("time metadata unavailable" in w for w in h.warnings)


def test_infer_history_carries_real_times_when_available(monkeypatch, tiny_fitted_model_and_traj):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    monkeypatch.setattr(trajectory_service, "get_window_time",
                         lambda v, s, w: {
                             "window_start_time": float(w), "window_end_time": float(w) + 1.5,
                             "window_mid_time": float(w) + 0.75, "window_duration": 1.5,
                             "time_available": True, "time_unit": "unknown/unverified",
                             "n_zero_time_diffs_in_window": 0,
                         })
    h = inference_service.infer_history("val0", "val", 5, before=2, after=2)
    assert [e.window_start_time for e in h.entries] == [3.0, 4.0, 5.0, 6.0, 7.0]
    assert all(e.time_available for e in h.entries)
    assert not any("time metadata unavailable" in w for w in h.warnings)


def test_infer_still_works_unchanged_alongside_the_new_capability(monkeypatch, tiny_fitted_model_and_traj):
    # conservation guard: the pre-existing single-window path is untouched.
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    rec = inference_service.infer("val0", "val", 5)
    assert rec.current_state.current_phase == max(
        rec.current_state.phase_probabilities, key=rec.current_state.phase_probabilities.get
    )
    summary = inference_service.trajectory_summary("val0", "val")
    assert summary.n_windows == 10


# ---------------------------------------------------------------------------
# ETAPE 4 -- top_k_probabilities (context compactness).
# The band is what the LLM actually sees; a full 15-entry distribution per
# window per field blew past the deployment's 4096-token ctx-size. These
# pin the contract: truncation keeps the LEADING phases verbatim, never
# perturbs the full-distribution scalars, discloses itself, and is OFF by
# default so every pre-existing caller is byte-identical.
# ---------------------------------------------------------------------------

def test_infer_history_full_distribution_by_default(monkeypatch, tiny_fitted_model_and_traj):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    h = inference_service.infer_history("val0", "val", 5, before=2, after=2)
    n_phases = len(h.entries[0].phase_probabilities)
    for e in h.entries:
        assert len(e.phase_probabilities) == n_phases
        assert len(e.next_phase_distribution) == n_phases
    assert not any("truncated" in w for w in h.warnings)


def test_infer_history_top_k_truncates_both_distributions(monkeypatch, tiny_fitted_model_and_traj):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    h = inference_service.infer_history("val0", "val", 5, before=2, after=2, top_k_probabilities=2)
    for e in h.entries:
        assert len(e.phase_probabilities) <= 2
        assert len(e.next_phase_distribution) <= 2


def test_infer_history_top_k_keeps_the_highest_probability_phases(monkeypatch, tiny_fitted_model_and_traj):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    full = inference_service.infer_history("val0", "val", 5, before=2, after=2)
    cut = inference_service.infer_history("val0", "val", 5, before=2, after=2, top_k_probabilities=3)
    for fe, ce in zip(full.entries, cut.entries):
        expected = [k for k, _ in sorted(fe.phase_probabilities.items(),
                                         key=lambda kv: (-kv[1], str(kv[0])))[:3]]
        assert sorted(ce.phase_probabilities) == sorted(expected)
        # kept values are verbatim, not renormalised or rounded
        for k in ce.phase_probabilities:
            assert ce.phase_probabilities[k] == fe.phase_probabilities[k]


def test_infer_history_top_k_does_not_perturb_full_distribution_scalars(
        monkeypatch, tiny_fitted_model_and_traj):
    # entropy / phase_probability / next_phase_probability are computed
    # upstream from the FULL distribution -- truncation must not move them.
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    full = inference_service.infer_history("val0", "val", 5, before=2, after=2)
    cut = inference_service.infer_history("val0", "val", 5, before=2, after=2, top_k_probabilities=1)
    for fe, ce in zip(full.entries, cut.entries):
        assert ce.entropy == fe.entropy
        assert ce.phase_probability == fe.phase_probability
        assert ce.next_phase_probability == fe.next_phase_probability
        assert ce.current_phase == fe.current_phase
        assert ce.most_likely_next_phase == fe.most_likely_next_phase


def test_infer_history_top_k_discloses_truncation_in_warnings(monkeypatch, tiny_fitted_model_and_traj):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    h = inference_service.infer_history("val0", "val", 5, before=2, after=2, top_k_probabilities=1)
    assert any("truncated" in w and "entropy" in w for w in h.warnings)


def test_infer_history_top_k_larger_than_the_distribution_is_a_noop(
        monkeypatch, tiny_fitted_model_and_traj):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    full = inference_service.infer_history("val0", "val", 5, before=1, after=1)
    wide = inference_service.infer_history("val0", "val", 5, before=1, after=1,
                                           top_k_probabilities=10_000)
    for fe, we in zip(full.entries, wide.entries):
        assert we.phase_probabilities == fe.phase_probabilities
    assert not any("truncated" in w for w in wide.warnings)


@pytest.mark.parametrize("bad_k", [0, -1])
def test_infer_history_rejects_nonsensical_top_k(monkeypatch, tiny_fitted_model_and_traj, bad_k):
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    with pytest.raises(ValueError):
        inference_service.infer_history("val0", "val", 5, before=2, after=2,
                                        top_k_probabilities=bad_k)


def test_infer_history_top_k_preserves_band_shape_and_order(monkeypatch, tiny_fitted_model_and_traj):
    # compaction is about distribution WIDTH, never about dropping windows:
    # temporal order and band size must be identical to the full call.
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    full = inference_service.infer_history("val0", "val", 5, before=3, after=2)
    cut = inference_service.infer_history("val0", "val", 5, before=3, after=2, top_k_probabilities=2)
    assert cut.window_starts == full.window_starts == sorted(full.window_starts)
    assert [e.offset_from_center for e in cut.entries] == [e.offset_from_center for e in full.entries]
    assert sum(e.is_center for e in cut.entries) == 1


def test_infer_history_top_k_keeps_observed_block_intact(monkeypatch, tiny_fitted_model_and_traj):
    # the OBSERVED annotation block is not a distribution and must never be
    # touched by a prompt-budget policy.
    model, traj = tiny_fitted_model_and_traj
    _wire_synthetic(monkeypatch, model, traj)
    full = inference_service.infer_history("val0", "val", 5, before=2, after=2)
    cut = inference_service.infer_history("val0", "val", 5, before=2, after=2, top_k_probabilities=1)
    for fe, ce in zip(full.entries, cut.entries):
        assert ce.to_dict()["observed"] == fe.to_dict()["observed"]
