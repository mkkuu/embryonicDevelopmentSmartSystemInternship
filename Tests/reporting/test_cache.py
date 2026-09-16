"""
Cache-mechanics tests, isolated from model correctness (already covered
elsewhere) via a FakeModel that counts how many times its expensive
methods are actually called -- the property under test is "at most once
per (model_version, split, video)", not "the numbers are right" (that's
tested against the real model in test_integration_real_val.py).

TEST = LOCKED note: cache.py itself is split-agnostic -- the guard against
split="test" happens upstream, in trajectory_service._check_split(),
called before _forward_pass() (and therefore before this cache) is ever
reached (see inference_service.infer()). That guard is already tested
exhaustively in test_trajectory_service.py/test_api.py and is unchanged
by this phase; not re-tested here to avoid duplicating coverage.
"""

import fcntl
import json

import numpy as np
import pytest

from reporting import cache as reporting_cache


class FakeTrajectory:
    def __init__(self, window_starts):
        self.window_starts = list(window_starts)


class FakeModel:
    """Records how many times its expensive methods were actually
    invoked -- the cache's whole job is to keep this at <= 1 per
    (model_version, split, video) across repeated calls."""

    def __init__(self, posterior, next_dist):
        self._posterior = posterior
        self._next_dist = next_dist
        self.filtering_calls = 0
        self.next_phase_calls = 0

    def filtering(self, traj):
        self.filtering_calls += 1
        return self._posterior

    def next_phase_distribution(self, traj):
        self.next_phase_calls += 1
        return self._next_dist


@pytest.fixture(autouse=True)
def _isolated_cache_root(tmp_path, monkeypatch):
    """Every test gets its own throwaway cache directory -- never touches
    the real Cache/ tree, and tests never interfere with each other."""
    monkeypatch.setattr(reporting_cache, "CACHE_ROOT", tmp_path / "cache_root")
    yield


def _make(posterior_val=0.5, k=4, t=6):
    posterior = np.full((t, k), posterior_val)
    next_dist = np.full((t, k), 1.0 / k)
    model = FakeModel(posterior, next_dist)
    traj = FakeTrajectory(window_starts=list(range(t)))
    return model, traj


def test_cache_absent_initially_is_a_miss():
    model, traj = _make()
    assert not reporting_cache.cache_entry_exists("semi_hmm", "v1", "val", "PatientX")
    posterior, next_dist, stats = reporting_cache.get_or_compute_forward_pass(
        model, "PatientX", "val", traj, "semi_hmm", "v1")
    assert stats.hit is False
    assert model.filtering_calls == 1
    assert model.next_phase_calls == 1
    assert reporting_cache.cache_entry_exists("semi_hmm", "v1", "val", "PatientX")


def test_second_request_is_a_cache_hit_and_does_not_recompute():
    model, traj = _make()
    reporting_cache.get_or_compute_forward_pass(model, "PatientX", "val", traj, "semi_hmm", "v1")
    posterior2, next_dist2, stats2 = reporting_cache.get_or_compute_forward_pass(
        model, "PatientX", "val", traj, "semi_hmm", "v1")
    assert stats2.hit is True
    assert model.filtering_calls == 1  # NOT incremented on the second call
    assert model.next_phase_calls == 1


def test_cached_result_is_numerically_identical_to_computed_result():
    model, traj = _make(posterior_val=0.3141592653589793, k=5, t=10)
    posterior_miss, next_dist_miss, _ = reporting_cache.get_or_compute_forward_pass(
        model, "PatientX", "val", traj, "semi_hmm", "v1")
    posterior_hit, next_dist_hit, _ = reporting_cache.get_or_compute_forward_pass(
        model, "PatientX", "val", traj, "semi_hmm", "v1")
    assert np.array_equal(posterior_miss, posterior_hit)
    assert np.array_equal(next_dist_miss, next_dist_hit)


def test_different_model_version_is_not_reused():
    model, traj = _make()
    reporting_cache.get_or_compute_forward_pass(model, "PatientX", "val", traj, "semi_hmm", "v1")
    reporting_cache.get_or_compute_forward_pass(model, "PatientX", "val", traj, "semi_hmm", "v2-balanced")
    assert model.filtering_calls == 2  # each version computed independently, never cross-reused
    assert model.next_phase_calls == 2


def test_different_split_is_not_reused():
    model, traj = _make()
    reporting_cache.get_or_compute_forward_pass(model, "PatientX", "val", traj, "semi_hmm", "v1")
    reporting_cache.get_or_compute_forward_pass(model, "PatientX", "train", traj, "semi_hmm", "v1")
    assert model.filtering_calls == 2


def test_changed_window_starts_invalidates_cache():
    """If the trajectory itself changed (e.g. a re-extracted embeddings
    cache with different filtering), a cache entry keyed by an old
    window_starts list must never be silently served for the new one."""
    model, traj = _make()
    reporting_cache.get_or_compute_forward_pass(model, "PatientX", "val", traj, "semi_hmm", "v1")
    traj.window_starts = [0, 1, 2, 3, 4, 5, 6]  # different from what was cached
    reporting_cache.get_or_compute_forward_pass(model, "PatientX", "val", traj, "semi_hmm", "v1")
    assert model.filtering_calls == 2


def test_corrupted_cache_file_is_treated_as_a_miss_not_a_crash():
    model, traj = _make()
    reporting_cache.get_or_compute_forward_pass(model, "PatientX", "val", traj, "semi_hmm", "v1")
    cache_file = reporting_cache._cache_dir("semi_hmm", "v1", "val", "PatientX") / "inference.json"
    cache_file.write_text("{not valid json...")
    posterior, next_dist, stats = reporting_cache.get_or_compute_forward_pass(
        model, "PatientX", "val", traj, "semi_hmm", "v1")
    assert stats.hit is False
    assert model.filtering_calls == 2  # recomputed, not crashed
    # and it self-heals: the corrupted file is overwritten with a valid one
    assert json.loads(cache_file.read_text())["model_version"] == "v1"


def test_cache_missing_required_field_is_treated_as_a_miss():
    model, traj = _make()
    reporting_cache.get_or_compute_forward_pass(model, "PatientX", "val", traj, "semi_hmm", "v1")
    cache_file = reporting_cache._cache_dir("semi_hmm", "v1", "val", "PatientX") / "inference.json"
    data = json.loads(cache_file.read_text())
    del data["next_phase_distribution"]
    cache_file.write_text(json.dumps(data))
    _, _, stats = reporting_cache.get_or_compute_forward_pass(model, "PatientX", "val", traj, "semi_hmm", "v1")
    assert stats.hit is False
    assert model.filtering_calls == 2


def test_different_video_names_are_independent():
    model, traj = _make()
    reporting_cache.get_or_compute_forward_pass(model, "PatientA", "val", traj, "semi_hmm", "v1")
    reporting_cache.get_or_compute_forward_pass(model, "PatientB", "val", traj, "semi_hmm", "v1")
    assert model.filtering_calls == 2
    reporting_cache.get_or_compute_forward_pass(model, "PatientA", "val", traj, "semi_hmm", "v1")
    assert model.filtering_calls == 2  # PatientA was already cached


def test_lock_is_released_after_a_completed_call():
    """A non-blocking flock attempt right after the call must succeed,
    proving the writer lock isn't held past the request that created it
    (otherwise every subsequent request for this video would hang)."""
    model, traj = _make()
    reporting_cache.get_or_compute_forward_pass(model, "PatientX", "val", traj, "semi_hmm", "v1")
    lock_path = reporting_cache._cache_dir("semi_hmm", "v1", "val", "PatientX") / "inference.json.lock"
    with open(lock_path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)  # raises BlockingIOError if still held
        fcntl.flock(fh, fcntl.LOCK_UN)


def test_atomic_write_leaves_no_tmp_file_on_success():
    model, traj = _make()
    reporting_cache.get_or_compute_forward_pass(model, "PatientX", "val", traj, "semi_hmm", "v1")
    cache_dir = reporting_cache._cache_dir("semi_hmm", "v1", "val", "PatientX")
    assert (cache_dir / "inference.json").exists()
    assert not (cache_dir / "inference.json.tmp").exists()


def test_cache_stats_report_load_and_compute_times():
    model, traj = _make()
    _, _, miss_stats = reporting_cache.get_or_compute_forward_pass(model, "PatientX", "val", traj, "semi_hmm", "v1")
    _, _, hit_stats = reporting_cache.get_or_compute_forward_pass(model, "PatientX", "val", traj, "semi_hmm", "v1")
    assert miss_stats.compute_time_seconds >= 0.0
    assert miss_stats.hit is False
    assert hit_stats.compute_time_seconds == 0.0  # nothing computed on a hit
    assert hit_stats.hit is True
