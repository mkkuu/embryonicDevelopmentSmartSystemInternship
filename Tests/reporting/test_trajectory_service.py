"""
Split-lock guard tests (must hold for EVERY public function, not just
one) and get_window_time logic tests via a monkeypatched, hand-built
metadata DataFrame -- avoids needing a real embeddings cache on disk for
these specific behaviors. list_videos/get_trajectory/get_windows/
get_ground_truth against a REAL cache are covered by
test_integration_real_val.py (Val only, never Test), per this project's
existing convention of synthetic unit tests + one real integration check.
"""

import pandas as pd
import pytest

from reporting import trajectory_service as ts


@pytest.fixture(autouse=True)
def _reset_time_index_cache():
    # _time_row_index() (docs/WEBAPP_TIMELINE_PERFORMANCE.md) caches per
    # split, built lazily from whatever _load_time_metadata() returns at
    # first use -- since most tests below monkeypatch _load_time_metadata
    # itself with a fresh fake DataFrame each time, the index cache must be
    # cleared between tests too, or a later test would silently see an
    # earlier test's fake data instead of its own.
    ts._TIME_ROW_INDEX_CACHE.clear()
    yield
    ts._TIME_ROW_INDEX_CACHE.clear()


@pytest.mark.parametrize("fn,args", [
    (ts.list_videos, ("test",)),
    (ts.get_trajectory, ("Patient_1", "test")),
    (ts.get_windows, ("Patient_1", "test")),
    (ts.get_ground_truth, ("Patient_1", "test", 0)),
    (ts._load_time_metadata, ("test",)),
    (ts.get_window_time, ("Patient_1", "test", 0)),
])
def test_every_public_function_refuses_test_split(fn, args):
    with pytest.raises(ts.TestSplitLockedError):
        fn(*args)


def test_check_split_rejects_unknown_split_value():
    with pytest.raises(ValueError):
        ts._check_split("holdout")


def test_check_split_accepts_val_and_train_case_insensitively():
    assert ts._check_split("VAL") == "val"
    assert ts._check_split("Train") == "train"


def _fake_time_df():
    return pd.DataFrame([
        {"video_name": "Patient_1", "window_start": 0, "window_start_time": 1.0, "window_end_time": 2.0,
         "window_mid_time": 1.5, "window_duration": 1.0, "time_data_complete": True,
         "n_zero_time_diffs_in_window": 0},
        {"video_name": "Patient_1", "window_start": 1, "window_start_time": 2.0, "window_end_time": 3.0,
         "window_mid_time": 2.5, "window_duration": 1.0, "time_data_complete": False,
         "n_zero_time_diffs_in_window": 0},
        {"video_name": "Patient_1", "window_start": 2, "window_start_time": 3.0, "window_end_time": 3.0,
         "window_mid_time": 3.0, "window_duration": 0.0, "time_data_complete": True,
         "n_zero_time_diffs_in_window": 3},
    ])


def test_get_window_time_returns_values_for_complete_row(monkeypatch):
    monkeypatch.setattr(ts, "_load_time_metadata", lambda split: _fake_time_df())
    result = ts.get_window_time("Patient_1", "val", 0)
    assert result["time_available"] is True
    assert result["window_end_time"] == 2.0
    assert result["time_unit"] == "unknown/unverified"  # never a named unit


def test_get_window_time_marks_incomplete_row_unavailable_not_zero(monkeypatch):
    monkeypatch.setattr(ts, "_load_time_metadata", lambda split: _fake_time_df())
    result = ts.get_window_time("Patient_1", "val", 1)
    assert result["time_available"] is False
    assert result["window_end_time"] is None  # never a fabricated 0.0


def test_get_window_time_missing_row_returns_unavailable(monkeypatch):
    monkeypatch.setattr(ts, "_load_time_metadata", lambda split: _fake_time_df())
    result = ts.get_window_time("Patient_1", "val", 999)
    assert result["time_available"] is False


def test_get_window_time_no_csv_at_all_returns_unavailable(monkeypatch):
    monkeypatch.setattr(ts, "_load_time_metadata", lambda split: None)
    result = ts.get_window_time("Patient_1", "val", 0)
    assert result["time_available"] is False
    assert result["time_unit"] == "unknown/unverified"


def test_get_window_time_surfaces_repeated_timestamp_artifact_count(monkeypatch):
    monkeypatch.setattr(ts, "_load_time_metadata", lambda split: _fake_time_df())
    result = ts.get_window_time("Patient_1", "val", 2)
    assert result["n_zero_time_diffs_in_window"] == 3  # exposed, not hidden or reinterpreted


# --- _time_row_index (docs/WEBAPP_TIMELINE_PERFORMANCE.md) -------------------

def test_time_row_index_matches_get_window_time_for_every_row(monkeypatch):
    # The index must produce, for every real row, exactly what get_window_time()
    # itself returns for that row -- proving the indexed lookup is not a
    # different computation, just a different access path.
    monkeypatch.setattr(ts, "_load_time_metadata", lambda split: _fake_time_df())
    for window_start in (0, 1, 2):
        via_lookup = ts.get_window_time("Patient_1", "val", window_start)
        expected = _fake_time_df()
        row = expected[expected["window_start"] == window_start].iloc[0]
        if bool(row["time_data_complete"]):
            assert via_lookup == {
                "window_start_time": float(row["window_start_time"]),
                "window_end_time": float(row["window_end_time"]),
                "window_mid_time": float(row["window_mid_time"]),
                "window_duration": float(row["window_duration"]),
                "time_available": True, "time_unit": "unknown/unverified",
                "n_zero_time_diffs_in_window": int(row["n_zero_time_diffs_in_window"]),
            }
        else:
            assert via_lookup["time_available"] is False


def test_time_row_index_is_computed_once_then_cached(monkeypatch):
    calls = {"n": 0}

    def counting_load(split):
        calls["n"] += 1
        return _fake_time_df()

    monkeypatch.setattr(ts, "_load_time_metadata", counting_load)
    ts.get_window_time("Patient_1", "val", 0)
    ts.get_window_time("Patient_1", "val", 1)
    ts.get_window_time("Patient_1", "val", 2)
    assert calls["n"] == 1, "_load_time_metadata must be called once per split, not once per window"


def test_time_row_index_does_not_leak_stale_data_across_a_monkeypatch_swap(monkeypatch):
    # Regression test for the exact risk this cache introduces: if a caller
    # swaps what _load_time_metadata returns (or, in production, if a
    # process were ever restarted with different source data), the index
    # MUST be rebuilt from the new data once cleared -- never silently
    # keep serving the old DataFrame's values.
    monkeypatch.setattr(ts, "_load_time_metadata", lambda split: _fake_time_df())
    first = ts.get_window_time("Patient_1", "val", 0)
    assert first["window_end_time"] == 2.0

    different_df = pd.DataFrame([
        {"video_name": "Patient_1", "window_start": 0, "window_start_time": 10.0, "window_end_time": 99.0,
         "window_mid_time": 50.0, "window_duration": 89.0, "time_data_complete": True,
         "n_zero_time_diffs_in_window": 0},
    ])
    monkeypatch.setattr(ts, "_load_time_metadata", lambda split: different_df)
    ts._TIME_ROW_INDEX_CACHE.clear()  # what a real caller must do -- the autouse fixture already
    # does this between tests; this test proves the cache actually honors that clear.
    second = ts.get_window_time("Patient_1", "val", 0)
    assert second["window_end_time"] == 99.0
