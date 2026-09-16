"""
The ONE real-data integration check requested by this phase's brief:
loads the actual frozen Semi-HMM (Results/evaluation/semi_hmm_weekend_phaseF/model/)
and one real Val trajectory (Patient_319 -- already used as the worked
example throughout docs/SCIENTIFIC_REPORT.md/PROJECT_CHECKPOINT_SEMI_HMM.md),
and exercises the full stack end-to-end: model_loader -> trajectory_service
-> inference_service -> schemas.

Skips cleanly (not a failure) if the real artifacts aren't present on this
machine -- this repo's own convention (CLAUDE.md) is that the local
checkout may not have the server-synced Results//Embeddings/ trees at
all. On the GPU server, where they exist, this test runs for real.

TEST = LOCKED: this file NEVER passes split="test" anywhere, and
Embeddings/resnet18/test/ is never referenced.
"""

from pathlib import Path

import pytest

from reporting import inference_service, model_loader, trajectory_service

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODEL_EXISTS = (model_loader.REFERENCE_MODEL_PATH / "state.json").exists()
_VAL_EMBEDDINGS_EXIST = (trajectory_service.EMBEDDINGS_ROOT / "val" / "manifest.json").exists()

pytestmark = pytest.mark.skipif(
    not (_MODEL_EXISTS and _VAL_EMBEDDINGS_EXIST),
    reason="Real frozen model / Val embeddings not present on this machine "
           "(expected on the GPU server, not necessarily in a local checkout).",
)


@pytest.fixture(autouse=True)
def _reset_caches():
    inference_service._POSTERIOR_CACHE.clear()
    inference_service._NEXT_PHASE_CACHE.clear()
    inference_service._DURATION_INFO_CACHE.clear()
    model_loader._MODEL_CACHE = None
    trajectory_service._TRAJECTORY_CACHE.clear()
    trajectory_service._TRAJECTORY_INDEX.clear()
    trajectory_service._TIME_METADATA_CACHE.clear()
    yield


def test_real_model_loads_and_validates():
    model = model_loader.get_model()
    assert model.n_states == 15
    assert model.dmax == 268
    assert model.duration_family == "negative_binomial"
    assert model.logreg_class_weight is None


def test_real_val_split_lists_106_videos():
    videos = trajectory_service.list_videos("val")
    assert len(videos) == 106


def test_real_test_split_is_refused():
    with pytest.raises(trajectory_service.TestSplitLockedError):
        trajectory_service.list_videos("test")


def test_real_trajectory_loads_for_a_known_video():
    videos = trajectory_service.list_videos("val")
    traj = trajectory_service.get_trajectory(videos[0], "val")
    assert len(traj) > 0
    assert traj.embeddings.shape[1] == 512


def test_real_infer_on_first_window_of_first_val_video():
    videos = trajectory_service.list_videos("val")
    video_name = videos[0]
    window_start = trajectory_service.get_windows(video_name, "val")[0]
    record = inference_service.infer(video_name, "val", window_start)

    assert record.current_state.current_phase in record.current_state.phase_probabilities
    total = sum(record.current_state.phase_probabilities.values())
    assert abs(total - 1.0) < 1e-6
    next_total = sum(record.next_phase.next_phase_distribution.values())
    assert abs(next_total - 1.0) < 1e-6
    assert record.model.model_configuration["dmax"] == 268
    assert record.window.time_unit == "unknown/unverified"


def test_real_infer_duration_unavailable_for_tPB2_if_current_phase():
    """Not every window will land on tPB2, so this test asserts the
    INVARIANT (if current_phase is tPB2 or tEB, duration_available must be
    False) rather than forcing a specific window.

    Deliberately bounded to 2 DISTINCT videos, window_start=0 only:
    infer() internally calls SemiHMMModel.next_phase_distribution(), whose
    existing (frozen, not touched here) implementation is a pure-Python
    O(T*K*Dmax) loop -- measured directly on this server this session at
    ~333s for a single 527-window Val trajectory. Per-video results are
    cached after the first call, but EACH NEW video costs a fresh ~5+
    minutes. A first version of this test scanned 15 videos and ran for
    over an hour before being caught and fixed -- see
    docs/REPORTING_API_IMPLEMENTATION_REPORT.md sec 13 (Performance) for
    the full writeup. window_start=0 is highly likely to land on tPB2
    (the dataset's dominant first-observed phase, ~81% of Train videos
    per docs/PROJECT_STATE.md's Phase 2 measurements), so 2 videos is
    enough to exercise this path without an expensive broad scan."""
    videos = trajectory_service.list_videos("val")
    found_censored_phase = False
    for video_name in videos[:2]:
        # window_start is a GLOBAL index into the split's flattened window
        # list (see docs/INFERENCE_SCHEMA.md), NOT a per-video offset --
        # it does not reset to 0 for every video (only the very first
        # video in the whole split's flattened order has a window at 0;
        # confirmed the hard way: Patient_367's first real window_start is
        # not 0, which failed an earlier version of this test). Always use
        # this trajectory's own first window.
        first_window = trajectory_service.get_windows(video_name, "val")[0]
        record = inference_service.infer(video_name, "val", first_window)
        if record.current_state.current_phase in ("tPB2", "tEB"):
            found_censored_phase = True
            assert record.duration.duration_available is False
            assert record.duration.expected_duration is None
    # Not a hard requirement that both sampled videos hit tPB2/tEB -- the
    # real assertion above already fires if one does; this is a smoke
    # check bounded for cost, not a guarantee of coverage.
    del found_censored_phase


def test_real_trajectory_summary_and_transition_chain():
    videos = trajectory_service.list_videos("val")
    summary = inference_service.trajectory_summary(videos[0], "val")
    assert summary.n_windows == len(trajectory_service.get_trajectory(videos[0], "val"))
    assert isinstance(summary.transition_chain, list)
    if summary.transition_chain:
        first_segment = summary.transition_chain[0]
        assert "phase" in first_segment
        assert "observed_duration_windows" in first_segment


def test_real_forward_pass_is_cached_across_windows():
    videos = trajectory_service.list_videos("val")
    video_name = videos[0]
    windows = trajectory_service.get_windows(video_name, "val")
    inference_service.infer(video_name, "val", windows[0])
    cached = inference_service._POSTERIOR_CACHE[(video_name, "val")]
    inference_service.infer(video_name, "val", windows[min(1, len(windows) - 1)])
    assert inference_service._POSTERIOR_CACHE[(video_name, "val")] is cached
