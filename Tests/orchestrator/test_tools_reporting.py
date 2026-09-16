"""
Integration tests for the Reporting-API-backed tools (get_current_inference,
get_trajectory, get_model_metadata) -- require torch AND the real frozen
Semi-HMM checkpoint / Val embeddings, skip cleanly otherwise (matches
Tests/reporting/test_integration_real_val.py's own convention exactly:
this repo's own local checkout may not have the server-synced Results//
Embeddings/ trees, CLAUDE.md).

TEST = LOCKED: this file never passes split="test" as a valid case --
the one split="test" test asserts it is REJECTED (ToolInputError), never
that it succeeds.
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from orchestrator import tools  # noqa: E402
from reporting import model_loader, trajectory_service  # noqa: E402

_MODEL_EXISTS = (model_loader.REFERENCE_MODEL_PATH / "state.json").exists()
_VAL_EMBEDDINGS_EXIST = (trajectory_service.EMBEDDINGS_ROOT / "val" / "manifest.json").exists()

pytestmark = pytest.mark.skipif(
    not (_MODEL_EXISTS and _VAL_EMBEDDINGS_EXIST),
    reason="Real frozen model / Val embeddings not present on this machine "
           "(expected on the GPU server, not necessarily in a local checkout).",
)


@pytest.fixture(autouse=True)
def _reset_caches():
    from reporting import inference_service

    inference_service._POSTERIOR_CACHE.clear()
    inference_service._NEXT_PHASE_CACHE.clear()
    model_loader._MODEL_CACHE = None
    trajectory_service._TRAJECTORY_CACHE.clear()
    trajectory_service._TRAJECTORY_INDEX.clear()
    trajectory_service._TIME_METADATA_CACHE.clear()
    yield


def _first_val_video_and_window():
    videos = trajectory_service.list_videos("val")
    video_id = videos[0]
    window = trajectory_service.get_windows(video_id, "val")[0]
    return video_id, window


def test_get_current_inference_returns_the_same_shape_as_inference_service():
    video_id, window = _first_val_video_and_window()
    result = tools.get_current_inference(video_id, window, split="val")
    assert result["sample"]["video_name"] == video_id
    assert result["window"]["window_start"] == window
    assert "current_state" in result and "phase_probabilities" in result["current_state"]


def test_get_current_inference_rejects_test_split():
    video_id, window = _first_val_video_and_window()
    with pytest.raises(tools.ToolInputError):
        tools.get_current_inference(video_id, window, split="test")


def test_get_current_inference_unknown_video_raises_not_found():
    with pytest.raises(tools.ToolNotFoundError):
        tools.get_current_inference("Patient_does_not_exist", 0, split="val")


def test_get_trajectory_returns_transition_chain():
    video_id, _ = _first_val_video_and_window()
    result = tools.get_trajectory(video_id, split="val")
    assert result["sample"]["video_name"] == video_id
    assert "transition_chain" in result


def test_get_trajectory_rejects_test_split():
    video_id, _ = _first_val_video_and_window()
    with pytest.raises(tools.ToolInputError):
        tools.get_trajectory(video_id, split="test")


def test_get_model_metadata_matches_the_reference_configuration():
    result = tools.get_model_metadata()
    assert result["model_name"] == "semi_hmm"
    assert result["model_configuration"]["dmax"] == 268
    assert result["model_configuration"]["duration_family"] == "negative_binomial"


# --- get_transition_events (Event/Transition RAG axis, P1,
# docs/EVENT_TRANSITION_RAG_IMPLEMENTATION.md) -- real integration tests,
# same skip-guard convention as every other tool in this file.

def test_get_transition_events_returns_observed_only_transitions():
    video_id, _ = _first_val_video_and_window()
    result = tools.get_transition_events(video_id, split="val")
    assert result["video_id"] == video_id
    assert isinstance(result["transitions"], list)
    assert result["n_transitions"] == len(result["transitions"])
    for t in result["transitions"]:
        assert t["observed"] is True
        assert t["provenance"] == "observed_annotation"
        assert "next_phase_distribution" not in t
        assert "duration_probability_at_observed" not in t


def test_get_transition_events_rejects_test_split():
    video_id, _ = _first_val_video_and_window()
    with pytest.raises(tools.ToolInputError):
        tools.get_transition_events(video_id, split="test")


def test_get_transition_events_unknown_video_raises_not_found():
    with pytest.raises(tools.ToolNotFoundError):
        tools.get_transition_events("Patient_does_not_exist", split="val")


def test_get_transition_events_window_filter_matches_unfiltered_bracket():
    video_id, _ = _first_val_video_and_window()
    all_events = tools.get_transition_events(video_id, split="val")["transitions"]
    if not all_events:
        pytest.skip(f"{video_id} has no observed phase transition in split=val to filter around.")
    first = all_events[0]
    filtered = tools.get_transition_events(video_id, split="val", window=first["window_start"])
    assert filtered["n_transitions"] >= 1
    assert any(t["from_phase"] == first["from_phase"] and t["to_phase"] == first["to_phase"]
               for t in filtered["transitions"])


# --- get_inference_history (multi-window / G1), real Val data ---------------

def test_get_inference_history_returns_ordered_band_with_separated_provenance():
    video_id = trajectory_service.list_videos("val")[0]
    windows = trajectory_service.get_windows(video_id, "val")
    center = windows[len(windows) // 2]
    result = tools.get_inference_history(video_id, center, before=3, after=3, split="val")
    assert result["sample"]["video_name"] == video_id
    assert result["center_window"] == center
    starts = result["window_starts"]
    assert starts == sorted(starts)  # temporally ordered
    assert center in starts
    for e in result["entries"]:
        assert set(e["model_derived"]) >= {"current_phase", "phase_probabilities", "entropy",
                                            "next_phase_distribution"}
        assert set(e["observed"]) == {"ground_truth_phase", "consistency_flag"}
        assert abs(sum(e["model_derived"]["phase_probabilities"].values()) - 1.0) < 1e-6
        assert e["model_derived"]["entropy"] >= 0.0
    centers = [e for e in result["entries"] if e["is_center"]]
    assert len(centers) == 1 and centers[0]["window_start"] == center


def test_get_inference_history_matches_get_current_inference_per_window():
    video_id = trajectory_service.list_videos("val")[0]
    windows = trajectory_service.get_windows(video_id, "val")
    center = windows[len(windows) // 2]
    hist = tools.get_inference_history(video_id, center, before=2, after=2, split="val")
    for e in hist["entries"]:
        single = tools.get_current_inference(video_id, e["window_start"], split="val")
        assert e["model_derived"]["current_phase"] == single["current_state"]["current_phase"]
        assert e["model_derived"]["phase_probabilities"] == single["current_state"]["phase_probabilities"]
        assert e["observed"]["ground_truth_phase"] == single["ground_truth"]["ground_truth_phase"]


def test_get_inference_history_clamps_at_trajectory_start():
    video_id = trajectory_service.list_videos("val")[0]
    first = trajectory_service.get_windows(video_id, "val")[0]
    result = tools.get_inference_history(video_id, first, before=5, after=3, split="val")
    assert result["n_windows_before"] == 0
    assert result["window_starts"][0] == first
    assert any("before" in w for w in result["warnings"])


def test_get_inference_history_rejects_test_split():
    video_id = trajectory_service.list_videos("val")[0]
    center = trajectory_service.get_windows(video_id, "val")[0]
    with pytest.raises(tools.ToolInputError):
        tools.get_inference_history(video_id, center, before=2, after=2, split="test")


def test_get_inference_history_unknown_window_raises_not_found():
    video_id = trajectory_service.list_videos("val")[0]
    with pytest.raises(tools.ToolNotFoundError):
        tools.get_inference_history(video_id, 10_000_000, before=2, after=2, split="val")


def test_get_inference_history_rejects_nonsensical_band():
    video_id = trajectory_service.list_videos("val")[0]
    center = trajectory_service.get_windows(video_id, "val")[0]
    with pytest.raises(tools.ToolInputError):
        tools.get_inference_history(video_id, center, before=0, after=0, split="val")


def test_get_inference_history_stays_within_one_video():
    videos = trajectory_service.list_videos("val")
    video_id = videos[0]
    windows = trajectory_service.get_windows(video_id, "val")
    win_set = set(windows)
    center = windows[len(windows) // 2]
    result = tools.get_inference_history(video_id, center, before=1000, after=1000, split="val")
    returned = result["window_starts"]
    assert set(returned).issubset(win_set)  # never a window from another video
    # contiguous slice of THIS video's real window list
    lo = windows.index(returned[0])
    assert windows[lo:lo + len(returned)] == returned


# --- ETAPE 4: context compactness, against REAL Val data ---

def test_get_inference_history_top_k_truncates_on_real_data():
    video_id = trajectory_service.list_videos("val")[0]
    windows = trajectory_service.get_windows(video_id, "val")
    center = windows[len(windows) // 2]
    full = tools.get_inference_history(video_id, center, before=3, after=3, split="val")
    cut = tools.get_inference_history(video_id, center, before=3, after=3, split="val",
                                      top_k_probabilities=4)
    n_full = len(full["entries"][0]["model_derived"]["phase_probabilities"])
    assert n_full > 4, "real model has more phases than the budget -- otherwise this proves nothing"
    for fe, ce in zip(full["entries"], cut["entries"]):
        fm, cm = fe["model_derived"], ce["model_derived"]
        assert len(cm["phase_probabilities"]) == 4
        assert len(cm["next_phase_distribution"]) == 4
        # kept entries are the true leaders, verbatim
        leaders = [k for k, _ in sorted(fm["phase_probabilities"].items(),
                                        key=lambda kv: (-kv[1], str(kv[0])))[:4]]
        assert sorted(cm["phase_probabilities"]) == sorted(leaders)
        for k in cm["phase_probabilities"]:
            assert cm["phase_probabilities"][k] == fm["phase_probabilities"][k]
        # full-distribution scalars survive untouched
        assert cm["entropy"] == fm["entropy"]
        assert cm["phase_probability"] == fm["phase_probability"]
        assert cm["current_phase"] == fm["current_phase"]
        # observed annotation never touched
        assert ce["observed"] == fe["observed"]
    assert any("truncated" in w for w in cut["warnings"])


def test_get_inference_history_top_k_materially_shrinks_the_payload():
    # the whole point of ETAPE 4 -- prove the compaction is real, not cosmetic.
    import json
    video_id = trajectory_service.list_videos("val")[0]
    windows = trajectory_service.get_windows(video_id, "val")
    center = windows[len(windows) // 2]
    full = tools.get_inference_history(video_id, center, before=5, after=5, split="val")
    cut = tools.get_inference_history(video_id, center, before=5, after=5, split="val",
                                      top_k_probabilities=4)
    # ~39% off the raw JSON here; the saving on the FLATTENED prompt the LLM
    # actually receives is considerably larger, because prompt.py gives every
    # dropped probability its own long dotted-path line.
    assert len(json.dumps(cut)) < 0.75 * len(json.dumps(full))
    assert cut["n_windows"] == full["n_windows"]  # no window was dropped
