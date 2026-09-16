"""
Synthetic, fully offline tests for Training/reporting/transition_events.py
-- the Event/Transition RAG axis's P1 derivation (OBSERVED phase
transitions + "skip" classification from ground-truth annotation
segments only, docs/EVENT_TRANSITION_RAG_IMPLEMENTATION.md). Same pattern
as Tests/reporting/test_inference_service.py: monkeypatches
model_loader.get_model()/trajectory_service against a small synthetic
model/trajectory, no real embeddings cache or frozen checkpoint needed.

Never touches HMM/Semi-HMM training logic -- transition_events() only
calls HMMModel._segments() (a pure, already-tested classmethod) and reads
Trajectory's own public fields.
"""

import torch

from evaluation.trajectory import Trajectory
from reporting import frame_mapping, model_loader, trajectory_service, transition_events as te_module


class _FakeModel:
    """Minimal stand-in -- transition_events() only ever reads
    `model.state_names` from the loaded model, never anything else."""

    def __init__(self, state_names):
        self.state_names = state_names


def make_trajectory(video_name, phases, window_starts):
    phases_t = torch.as_tensor(phases, dtype=torch.long)
    return Trajectory(
        video_name=video_name,
        window_starts=list(window_starts),
        embeddings=torch.zeros((len(phases), 4), dtype=torch.float32),
        consistency_flag=torch.zeros(len(phases), dtype=torch.long),
        first_frame_phase=phases_t.clone(),
        last_frame_phase=phases_t,
    )


def _wire(monkeypatch, model, traj, time_available=False):
    monkeypatch.setattr(model_loader, "get_model", lambda: model)
    monkeypatch.setattr(trajectory_service, "get_trajectory", lambda video_name, split: traj)
    if time_available:
        monkeypatch.setattr(
            trajectory_service, "get_window_time",
            lambda video_name, split, window_start: {
                "window_start_time": float(window_start) * 0.1, "window_end_time": float(window_start) * 0.1 + 0.05,
                "window_mid_time": float(window_start) * 0.1 + 0.025, "window_duration": 0.05,
                "time_available": True, "time_unit": "unknown/unverified", "n_zero_time_diffs_in_window": 0,
            },
        )
    else:
        monkeypatch.setattr(
            trajectory_service, "get_window_time",
            lambda video_name, split, window_start: {
                "window_start_time": None, "window_end_time": None, "window_mid_time": None,
                "window_duration": None, "time_available": False,
                "time_unit": "unknown/unverified", "n_zero_time_diffs_in_window": None,
            },
        )


# Phases: t2(0,0,0) -> t3(1,1) -> t6(2,2) ; window_starts 100..106 (7 windows).
# Segments: [t2: rows 0-2], [t3: rows 3-4], [t6: rows 5-6].
# Transition 1: t2 -> t3, phase_distance=1 (not a skip).
# Transition 2: t3 -> t6, phase_distance=3 (a skip: t4/t5 never annotated for
# this synthetic video -- t4/t5 are still real, valid canonical phases, just
# absent from this one video's ground truth, which is exactly what a real
# skip transition means).
PHASE_NAMES = ["tPB2", "tPNa", "tPNf", "t2", "t3", "t4", "t5", "t6"]
# indices: tPB2=0 tPNa=1 tPNf=2 t2=3 t3=4 t4=5 t5=6 t6=7
PHASES = [3, 3, 3, 4, 4, 7, 7]
WINDOW_STARTS = [100, 101, 102, 103, 104, 105, 106]


def test_normal_transition_is_not_flagged_as_skip(monkeypatch):
    model = _FakeModel(PHASE_NAMES)
    traj = make_trajectory("val0", PHASES, WINDOW_STARTS)
    _wire(monkeypatch, model, traj)

    events = te_module.transition_events("val0", "val")
    assert len(events) == 2
    first = events[0]
    assert first.from_phase == "t2" and first.to_phase == "t3"
    assert first.phase_distance == 1
    assert first.is_skip is False
    assert first.observed is True
    assert first.provenance == "observed_annotation"


def test_skip_transition_is_flagged(monkeypatch):
    model = _FakeModel(PHASE_NAMES)
    traj = make_trajectory("val0", PHASES, WINDOW_STARTS)
    _wire(monkeypatch, model, traj)

    events = te_module.transition_events("val0", "val")
    second = events[1]
    assert second.from_phase == "t3" and second.to_phase == "t6"
    assert second.phase_distance == 3
    assert second.is_skip is True


def test_window_bounds_bracket_the_transition_correctly(monkeypatch):
    model = _FakeModel(PHASE_NAMES)
    traj = make_trajectory("val0", PHASES, WINDOW_STARTS)
    _wire(monkeypatch, model, traj)

    events = te_module.transition_events("val0", "val")
    # t2 -> t3: last t2 window is row 2 (window_start=102), first t3 window is row 3 (window_start=103).
    assert events[0].window_start == 102
    assert events[0].window_end == 103
    # t3 -> t6: last t3 window is row 4 (window_start=104), first t6 window is row 5 (window_start=105).
    assert events[1].window_start == 104
    assert events[1].window_end == 105


def test_negative_phase_distance_is_never_assumed_impossible(monkeypatch):
    """This project's own frozen Semi-HMM training data showed ZERO
    backward transitions across all 704 real videos (hmm.py, Phase 2) --
    but transition_events() must never special-case or assume this for
    an arbitrary trajectory; a synthetic backward sequence must be
    reported honestly, not crash or get silently reordered."""
    model = _FakeModel(PHASE_NAMES)
    backward_phases = [4, 4, 3, 3]  # t3 -> t2, backward (index 4 -> 3)
    traj = make_trajectory("val_backward", backward_phases, [200, 201, 202, 203])
    _wire(monkeypatch, model, traj)

    events = te_module.transition_events("val_backward", "val")
    assert len(events) == 1
    assert events[0].from_phase == "t3" and events[0].to_phase == "t2"
    assert events[0].phase_distance == -1
    assert events[0].is_skip is False  # abs(-1) == 1, not a skip


def test_no_transitions_for_a_single_segment_trajectory(monkeypatch):
    model = _FakeModel(PHASE_NAMES)
    traj = make_trajectory("val_flat", [3, 3, 3, 3], [100, 101, 102, 103])
    _wire(monkeypatch, model, traj)

    events = te_module.transition_events("val_flat", "val")
    assert events == []


def test_window_filter_returns_only_the_bracketing_transition(monkeypatch):
    model = _FakeModel(PHASE_NAMES)
    traj = make_trajectory("val0", PHASES, WINDOW_STARTS)
    _wire(monkeypatch, model, traj)

    events_near_first = te_module.transition_events("val0", "val", window=102)
    assert len(events_near_first) == 1
    assert events_near_first[0].from_phase == "t2"

    events_near_second = te_module.transition_events("val0", "val", window=105)
    assert len(events_near_second) == 1
    assert events_near_second[0].from_phase == "t3"


def test_window_filter_with_no_bracketing_transition_returns_empty_list(monkeypatch):
    model = _FakeModel(PHASE_NAMES)
    traj = make_trajectory("val_flat", [3, 3, 3, 3], [100, 101, 102, 103])
    _wire(monkeypatch, model, traj)

    events = te_module.transition_events("val_flat", "val", window=101)
    assert events == []


def test_frame_bounds_unavailable_when_frame_mapping_fails(monkeypatch):
    model = _FakeModel(PHASE_NAMES)
    traj = make_trajectory("val0", PHASES, WINDOW_STARTS)
    _wire(monkeypatch, model, traj)

    def _raise(*a, **kw):
        raise frame_mapping.AnnotationsNotFoundError("no annotation file for this synthetic video")

    monkeypatch.setattr(frame_mapping, "window_frame_numbers", _raise)

    events = te_module.transition_events("val0", "val")
    assert events[0].frame_mapping_available is False
    assert events[0].frame_start is None
    assert events[0].frame_end is None


def test_frame_bounds_available_when_frame_mapping_succeeds(monkeypatch):
    model = _FakeModel(PHASE_NAMES)
    traj = make_trajectory("val0", PHASES, WINDOW_STARTS)
    _wire(monkeypatch, model, traj)

    def fake_window_frame_numbers(video_name, split, window_start):
        return list(range(window_start, window_start + 8))

    monkeypatch.setattr(frame_mapping, "window_frame_numbers", fake_window_frame_numbers)

    events = te_module.transition_events("val0", "val")
    assert events[0].frame_mapping_available is True
    assert events[0].frame_start == 102 + 7  # last frame of window_start=102's 8-frame window
    assert events[0].frame_end == 103  # first frame of window_end=103's 8-frame window


def test_time_available_when_both_bracketing_windows_have_time_data(monkeypatch):
    model = _FakeModel(PHASE_NAMES)
    traj = make_trajectory("val0", PHASES, WINDOW_STARTS)
    _wire(monkeypatch, model, traj, time_available=True)

    events = te_module.transition_events("val0", "val")
    assert events[0].time_available is True
    assert events[0].window_start_time is not None
    assert events[0].window_end_time is not None


def test_split_test_is_refused():
    from reporting.trajectory_service import TestSplitLockedError

    try:
        te_module.transition_events("val0", "test")
        assert False, "expected TestSplitLockedError"
    except TestSplitLockedError:
        pass


def test_unknown_video_raises_key_error(monkeypatch):
    model = _FakeModel(PHASE_NAMES)

    def _raise_unknown(video_name, split):
        raise KeyError(f"video_name={video_name!r} not found in split={split!r}.")

    monkeypatch.setattr(model_loader, "get_model", lambda: model)
    monkeypatch.setattr(trajectory_service, "get_trajectory", _raise_unknown)

    try:
        te_module.transition_events("Patient_999999", "val")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_to_dict_shape(monkeypatch):
    model = _FakeModel(PHASE_NAMES)
    traj = make_trajectory("val0", PHASES, WINDOW_STARTS)
    _wire(monkeypatch, model, traj)

    events = te_module.transition_events("val0", "val")
    d = events[0].to_dict()
    assert d["observed"] is True
    assert d["provenance"] == "observed_annotation"
    assert set(d.keys()) == {
        "from_phase", "to_phase", "from_phase_index", "to_phase_index", "observed", "provenance",
        "window_start", "window_end", "frame_start", "frame_end", "frame_mapping_available",
        "window_start_time", "window_end_time", "time_available", "phase_distance", "is_skip",
    }
    # No model-derived field anywhere -- the whole point of keeping this
    # dataclass observation-only (docs/EVENT_ANOMALY_RAG_ANALYSIS.md sec 6).
    assert "next_phase_distribution" not in d
    assert "duration_probability_at_observed" not in d
