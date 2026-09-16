"""
Derives OBSERVED phase transitions from a trajectory's ground-truth
annotation sequence -- Event/Transition RAG axis, P1
(docs/EVENT_ANOMALY_RAG_ANALYSIS.md sec 16 items 1-2,
docs/EVENT_TRANSITION_RAG_IMPLEMENTATION.md). Reporting/orchestration-
layer composition ONLY: reuses `HMMModel._segments()` (already a
cross-module-reused classmethod -- `SemiHMMModel.transition_chain()`
calls it the exact same way, `Training/evaluation/models/semi_hmm.py`)
and `Trajectory.window_starts`/`last_frame_phase` (both already-existing,
public, read-only fields) -- never reimplements the run-length
segmentation, never touches HMM/Semi-HMM training logic, never refits
anything.

Direct cleavage / reversal / fragmentation / multinucleation are NEVER
computed or referenced anywhere in this module -- confirmed absent from
this project's real data (docs/EVENT_ANOMALY_RAG_ANALYSIS.md sec 3). This
module derives ONLY what the ground-truth phase-segment sequence can
actually support: observed phase transitions and their "skip"
classification (|phase index delta| > 1).

tHB (docs/EVENT_ANOMALY_RAG_ANALYSIS.md sec 15): the loaded model's own
`state_names` (15 entries, `DEFAULT_PHASE_NAMES`) never includes "tHB" --
any window whose real annotation would have been "tHB" was already
excluded upstream, at `Embryo_Transition_Dataset`/embedding-build time,
so it can never appear in `Trajectory.last_frame_phase` and this module
never has to special-case it. NOT fixed here -- a pre-existing,
documented limitation of the whole pipeline, out of this task's scope.

TEST = LOCKED, same guard as every other `Training/reporting/` module --
every function here that accepts a `split` refuses "test" immediately via
`trajectory_service._check_split()`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.models.hmm import HMMModel  # noqa: E402

from . import frame_mapping, model_loader, trajectory_service  # noqa: E402
from .schemas import TransitionEvent  # noqa: E402


def _frame_bounds(video_name: str, split: str, window_start: int, window_end: int):
    """Best-effort real-frame lookup for the two bracketing windows --
    never fabricated, never crashes the whole call. Returns
    (frame_start, frame_end, frame_mapping_available). frame_mapping.py
    needs raw `Data/` on disk (annotations + JPEGs); when that isn't
    present in this environment (e.g. this repo's own bare local
    checkout, CLAUDE.md), FrameMappingError is caught here and the caller
    gets an honest (None, None, False), matching this project's
    "explicit fallback, never a silent no-op" discipline (same pattern as
    `DurationInfo.duration_available`/`WindowInfo.time_available`)."""
    try:
        start_frames = frame_mapping.window_frame_numbers(video_name, split, window_start)
        end_frames = frame_mapping.window_frame_numbers(video_name, split, window_end)
        return start_frames[-1], end_frames[0], True
    except frame_mapping.FrameMappingError:
        return None, None, False


def transition_events(video_name: str, split: str, window: Optional[int] = None) -> List[TransitionEvent]:
    """All OBSERVED phase transitions in this video's ground-truth
    sequence, each carrying `observed=True`/`provenance="observed_annotation"`
    and NO model-derived field (docs/EVENT_ANOMALY_RAG_ANALYSIS.md sec 6 --
    a caller wanting the model's own view calls get_current_inference
    separately). When `window` is given, only transitions whose
    [window_start, window_end] bracket contains it are returned (an empty
    list, not an error, if none does -- "no transition observed around
    this window" is a legitimate, honest answer, not a failure).

    RETROSPECTIVE (needs the full trajectory), same as
    `inference_service.trajectory_summary()` -- never served for a
    partial/live sample."""
    trajectory_service._check_split(split)
    model = model_loader.get_model()
    traj = trajectory_service.get_trajectory(video_name, split)
    phase_names = model.state_names

    segments = HMMModel._segments(traj)
    events: List[TransitionEvent] = []
    for (phase_a, _start_a, end_a), (phase_b, start_b, _end_b) in zip(segments, segments[1:]):
        window_start = traj.window_starts[end_a]
        window_end = traj.window_starts[start_b]
        frame_start, frame_end, frame_available = _frame_bounds(video_name, split, window_start, window_end)
        time_a = trajectory_service.get_window_time(video_name, split, window_start)
        time_b = trajectory_service.get_window_time(video_name, split, window_end)
        phase_distance = phase_b - phase_a
        events.append(TransitionEvent(
            from_phase=phase_names[phase_a], to_phase=phase_names[phase_b],
            from_phase_index=phase_a, to_phase_index=phase_b,
            observed=True, provenance="observed_annotation",
            window_start=window_start, window_end=window_end,
            frame_start=frame_start, frame_end=frame_end, frame_mapping_available=frame_available,
            window_start_time=time_a["window_start_time"], window_end_time=time_b["window_end_time"],
            time_available=bool(time_a["time_available"] and time_b["time_available"]),
            phase_distance=phase_distance, is_skip=abs(phase_distance) > 1,
        ))

    if window is not None:
        events = [e for e in events if e.window_start <= window <= e.window_end]
    return events
