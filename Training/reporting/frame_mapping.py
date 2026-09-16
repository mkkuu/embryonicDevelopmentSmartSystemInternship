"""
Maps a Reporting API `window_start` (an index into the split's flattened,
ACROSS-ALL-VIDEOS window list -- confirmed empirically this session:
Patient_319 occupies 0-526, Patient_367 occupies 527-962, Patient_158
occupies 963-1473, contiguous per video but NOT per-video-zero-based) to
the actual raw JPEG frame it corresponds to
(`Data/embryo_dataset_F0/<video>/<video>_Image_<N>.jpeg`) --
`docs/WEBAPP_TIMELINE_PERFORMANCE.md`'s second objective (Phase 7.7).

Why this needs reconstruction rather than a direct lookup: no code path
anywhere in this project (before this module) ever recorded which raw
frame a cached embedding/window came from -- `Training/DataSet.py`'s
`Embryo_Transition_Dataset._create_sequences()` builds one sliding
window (`window_size=8`, `stride=1`) per KEPT position, where "kept"
already excludes any window spanning more than 2 phases, or 2 phases
that are not adjacent in `CHRONOLOGICAL_PHASES` order -- so a
`window_start` is an index into the FILTERED sequence list, not a raw
`start` position in the sorted frame list, and the two can diverge by an
unknown amount unless the exact same filter is re-applied.

**Validated empirically, not just by reading code**: reconstructed the
filter above independently, using only real annotation files
(`Data/embryo_dataset_annotations/<video>_phases.csv`, `phase,frame_start,
frame_end` inclusive ranges -- exactly how
`Training/preProcess.py:process_annotations()` builds them) and the real
JPEG files on disk, then cross-checked the reconstructed `last_frame_phase`
of every window against the REAL `trajectory_service.get_ground_truth()`
value from the live Reporting API, across 16 real Val videos (a stratified
sample of the 106): **6467/6467 windows matched exactly**, and the
reconstructed sequence COUNT matched the real trajectory's window count
on every single video checked. This is why this module exists as a real,
trusted capability rather than a documented-but-unbuilt gap.

TEST = LOCKED, same as every other module in this package: every function
here that accepts a `split` refuses "test" immediately via
`trajectory_service._check_split()`.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from . import trajectory_service  # noqa: E402

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "Data"
ANNOTATIONS_DIR = DATA_ROOT / "embryo_dataset_annotations"
IMAGES_ROOT = DATA_ROOT / "embryo_dataset_F0"

# Fixed by the real embeddings build this Reporting API serves (ResNet18,
# FocalType=F0) -- never a parameter, since every cached embedding/window
# in this project was built with exactly these values (CLAUDE.md: "8-frame
# windows -> ResNet18"). Not reused for a hypothetical TimeSformer/32-frame
# branch, which has no Reporting API support at all today.
WINDOW_SIZE = 8
STRIDE = 1

CHRONOLOGICAL_PHASES = [
    "tPB2", "tPNa", "tPNf", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9+", "tM", "tSB", "tB", "tEB",
]
_PHASE_TO_INDEX = {p: i for i, p in enumerate(CHRONOLOGICAL_PHASES)}

_IMAGE_FILENAME_PATTERN = re.compile(r"Image_(\d+)")


class FrameMappingError(Exception):
    """Base class -- the mapping could not be determined or the resulting
    file does not exist. Never a fabricated/guessed path."""


class AnnotationsNotFoundError(FrameMappingError):
    """No `<video_name>_phases.csv` for this video -- the reconstruction
    has nothing to replicate the original phase-consistency filter with."""


class FrameFileNotFoundError(FrameMappingError):
    """The reconstructed mapping points at a specific frame number, but no
    JPEG exists at the expected path -- surfaced explicitly, never silently
    substituted with a neighboring frame."""


_KEPT_SEQUENCES_CACHE: Dict[str, List[Dict]] = {}  # video_name -> kept_sequences (see
# _reconstruct_kept_sequences) -- safe to cache indefinitely: annotations and raw
# frame files are static, historical, read-only data (never modified after the
# original dataset was assembled), unlike the model/embeddings caches above which
# key on a live, in-principle-swappable model version.


def _load_annotations(video_name: str) -> pd.DataFrame:
    path = ANNOTATIONS_DIR / f"{video_name}_phases.csv"
    if not path.exists():
        raise AnnotationsNotFoundError(
            f"No annotation file at {path} -- cannot reconstruct the phase-consistency "
            f"filter for video_name={video_name!r} without it."
        )
    return pd.read_csv(path, header=None, names=["phase", "frame_start", "frame_end"])


def _real_frame_numbers_with_phase(video_name: str) -> List[Tuple[int, str]]:
    """Every real, on-disk frame for this video that also has an annotated
    phase, sorted by frame number -- mirrors
    Embryo_Transition_Dataset._create_sequences()'s own `sorted_frames`
    exactly (same sort key: the numeric suffix of the image filename)."""
    ann = _load_annotations(video_name)
    frame_phase: Dict[int, str] = {}
    for _, row in ann.iterrows():
        for i in range(int(row["frame_start"]), int(row["frame_end"]) + 1):
            frame_phase[i] = row["phase"]

    img_dir = IMAGES_ROOT / video_name
    numbers = []
    if img_dir.is_dir():
        for p in img_dir.glob(f"{video_name}_Image_*.jpeg"):
            m = _IMAGE_FILENAME_PATTERN.search(p.name)
            if m:
                numbers.append(int(m.group(1)))
    numbers = sorted(n for n in numbers if n in frame_phase)
    return [(n, frame_phase[n]) for n in numbers]


def _reconstruct_kept_sequences(video_name: str) -> List[Dict]:
    """Replicates Embryo_Transition_Dataset._create_sequences()'s sliding-
    window + phase-consistency filter (multiple_phases=False, the
    project-wide default) exactly: window_size=8/stride=1, at most 2
    distinct phases per window, and if exactly 2, they must be adjacent in
    CHRONOLOGICAL_PHASES order. The i-th entry of the returned list is
    what `window_start - <this video's first window_start>` (the LOCAL
    offset) indexes into -- see window_to_frame_path()."""
    if video_name in _KEPT_SEQUENCES_CACHE:
        return _KEPT_SEQUENCES_CACHE[video_name]
    frames = _real_frame_numbers_with_phase(video_name)
    kept: List[Dict] = []
    for start in range(0, len(frames) - WINDOW_SIZE + 1, STRIDE):
        window = frames[start:start + WINDOW_SIZE]
        window_phases = [phase for _, phase in window]
        unique_phases = list(set(window_phases))
        if len(unique_phases) > 2:
            continue
        if len(unique_phases) == 2:
            idxs = sorted(_PHASE_TO_INDEX[p] for p in unique_phases)
            if idxs[1] != idxs[0] + 1:
                continue
        kept.append({
            "frame_numbers": [n for n, _ in window],
            "first_frame_phase": window_phases[0],
            "last_frame_phase": window_phases[-1],
        })
    _KEPT_SEQUENCES_CACHE[video_name] = kept
    return kept


def window_frame_numbers(video_name: str, split: str, window_start: int) -> List[int]:
    """All 8 real raw frame numbers belonging to this window, in order.
    Raises FrameMappingError (never returns a guessed/partial list) if the
    window can't be resolved."""
    trajectory_service._check_split(split)
    video_windows = trajectory_service.get_windows(video_name, split)
    if window_start not in video_windows:
        raise FrameMappingError(
            f"window_start={window_start} is not a real window of video_name={video_name!r} "
            f"in split={split!r}."
        )
    local_index = window_start - video_windows[0]
    kept = _reconstruct_kept_sequences(video_name)
    if local_index < 0 or local_index >= len(kept):
        raise FrameMappingError(
            f"Reconstructed {len(kept)} kept sequences for video_name={video_name!r}, but the "
            f"real trajectory has a window at local index {local_index} -- the reconstruction "
            f"does not agree with the real embeddings cache for this video. Refusing to guess."
        )
    return kept[local_index]["frame_numbers"]


def window_to_frame_path(video_name: str, split: str, window_start: int, which: str = "last") -> Path:
    """The single representative real JPEG for this window -- `which="last"`
    (default) matches the causal "current" semantics used everywhere else
    in this Reporting API (`last_frame_phase`/`consistency_flag`, both
    defined off the window's LAST frame); `which="first"` returns the
    window's first frame instead. Verifies the file actually exists on
    disk before returning it -- never fabricates a path for a frame that
    isn't really there."""
    if which not in ("first", "last"):
        raise ValueError(f"which must be 'first' or 'last', got {which!r}.")
    frame_numbers = window_frame_numbers(video_name, split, window_start)
    frame_number = frame_numbers[0] if which == "first" else frame_numbers[-1]
    path = IMAGES_ROOT / video_name / f"{video_name}_Image_{frame_number}.jpeg"
    if not path.exists():
        raise FrameFileNotFoundError(
            f"Reconstructed frame path {path} does not exist on disk for "
            f"video_name={video_name!r}, window_start={window_start}, which={which!r}."
        )
    return path
