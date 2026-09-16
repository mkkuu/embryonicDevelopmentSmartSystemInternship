"""
Loads cached embeddings/metadata and exposes them as Trajectory objects
and per-window time lookups, without any caller needing to know
EmbeddingDataset/EmbeddingCache/Embryo_Transition_Dataset internals.

TEST = LOCKED: every public function here takes a `split` argument and
refuses "test" immediately, before touching the filesystem -- verified
this session that Embeddings/resnet18/test/ genuinely exists on disk (it
must, to be refusable), so the guard is a real block, not defending
against a non-existent path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from embeddings.dataset import EmbeddingDataset  # noqa: E402
from evaluation.trajectory import Trajectory, group_into_trajectories  # noqa: E402

EMBEDDINGS_ROOT = Path(__file__).resolve().parent.parent.parent / "Embeddings" / "resnet18"

ALLOWED_SPLITS = ("train", "val")  # "test" is deliberately absent, not filtered elsewhere


class TestSplitLockedError(PermissionError):
    """Raised whenever any caller requests split='test' -- Test = LOCKED
    for the entire SciML extension (docs/SCIENTIFIC_REPORT.md sec 21),
    and the Reporting API is not an exception. This is a hard, tested
    guard, not a convention callers are trusted to honor themselves."""


def _check_split(split: str) -> str:
    split = split.lower()
    if split == "test":
        raise TestSplitLockedError(
            "split='test' was requested. Test is locked for this entire project -- "
            "see docs/SCIENTIFIC_REPORT.md sec 21 and docs/REPRODUCIBILITY.md. "
            "The Reporting API refuses to load Embeddings/resnet18/test/ under any circumstance."
        )
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"split must be one of {ALLOWED_SPLITS!r}, got {split!r}.")
    return split


_TRAJECTORY_CACHE: Dict[str, List[Trajectory]] = {}
_TRAJECTORY_INDEX: Dict[str, Dict[str, int]] = {}  # split -> {video_name: index into the list above}
_TIME_METADATA_CACHE: Dict[str, Optional[pd.DataFrame]] = {}


def _load_trajectories(split: str) -> List[Trajectory]:
    split = _check_split(split)
    if split in _TRAJECTORY_CACHE:
        return _TRAJECTORY_CACHE[split]
    trajectories = group_into_trajectories(EmbeddingDataset(EMBEDDINGS_ROOT, split))
    _TRAJECTORY_CACHE[split] = trajectories
    _TRAJECTORY_INDEX[split] = {t.video_name: i for i, t in enumerate(trajectories)}
    return trajectories


def list_videos(split: str) -> List[str]:
    """All video names available in this split, in cache order (stable,
    not sorted -- matches the order group_into_trajectories itself
    produces, so a caller cross-referencing indices elsewhere gets the
    same order)."""
    trajectories = _load_trajectories(split)
    return [t.video_name for t in trajectories]


def get_trajectory(video_name: str, split: str) -> Trajectory:
    _load_trajectories(split)  # ensures the cache/index for this split exist
    index = _TRAJECTORY_INDEX[split]
    if video_name not in index:
        raise KeyError(f"video_name={video_name!r} not found in split={split!r}.")
    return _TRAJECTORY_CACHE[split][index[video_name]]


def get_windows(video_name: str, split: str) -> List[int]:
    return list(get_trajectory(video_name, split).window_starts)


def get_ground_truth(video_name: str, split: str, window_start: int) -> Tuple[Optional[int], Optional[int]]:
    """Returns (last_frame_phase_index, consistency_flag) for the given
    window, or (None, None) if window_start is not present in this
    trajectory."""
    traj = get_trajectory(video_name, split)
    try:
        t = traj.window_starts.index(window_start)
    except ValueError:
        return None, None
    return int(traj.last_frame_phase[t]), int(traj.consistency_flag[t])


def _load_time_metadata(split: str) -> Optional[pd.DataFrame]:
    """metadata_with_time.csv, built additively by the HMM branch's own
    Phase 1 join (docs/DATA_LINEAGE.md) -- NOT present for every possible
    split/cache in principle, so this returns None (not an error) if the
    file doesn't exist, and every caller must handle that explicitly
    rather than assume time data always exists."""
    split = _check_split(split)
    if split in _TIME_METADATA_CACHE:
        return _TIME_METADATA_CACHE[split]
    path = EMBEDDINGS_ROOT / split / "metadata_with_time.csv"
    df = pd.read_csv(path) if path.exists() else None
    _TIME_METADATA_CACHE[split] = df
    return df


_TIME_ROW_INDEX_CACHE: Dict[str, Dict[Tuple[str, int], dict]] = {}  # split -> (video_name,
# window_start) -> the exact dict get_window_time() returns for that row. Built once per
# split (docs/WEBAPP_TIMELINE_PERFORMANCE.md): a per-video timeline built by calling
# get_window_time() once per window used to re-scan the ENTIRE metadata_with_time.csv
# (a pandas boolean mask over all ~43k rows) on every single call -- O(n_windows x N).
# This index turns every get_window_time() call, for every caller (including
# inference_service.infer()'s own internal use), into an O(1) dict lookup instead,
# built with the exact same per-row logic (same time_data_complete gate, same fields) --
# not a different computation, purely an access-path change. Verified bit-identical to
# the original per-call scan by Tests/reporting/test_trajectory_service.py.


def _default_time_result() -> dict:
    return {
        "window_start_time": None, "window_end_time": None, "window_mid_time": None,
        "window_duration": None, "time_available": False, "time_unit": "unknown/unverified",
        "n_zero_time_diffs_in_window": None,
    }


def _time_row_index(split: str) -> Dict[Tuple[str, int], dict]:
    if split in _TIME_ROW_INDEX_CACHE:
        return _TIME_ROW_INDEX_CACHE[split]
    df = _load_time_metadata(split)
    index: Dict[Tuple[str, int], dict] = {}
    if df is not None:
        for row in df.itertuples(index=False):
            if bool(getattr(row, "time_data_complete", False)):
                index[(row.video_name, row.window_start)] = {
                    "window_start_time": float(row.window_start_time),
                    "window_end_time": float(row.window_end_time),
                    "window_mid_time": float(row.window_mid_time),
                    "window_duration": float(row.window_duration),
                    "time_available": True,
                    "time_unit": "unknown/unverified",
                    "n_zero_time_diffs_in_window": int(row.n_zero_time_diffs_in_window),
                }
            else:
                index[(row.video_name, row.window_start)] = _default_time_result()
    _TIME_ROW_INDEX_CACHE[split] = index
    return index


def get_window_time(video_name: str, split: str, window_start: int) -> dict:
    """Returns a dict with window_start_time/window_end_time/window_mid_time/
    window_duration/time_available/time_unit/n_zero_time_diffs_in_window --
    directly usable to build schemas.WindowInfo. time_unit is ALWAYS
    "unknown/unverified" (copied verbatim from metadata_with_time.manifest.json,
    read directly this session for both train and val -- never converted
    to a named time unit by this function). All four *_time fields are
    None, time_available=False when the row is absent OR when the source
    CSV itself doesn't exist for this split -- never a fabricated 0.0."""
    index = _time_row_index(split)
    return dict(index.get((video_name, window_start), _default_time_result()))
