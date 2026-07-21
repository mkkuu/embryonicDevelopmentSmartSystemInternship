"""
Trajectory-shaped data: the standard unit every Model in this framework
consumes and produces.

A single embedded window (from embeddings.EmbeddingDataset) is not, on
its own, informative about developmental dynamics — the central premise
of this project (see RESEARCH_BLUEPRINT.md) is that windows only mean
something in relation to the other windows of the same video, ordered in
time. This module groups the flat, i.i.d.-looking EmbeddingDataset into
that shape exactly once, so no model or metric downstream has to
re-derive video/ordering structure independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from embeddings.dataset import EmbeddingDataset


@dataclass
class Trajectory:
    """All cached windows belonging to one video, ordered by window_start."""

    video_name: str
    window_starts: List[int]
    embeddings: torch.Tensor  # (T, D)
    consistency_flag: torch.Tensor  # (T,) ground truth, {0, 1}
    first_frame_phase: torch.Tensor  # (T,)
    last_frame_phase: torch.Tensor  # (T,)

    def __len__(self) -> int:
        return len(self.window_starts)


@dataclass
class TrajectoryPrediction:
    """A model's output for one trajectory, aligned window-for-window
    with the Trajectory it was predicted from — checked, not assumed, by
    metrics.compute_metrics()."""

    video_name: str
    window_starts: List[int]
    consistency_flag_prob: torch.Tensor  # (T,) predicted P(transition), in [0, 1]

    def __len__(self) -> int:
        return len(self.window_starts)


def group_into_trajectories(dataset: EmbeddingDataset) -> List[Trajectory]:
    """
    Groups a flat EmbeddingDataset (one item per window) into one
    Trajectory per video_name, sorted by window_start within each video.
    This is the standard entry point every experiment should use to turn
    a cached embedding split into the shape fit()/predict() expect —
    reimplementing this grouping per-experiment is exactly the kind of
    duplication this framework exists to prevent.
    """
    by_video: Dict[str, list] = {}
    for idx in range(len(dataset)):
        embedding, label = dataset[idx]
        video = label["video_name"]
        by_video.setdefault(video, []).append((label["window_start"], embedding, label))

    trajectories: List[Trajectory] = []
    for video, items in by_video.items():
        items.sort(key=lambda x: x[0])
        window_starts = [w for w, _, _ in items]
        embeddings = torch.stack([e for _, e, _ in items], dim=0)
        consistency_flag = torch.stack([lbl["consistency_flag"] for _, _, lbl in items], dim=0)
        first_frame_phase = torch.stack([lbl["first_frame_phase"] for _, _, lbl in items], dim=0)
        last_frame_phase = torch.stack([lbl["last_frame_phase"] for _, _, lbl in items], dim=0)
        trajectories.append(
            Trajectory(
                video_name=video,
                window_starts=window_starts,
                embeddings=embeddings,
                consistency_flag=consistency_flag,
                first_frame_phase=first_frame_phase,
                last_frame_phase=last_frame_phase,
            )
        )
    return trajectories


def shuffle_trajectory(trajectory: Trajectory, rng: np.random.RandomState) -> Trajectory:
    """
    Returns a new Trajectory holding the same set of (embedding,
    consistency_flag, first_frame_phase, last_frame_phase) tuples as the
    input, randomly reordered — window_starts are kept in their original
    sorted positions, so the result represents "the same windows,
    presented in a scrambled temporal sequence," not a different set of
    windows or relabeled positions.

    Used by the frame-shuffle sanity experiment (RESEARCH_BLUEPRINT.md
    Part VIII / the reproducibility PR discipline established in Sprint
    0) to test whether a model's apparent use of temporal structure
    survives destroying that structure. BaseRateModel and
    IdentityDynamicsModel must be EXACTLY unaffected by this, by
    construction (see test_predictions_are_invariant_to_window_order in
    Tests/evaluation/models/test_identity_dynamics.py for the underlying
    guarantee) — if they are not, in a real run, the shuffle
    implementation itself is broken, not the models under test. That is
    a built-in validity check on this function, not just on the
    experiment that calls it.
    """
    n = len(trajectory)
    # Explicit torch.LongTensor, not a bare numpy array — safer across
    # torch versions than relying on numpy-array-into-tensor fancy
    # indexing behaving identically everywhere.
    permutation = torch.from_numpy(rng.permutation(n)).long()
    return Trajectory(
        video_name=trajectory.video_name,
        window_starts=trajectory.window_starts,
        embeddings=trajectory.embeddings[permutation],
        consistency_flag=trajectory.consistency_flag[permutation],
        first_frame_phase=trajectory.first_frame_phase[permutation],
        last_frame_phase=trajectory.last_frame_phase[permutation],
    )
