"""
torch.utils.data.Dataset wrapper around a cached EmbeddingCache, so cached
embeddings are consumable through the exact same DataLoader machinery the
rest of this codebase already uses for Embryo_Transition_Dataset — no new
consumption pattern for downstream experiments (E1 and beyond) to learn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from .cache import CacheManifest, CacheValidationError, EmbeddingCache


class EmbeddingDataset(Dataset):
    """
    Parameters
    ----------
    cache_root : Path
        Directory containing manifest.json and one subdirectory per split
        (as written by build_cache.py) — i.e. Embeddings/{model_name}/.
    split : str
        One of "train", "val", "test" — must match a subdirectory under
        cache_root.
    expected_manifest : CacheManifest, optional
        If given, the on-disk manifest is checked against it and a
        CacheValidationError is raised on any mismatch, rather than
        silently serving a stale or wrongly-configured cache. Pass this
        whenever the calling experiment has its own opinion about which
        window_size/stride/checkpoint it needs.
    lazy : bool
        Passed through to EmbeddingCache.load(); True by default.

    __getitem__ returns
    --------------------
    (embedding, label) where label is a dict with keys:
      'consistency_flag', 'first_frame_phase', 'last_frame_phase'
        — the same supervision Embryo_Transition_Dataset.__getitem__
          already produces, unchanged.
      'video_name', 'window_start'
        — not exposed by the raw dataset at all; needed by any experiment
          that has to group windows back into per-video trajectories
          rather than treat them as i.i.d. samples.
    """

    def __init__(
        self,
        cache_root: Path,
        split: str,
        expected_manifest: Optional[CacheManifest] = None,
        lazy: bool = True,
    ):
        cache_root = Path(cache_root)
        manifest_path = cache_root / "manifest.json"
        if not manifest_path.exists():
            raise CacheValidationError(f"No manifest.json found at {cache_root}.")
        self.manifest = CacheManifest.from_json(manifest_path)

        if expected_manifest is not None:
            mismatches = self.manifest.matches(expected_manifest)
            if mismatches:
                raise CacheValidationError(
                    f"Cache at {cache_root} does not match the requested "
                    f"configuration:\n  - " + "\n  - ".join(mismatches) +
                    f"\nRegenerate the cache with build_cache.py, or point "
                    f"at the correct cache_root, rather than proceeding "
                    f"with a mismatched cache."
                )

        self.split = split
        self._cache = EmbeddingCache.load(cache_root / split, lazy=lazy)

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int):
        embedding, row = self._cache[idx]
        label = {
            "consistency_flag": torch.tensor(int(row["consistency_flag"]), dtype=torch.long),
            "first_frame_phase": torch.tensor(int(row["first_frame_phase"]), dtype=torch.long),
            "last_frame_phase": torch.tensor(int(row["last_frame_phase"]), dtype=torch.long),
            "video_name": row["video_name"],
            "window_start": int(row["window_start"]),
        }
        return embedding, label
