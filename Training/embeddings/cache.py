"""
On-disk storage format for cached embeddings.

Layout, per (model_name, split):

    Embeddings/{model_name}/manifest.json
    Embeddings/{model_name}/{split}/embeddings.pt   # (N, D) float32 tensor
    Embeddings/{model_name}/{split}/metadata.csv    # N rows, one per embedding

`embeddings.pt` and `metadata.csv` are aligned by an explicit
`embedding_index` column in the metadata, never by implicit row order —
alignment is validated at load time, not assumed.

manifest.json records everything needed to know whether a cache is still
valid for a given (checkpoint, dataset config) combination. It is checked
every time a cache is opened, not trusted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch


class CacheValidationError(RuntimeError):
    """Raised when an on-disk cache is corrupt, incomplete, or does not
    match the configuration requesting it."""


def sha256_of_file(path: Path, chunk_size: int = 2**20) -> str:
    """Content checksum used as checkpoint provenance in the manifest —
    the same discipline used for BASELINE.md's checkpoint checksum."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CacheManifest:
    model_name: str
    checkpoint_path: Optional[str]
    checkpoint_sha256: Optional[str]
    window_size: int
    stride: int
    focal_type: str
    image_size: int
    mode: str  # "image_seq" or "video"
    extractor_version: str
    created_at: str

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))

    @classmethod
    def from_json(cls, path: Path) -> "CacheManifest":
        return cls(**json.loads(path.read_text()))

    def matches(self, other: "CacheManifest", *, check_checkpoint: bool = True) -> List[str]:
        """Return a list of human-readable mismatches between this manifest
        (the cache on disk) and `other` (what the caller actually wants).
        An empty list means they match."""
        mismatches = []
        for field in ("model_name", "window_size", "stride", "focal_type", "image_size", "mode"):
            mine, theirs = getattr(self, field), getattr(other, field)
            if mine != theirs:
                mismatches.append(f"{field}: cache has {mine!r}, requested {theirs!r}")
        if check_checkpoint and self.checkpoint_sha256 != other.checkpoint_sha256:
            mismatches.append(
                f"checkpoint_sha256: cache was built from {self.checkpoint_sha256!r}, "
                f"requested {other.checkpoint_sha256!r}"
            )
        return mismatches


class EmbeddingCache:
    """Read/write access to one (model_name, split) embedding cache directory."""

    EMBEDDINGS_FILENAME = "embeddings.pt"
    METADATA_FILENAME = "metadata.csv"

    def __init__(self, embeddings: torch.Tensor, metadata: pd.DataFrame):
        if len(metadata) != embeddings.shape[0]:
            raise CacheValidationError(
                f"Metadata has {len(metadata)} rows but the embeddings tensor "
                f"has {embeddings.shape[0]} rows — cache is corrupt or was "
                f"written incompletely."
            )
        if "embedding_index" not in metadata.columns:
            raise CacheValidationError(
                "metadata.csv is missing the required 'embedding_index' column."
            )
        if not metadata["embedding_index"].is_unique:
            raise CacheValidationError("metadata.csv has duplicate embedding_index values.")
        if len(metadata) and metadata["embedding_index"].max() >= embeddings.shape[0]:
            raise CacheValidationError(
                "metadata.csv references embedding_index values outside the "
                "range of the embeddings tensor."
            )
        self.embeddings = embeddings
        self.metadata = metadata.set_index("embedding_index", drop=False).sort_index()

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, embedding_index: int):
        return self.embeddings[embedding_index], self.metadata.loc[embedding_index]

    @classmethod
    def save(cls, embeddings: torch.Tensor, metadata: pd.DataFrame, split_dir: Path) -> None:
        split_dir.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, split_dir / cls.EMBEDDINGS_FILENAME)
        metadata.to_csv(split_dir / cls.METADATA_FILENAME, index=False)

    @classmethod
    def load(cls, split_dir: Path, *, lazy: bool = True) -> "EmbeddingCache":
        embeddings_path = split_dir / cls.EMBEDDINGS_FILENAME
        metadata_path = split_dir / cls.METADATA_FILENAME
        if not embeddings_path.exists() or not metadata_path.exists():
            raise CacheValidationError(
                f"No cache found at {split_dir} (expected "
                f"{cls.EMBEDDINGS_FILENAME} and {cls.METADATA_FILENAME}). "
                f"Run build_cache.py for this split first."
            )
        embeddings = cls._load_tensor(embeddings_path, lazy=lazy)
        metadata = pd.read_csv(metadata_path)
        return cls(embeddings, metadata)

    @staticmethod
    def _load_tensor(path: Path, *, lazy: bool) -> torch.Tensor:
        if lazy:
            try:
                return torch.load(path, mmap=True)
            except TypeError:
                # Installed torch predates the `mmap` kwarg (added in
                # torch 2.1). Fall back to a full, eager load — correct,
                # just not lazy. Pin a torch>=2.1 in requirements.txt to
                # get real lazy loading; this dataset is small enough that
                # the fallback is a non-issue in practice either way.
                pass
        return torch.load(path)
