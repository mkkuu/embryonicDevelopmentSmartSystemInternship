"""
ExperimentConfig — everything needed to know exactly what an experiment
run was, saved alongside every result as JSON. The same provenance
discipline already established for embedding caches
(embeddings.cache.CacheManifest) and for the baseline reproduction
protocol (BASELINE.md), applied to full experiment runs.

Deliberately does not duplicate every embeddings.CacheManifest field
(window_size, stride, focal_type, ...) — that would create two sources of
truth for the same configuration. This config only names *which* cache to
use (embedding_model_name); the cache's own manifest remains authoritative
for how it was built.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ExperimentConfig:
    experiment_name: str
    model_kwargs: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # model_name -> its own kwargs
    cache_root: str = "../Embeddings"
    embedding_model_name: str = "resnet18"
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    output_dir: str = "../Results/evaluation"
    confidence_level: float = 0.95

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "ExperimentConfig":
        return cls(**json.loads(path.read_text()))
