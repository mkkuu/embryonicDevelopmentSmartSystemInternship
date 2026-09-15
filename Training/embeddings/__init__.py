"""
Embedding extraction, caching, and lazy loading for the existing
Training/ModelBuilder.py encoders.

This package is purely additive: it imports from DataSet.py and
ModelBuilder.py but never modifies them. See build_cache.py for the
orchestration entry point.
"""

from .cache import CacheManifest, CacheValidationError, EmbeddingCache
from .dataset import EmbeddingDataset

# .extractor is deliberately NOT imported here (not even for re-export):
# it imports ModelBuilder, whose module-level ConfigArgs() parses the real
# process sys.argv at import time. Importing this package should never have
# that side effect just because some caller wanted EmbeddingDataset/
# CacheManifest — only build_cache.py actually needs EmbeddingExtractor, and
# it imports `.extractor` directly (with its own sys.argv guard around the
# ModelBuilder import) for exactly that reason. Consumers that need
# EmbeddingExtractor/EmbeddingExtractionError should import them explicitly
# from `embeddings.extractor`.

__all__ = [
    "CacheManifest",
    "CacheValidationError",
    "EmbeddingCache",
    "EmbeddingDataset",
]
