"""
Embedding extraction, caching, and lazy loading for the existing
Training/ModelBuilder.py encoders.

This package is purely additive: it imports from DataSet.py and
ModelBuilder.py but never modifies them. See build_cache.py for the
orchestration entry point.
"""

from .cache import CacheManifest, CacheValidationError, EmbeddingCache
from .dataset import EmbeddingDataset
from .extractor import EmbeddingExtractionError, EmbeddingExtractor

__all__ = [
    "CacheManifest",
    "CacheValidationError",
    "EmbeddingCache",
    "EmbeddingDataset",
    "EmbeddingExtractionError",
    "EmbeddingExtractor",
]
