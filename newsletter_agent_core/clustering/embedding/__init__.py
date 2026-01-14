"""
Embedding package for newsletter clustering engine.

Provides sentence transformer-based embeddings with caching capabilities.
"""

from .service import EmbeddingService, EmbeddingCache

__all__ = [
    "EmbeddingService",
    "EmbeddingCache"
]