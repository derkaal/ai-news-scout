"""
Embedding Service for Newsletter Clustering

Provides sentence transformer-based embeddings with file-based caching.
Optimized for performance with batch processing and memory management.
"""

import os
import json
import hashlib
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError:
    SentenceTransformer = None
    torch = None

from ..config.settings import EmbeddingConfig, CacheConfig


class EmbeddingCache:
    """File-based caching system for embeddings."""
    
    def __init__(self, config: CacheConfig, cache_dir: str):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.logger = logging.getLogger(__name__)
        
        # Load existing metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache metadata: {e}")
        return {"entries": {}, "total_size_mb": 0, "last_cleanup": None}
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save cache metadata: {e}")
    
    def _generate_cache_key(self, texts: List[str], model_name: str) -> str:
        """Generate a unique cache key for the given texts and model."""
        content = f"{model_name}:{':'.join(sorted(texts))}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{self.config.cache_key_prefix}_{cache_key}.pkl"
    
    def _is_expired(self, timestamp: str) -> bool:
        """Check if a cache entry is expired."""
        try:
            cached_time = datetime.fromisoformat(timestamp)
            expiry_time = cached_time + timedelta(hours=self.config.cache_ttl_hours)
            return datetime.now() > expiry_time
        except Exception:
            return True
    
    def get(self, texts: List[str], model_name: str) -> Optional[np.ndarray]:
        """Retrieve embeddings from cache."""
        if not self.config.cache_type == "file":
            return None
        
        cache_key = self._generate_cache_key(texts, model_name)
        cache_path = self._get_cache_path(cache_key)
        
        # Check if cache entry exists and is not expired
        if cache_key in self.metadata["entries"]:
            entry = self.metadata["entries"][cache_key]
            if self._is_expired(entry["timestamp"]):
                self._remove_entry(cache_key)
                return None
        else:
            return None
        
        # Load embeddings from disk
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    embeddings = pickle.load(f)
                self.logger.debug(f"Cache hit for {len(texts)} texts")
                return embeddings
            except Exception as e:
                self.logger.warning(f"Failed to load cached embeddings: {e}")
                self._remove_entry(cache_key)
        
        return None
    
    def set(self, texts: List[str], model_name: str, embeddings: np.ndarray):
        """Store embeddings in cache."""
        if not self.config.cache_type == "file":
            return
        
        cache_key = self._generate_cache_key(texts, model_name)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            # Save embeddings to disk
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f)
            
            # Calculate file size
            file_size_mb = cache_path.stat().st_size / (1024 * 1024)
            
            # Update metadata
            self.metadata["entries"][cache_key] = {
                "timestamp": datetime.now().isoformat(),
                "size_mb": file_size_mb,
                "num_texts": len(texts),
                "model_name": model_name
            }
            
            self.metadata["total_size_mb"] = sum(
                entry["size_mb"] for entry in self.metadata["entries"].values()
            )
            
            self._save_metadata()
            self.logger.debug(f"Cached embeddings for {len(texts)} texts")
            
            # Check if cleanup is needed
            if self.metadata["total_size_mb"] > self.config.max_cache_size_mb:
                self._cleanup_cache()
                
        except Exception as e:
            self.logger.error(f"Failed to cache embeddings: {e}")
    
    def _remove_entry(self, cache_key: str):
        """Remove a cache entry."""
        cache_path = self._get_cache_path(cache_key)
        
        try:
            if cache_path.exists():
                cache_path.unlink()
            
            if cache_key in self.metadata["entries"]:
                del self.metadata["entries"][cache_key]
                self._save_metadata()
                
        except Exception as e:
            self.logger.error(f"Failed to remove cache entry: {e}")
    
    def _cleanup_cache(self):
        """Remove old cache entries to stay within size limits."""
        self.logger.info("Starting cache cleanup...")
        
        # Sort entries by timestamp (oldest first)
        entries = list(self.metadata["entries"].items())
        entries.sort(key=lambda x: x[1]["timestamp"])
        
        # Remove entries until we're under the size limit
        target_size = self.config.max_cache_size_mb * 0.8  # Leave some headroom
        current_size = self.metadata["total_size_mb"]
        
        removed_count = 0
        for cache_key, entry in entries:
            if current_size <= target_size:
                break
            
            self._remove_entry(cache_key)
            current_size -= entry["size_mb"]
            removed_count += 1
        
        self.metadata["total_size_mb"] = current_size
        self.metadata["last_cleanup"] = datetime.now().isoformat()
        self._save_metadata()
        
        self.logger.info(f"Cache cleanup completed. Removed {removed_count} entries.")


class EmbeddingService:
    """Service for generating and caching text embeddings."""
    
    def __init__(self, config: EmbeddingConfig, cache_config: CacheConfig):
        self.config = config
        self.cache_config = cache_config
        self.logger = logging.getLogger(__name__)
        
        # Initialize cache
        self.cache = EmbeddingCache(cache_config, config.cache_dir)
        
        # Initialize model (lazy loading)
        self._model = None
        self._device = None
        
        # Performance tracking
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_embeddings_generated": 0,
            "total_processing_time": 0.0
        }
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the sentence transformer model."""
        if self._model is None:
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers is required for embedding generation. "
                    "Install with: pip install sentence-transformers"
                )
            
            self.logger.info(f"Loading embedding model: {self.config.model_name}")
            
            # Determine device
            if self.config.device == "auto":
                self._device = "cuda" if torch and torch.cuda.is_available() else "cpu"
            else:
                self._device = self.config.device
            
            self._model = SentenceTransformer(
                self.config.model_name,
                device=self._device
            )
            
            # Set max sequence length
            if hasattr(self._model, 'max_seq_length'):
                self._model.max_seq_length = self.config.max_sequence_length
            
            self.logger.info(f"Model loaded on device: {self._device}")
        
        return self._model
    
    def generate_embeddings(
        self, 
        texts: List[str], 
        use_cache: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            use_cache: Whether to use caching
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not texts:
            raise ValueError("Cannot generate embeddings for empty text list")
        
        start_time = datetime.now()
        
        # Try to get from cache first
        if use_cache and self.config.cache_enabled:
            cached_embeddings = self.cache.get(texts, self.config.model_name)
            if cached_embeddings is not None:
                self.stats["cache_hits"] += 1
                self.logger.debug(f"Retrieved {len(texts)} embeddings from cache")
                return cached_embeddings
            else:
                self.stats["cache_misses"] += 1
        
        # Generate embeddings
        self.logger.info(f"Generating embeddings for {len(texts)} texts")
        
        try:
            # Process in batches to manage memory
            all_embeddings = []
            
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i:i + self.config.batch_size]
                
                batch_embeddings = self.model.encode(
                    batch_texts,
                    batch_size=min(self.config.batch_size, len(batch_texts)),
                    show_progress_bar=show_progress and i == 0,  # Only show for first batch
                    normalize_embeddings=self.config.normalize_embeddings,
                    convert_to_numpy=True
                )
                
                all_embeddings.append(batch_embeddings)
                
                if show_progress and len(texts) > self.config.batch_size:
                    progress = min(100, (i + len(batch_texts)) / len(texts) * 100)
                    self.logger.info(f"Embedding progress: {progress:.1f}%")
            
            # Combine all embeddings
            embeddings = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
            
            # Cache the results
            if use_cache and self.config.cache_enabled:
                self.cache.set(texts, self.config.model_name, embeddings)
            
            # Update stats
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats["total_embeddings_generated"] += len(texts)
            self.stats["total_processing_time"] += processing_time
            
            self.logger.info(
                f"Generated {len(texts)} embeddings in {processing_time:.2f}s "
                f"({len(texts)/processing_time:.1f} texts/sec)"
            )
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def generate_single_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Generate embedding for a single text."""
        embeddings = self.generate_embeddings([text], use_cache=use_cache)
        return embeddings[0] if len(embeddings) > 0 else np.array([])
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        if len(embedding1) == 0 or len(embedding2) == 0:
            return 0.0
        
        # Normalize if not already normalized
        if not self.config.normalize_embeddings:
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        return float(np.dot(embedding1, embedding2))
    
    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise similarity matrix for embeddings."""
        if len(embeddings) == 0:
            return np.array([])
        
        # Normalize if not already normalized
        if not self.config.normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
        
        return np.dot(embeddings, embeddings.T)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_enabled": self.config.cache_enabled,
            "cache_dir": str(self.cache.cache_dir),
            "total_entries": len(self.cache.metadata["entries"]),
            "total_size_mb": self.cache.metadata["total_size_mb"],
            "max_size_mb": self.cache_config.max_cache_size_mb,
            "last_cleanup": self.cache.metadata.get("last_cleanup"),
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "hit_rate": (
                self.stats["cache_hits"] / (self.stats["cache_hits"] + self.stats["cache_misses"])
                if (self.stats["cache_hits"] + self.stats["cache_misses"]) > 0 else 0
            )
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "total_embeddings_generated": self.stats["total_embeddings_generated"],
            "total_processing_time": self.stats["total_processing_time"],
            "average_time_per_embedding": (
                self.stats["total_processing_time"] / self.stats["total_embeddings_generated"]
                if self.stats["total_embeddings_generated"] > 0 else 0
            ),
            "model_name": self.config.model_name,
            "device": self._device,
            "batch_size": self.config.batch_size
        }
    
    def clear_cache(self):
        """Clear all cached embeddings."""
        self.logger.info("Clearing embedding cache...")
        
        # Remove all cache files
        for cache_key in list(self.cache.metadata["entries"].keys()):
            self.cache._remove_entry(cache_key)
        
        # Reset metadata
        self.cache.metadata = {"entries": {}, "total_size_mb": 0, "last_cleanup": None}
        self.cache._save_metadata()
        
        self.logger.info("Cache cleared successfully")