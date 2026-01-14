"""
Clustering Configuration Management

Centralized configuration for the newsletter clustering engine.
Supports environment-based configuration with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str = "all-MiniLM-L6-v2"  # Lightweight, fast sentence transformer
    cache_dir: str = field(default_factory=lambda: str(Path.home() / ".newsletter_agent" / "embeddings"))
    cache_enabled: bool = True
    batch_size: int = 32
    max_sequence_length: int = 512
    normalize_embeddings: bool = True
    device: str = "cpu"  # Will auto-detect GPU if available


@dataclass
class HDBSCANConfig:
    """Configuration for HDBSCAN clustering algorithm."""
    min_cluster_size: int = 3
    min_samples: int = 2
    cluster_selection_epsilon: float = 0.0
    max_cluster_size: Optional[int] = None
    metric: str = "euclidean"
    cluster_selection_method: str = "eom"  # excess of mass
    allow_single_cluster: bool = False
    prediction_data: bool = True


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical clustering algorithm."""
    linkage: str = "ward"  # ward, complete, average, single
    distance_threshold: Optional[float] = None
    n_clusters: Optional[int] = None
    affinity: str = "euclidean"
    compute_full_tree: str = "auto"
    compute_distances: bool = False


@dataclass
class HybridConfig:
    """Configuration for hybrid clustering approach."""
    primary_algorithm: str = "hdbscan"  # hdbscan or hierarchical
    fallback_algorithm: str = "hierarchical"
    min_cluster_quality_score: float = 0.3
    max_clusters: int = 50
    min_clusters: int = 2
    similarity_threshold: float = 0.7
    enable_post_processing: bool = True


@dataclass
class ValidationConfig:
    """Configuration for clustering validation."""
    min_silhouette_score: float = 0.2
    max_noise_ratio: float = 0.3
    min_cluster_coherence: float = 0.4
    enable_diversity_check: bool = True
    min_source_diversity: float = 0.2
    quality_metrics: List[str] = field(default_factory=lambda: [
        "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"
    ])


@dataclass
class PerformanceConfig:
    """Configuration for performance constraints."""
    max_processing_time_seconds: int = 30
    max_memory_usage_gb: float = 2.0
    enable_parallel_processing: bool = True
    max_workers: Optional[int] = None  # Auto-detect based on CPU cores
    chunk_size: int = 100
    enable_progress_tracking: bool = True


@dataclass
class CacheConfig:
    """Configuration for caching system."""
    cache_type: str = "file"  # file, redis (future)
    cache_ttl_hours: int = 24
    max_cache_size_mb: int = 500
    enable_compression: bool = True
    cache_key_prefix: str = "newsletter_clustering"


@dataclass
class ClusteringConfig:
    """Main configuration class for the clustering engine."""
    
    # Sub-configurations
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    hdbscan: HDBSCANConfig = field(default_factory=HDBSCANConfig)
    hierarchical: HierarchicalConfig = field(default_factory=HierarchicalConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    
    # General settings
    default_algorithm: str = "hybrid"
    enable_logging: bool = True
    log_level: str = "INFO"
    random_seed: int = 42
    
    # Integration settings
    google_sheets_integration: bool = True
    add_clustering_metadata: bool = True
    preserve_original_data: bool = True
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        self._validate_config()
        self._setup_directories()
        self._apply_environment_overrides()
    
    def _validate_config(self):
        """Validate configuration values."""
        if self.default_algorithm not in ["hdbscan", "hierarchical", "hybrid"]:
            raise ValueError(f"Invalid default_algorithm: {self.default_algorithm}")
        
        if self.hybrid.primary_algorithm not in ["hdbscan", "hierarchical"]:
            raise ValueError(f"Invalid primary_algorithm: {self.hybrid.primary_algorithm}")
        
        if self.performance.max_processing_time_seconds <= 0:
            raise ValueError("max_processing_time_seconds must be positive")
        
        if self.performance.max_memory_usage_gb <= 0:
            raise ValueError("max_memory_usage_gb must be positive")
    
    def _setup_directories(self):
        """Create necessary directories."""
        cache_dir = Path(self.embedding.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides."""
        # Embedding overrides
        if os.getenv("CLUSTERING_EMBEDDING_MODEL"):
            self.embedding.model_name = os.getenv("CLUSTERING_EMBEDDING_MODEL")
        
        if os.getenv("CLUSTERING_CACHE_DIR"):
            self.embedding.cache_dir = os.getenv("CLUSTERING_CACHE_DIR")
        
        # Performance overrides
        if os.getenv("CLUSTERING_MAX_TIME"):
            self.performance.max_processing_time_seconds = int(os.getenv("CLUSTERING_MAX_TIME"))
        
        if os.getenv("CLUSTERING_MAX_MEMORY"):
            self.performance.max_memory_usage_gb = float(os.getenv("CLUSTERING_MAX_MEMORY"))
        
        # Algorithm overrides
        if os.getenv("CLUSTERING_DEFAULT_ALGORITHM"):
            self.default_algorithm = os.getenv("CLUSTERING_DEFAULT_ALGORITHM")
        
        # Logging overrides
        if os.getenv("CLUSTERING_LOG_LEVEL"):
            self.log_level = os.getenv("CLUSTERING_LOG_LEVEL")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "embedding": self.embedding.__dict__,
            "hdbscan": self.hdbscan.__dict__,
            "hierarchical": self.hierarchical.__dict__,
            "hybrid": self.hybrid.__dict__,
            "validation": self.validation.__dict__,
            "performance": self.performance.__dict__,
            "cache": self.cache.__dict__,
            "default_algorithm": self.default_algorithm,
            "enable_logging": self.enable_logging,
            "log_level": self.log_level,
            "random_seed": self.random_seed,
            "google_sheets_integration": self.google_sheets_integration,
            "add_clustering_metadata": self.add_clustering_metadata,
            "preserve_original_data": self.preserve_original_data
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ClusteringConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        # Update sub-configurations
        if "embedding" in config_dict:
            for key, value in config_dict["embedding"].items():
                setattr(config.embedding, key, value)
        
        if "hdbscan" in config_dict:
            for key, value in config_dict["hdbscan"].items():
                setattr(config.hdbscan, key, value)
        
        if "hierarchical" in config_dict:
            for key, value in config_dict["hierarchical"].items():
                setattr(config.hierarchical, key, value)
        
        if "hybrid" in config_dict:
            for key, value in config_dict["hybrid"].items():
                setattr(config.hybrid, key, value)
        
        if "validation" in config_dict:
            for key, value in config_dict["validation"].items():
                setattr(config.validation, key, value)
        
        if "performance" in config_dict:
            for key, value in config_dict["performance"].items():
                setattr(config.performance, key, value)
        
        if "cache" in config_dict:
            for key, value in config_dict["cache"].items():
                setattr(config.cache, key, value)
        
        # Update main configuration
        main_keys = [
            "default_algorithm", "enable_logging", "log_level", "random_seed",
            "google_sheets_integration", "add_clustering_metadata", "preserve_original_data"
        ]
        
        for key in main_keys:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        return config


def get_config_from_env() -> ClusteringConfig:
    """Get configuration with environment variable overrides applied."""
    return ClusteringConfig()


def create_development_config() -> ClusteringConfig:
    """Create a configuration optimized for development."""
    config = ClusteringConfig()
    config.enable_logging = True
    config.log_level = "DEBUG"
    config.performance.enable_progress_tracking = True
    config.cache.cache_ttl_hours = 1  # Shorter cache for development
    return config


def create_production_config() -> ClusteringConfig:
    """Create a configuration optimized for production."""
    config = ClusteringConfig()
    config.enable_logging = True
    config.log_level = "INFO"
    config.performance.enable_parallel_processing = True
    config.cache.cache_ttl_hours = 24
    config.cache.max_cache_size_mb = 1000  # Larger cache for production
    return config