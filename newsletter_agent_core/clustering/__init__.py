"""
Newsletter Agent Clustering Engine

A modular clustering system for newsletter content analysis and grouping.
Provides semantic clustering capabilities with multiple algorithms and caching.

Key Components:
- Orchestrator: Main coordination service
- Embedding Service: Sentence transformer-based embeddings with caching
- Clustering Algorithms: HDBSCAN, Hierarchical, and Hybrid approaches
- Validation: Result validation and quality metrics
- Configuration: Centralized settings management

Performance Targets:
- <30 seconds for 400 items
- <2GB memory usage
- File-based caching with Redis support planned

Usage:
    from newsletter_agent_core.clustering import ClusteringOrchestrator
    
    orchestrator = ClusteringOrchestrator()
    results = orchestrator.cluster_items(newsletter_items)
"""

from .orchestrator import ClusteringOrchestrator
from .config.settings import ClusteringConfig
from .embedding.service import EmbeddingService
from .algorithms.hdbscan_clusterer import HDBSCANClusterer
from .algorithms.hierarchical_clusterer import HierarchicalClusterer
from .algorithms.hybrid_clusterer import HybridClusterer
from .validation.validator import ClusteringValidator

__version__ = "1.0.0"
__author__ = "Newsletter Agent Core Team"

__all__ = [
    "ClusteringOrchestrator",
    "ClusteringConfig",
    "EmbeddingService",
    "HDBSCANClusterer",
    "HierarchicalClusterer",
    "HybridClusterer",
    "ClusteringValidator"
]

# Module-level configuration
DEFAULT_CONFIG = ClusteringConfig()


def get_default_orchestrator():
    """Get a default configured clustering orchestrator."""
    return ClusteringOrchestrator(config=DEFAULT_CONFIG)