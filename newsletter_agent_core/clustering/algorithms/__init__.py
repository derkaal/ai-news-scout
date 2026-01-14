"""
Clustering algorithms package for newsletter clustering engine.

Provides multiple clustering algorithms with consistent interfaces.
"""

from .hdbscan_clusterer import HDBSCANClusterer, ClusteringResult
from .hierarchical_clusterer import HierarchicalClusterer
from .hybrid_clusterer import HybridClusterer

__all__ = [
    "HDBSCANClusterer",
    "HierarchicalClusterer",
    "HybridClusterer",
    "ClusteringResult"
]