"""
HDBSCAN Clustering Algorithm Implementation

Provides density-based clustering for newsletter content using HDBSCAN.
Optimized for performance with graceful fallback mechanisms.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import hdbscan
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
except ImportError:
    hdbscan = None
    silhouette_score = None
    calinski_harabasz_score = None

from ..config.settings import HDBSCANConfig


class ClusteringResult:
    """Container for clustering results with metadata."""
    
    def __init__(
        self,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        cluster_persistence: Optional[np.ndarray] = None,
        outlier_scores: Optional[np.ndarray] = None,
        algorithm: str = "hdbscan",
        processing_time: float = 0.0,
        n_clusters: int = 0,
        n_noise: int = 0,
        silhouette_score: Optional[float] = None,
        calinski_harabasz_score: Optional[float] = None
    ):
        self.labels = labels
        self.probabilities = probabilities
        self.cluster_persistence = cluster_persistence
        self.outlier_scores = outlier_scores
        self.algorithm = algorithm
        self.processing_time = processing_time
        self.n_clusters = n_clusters
        self.n_noise = n_noise
        self.silhouette_score = silhouette_score
        self.calinski_harabasz_score = calinski_harabasz_score
        
        # Calculate additional metrics
        self.noise_ratio = n_noise / len(labels) if len(labels) > 0 else 0.0
        self.largest_cluster_size = self._calculate_largest_cluster_size()
        self.cluster_sizes = self._calculate_cluster_sizes()
    
    def _calculate_largest_cluster_size(self) -> int:
        """Calculate the size of the largest cluster."""
        if len(self.labels) == 0:
            return 0
        
        unique_labels = np.unique(self.labels)
        cluster_labels = unique_labels[unique_labels != -1]  # Exclude noise
        
        if len(cluster_labels) == 0:
            return 0
        
        cluster_sizes = [np.sum(self.labels == label) for label in cluster_labels]
        return max(cluster_sizes) if cluster_sizes else 0
    
    def _calculate_cluster_sizes(self) -> Dict[int, int]:
        """Calculate sizes for all clusters."""
        if len(self.labels) == 0:
            return {}
        
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique_labels.tolist(), counts.tolist()))
    
    def get_cluster_items(self, cluster_id: int) -> np.ndarray:
        """Get indices of items in a specific cluster."""
        return np.where(self.labels == cluster_id)[0]
    
    def get_noise_items(self) -> np.ndarray:
        """Get indices of items classified as noise."""
        return np.where(self.labels == -1)[0]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "algorithm": self.algorithm,
            "processing_time": self.processing_time,
            "n_clusters": self.n_clusters,
            "n_noise": self.n_noise,
            "noise_ratio": self.noise_ratio,
            "largest_cluster_size": self.largest_cluster_size,
            "cluster_sizes": self.cluster_sizes,
            "silhouette_score": self.silhouette_score,
            "calinski_harabasz_score": self.calinski_harabasz_score,
            "labels": self.labels.tolist() if self.labels is not None else None,
            "probabilities": (
                self.probabilities.tolist() 
                if self.probabilities is not None else None
            )
        }


class HDBSCANClusterer:
    """HDBSCAN-based clustering implementation."""
    
    def __init__(self, config: HDBSCANConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Validate dependencies
        if hdbscan is None:
            raise ImportError(
                "hdbscan is required for HDBSCAN clustering. "
                "Install with: pip install hdbscan"
            )
        
        # Initialize clusterer
        self.clusterer = None
        self._initialize_clusterer()
    
    def _initialize_clusterer(self):
        """Initialize the HDBSCAN clusterer with configuration."""
        try:
            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.config.min_cluster_size,
                min_samples=self.config.min_samples,
                cluster_selection_epsilon=self.config.cluster_selection_epsilon,
                max_cluster_size=self.config.max_cluster_size,
                metric=self.config.metric,
                cluster_selection_method=self.config.cluster_selection_method,
                allow_single_cluster=self.config.allow_single_cluster,
                prediction_data=self.config.prediction_data
            )
            
            self.logger.info("HDBSCAN clusterer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize HDBSCAN clusterer: {e}")
            raise
    
    def fit_predict(self, embeddings: np.ndarray) -> ClusteringResult:
        """
        Perform clustering on embeddings.
        
        Args:
            embeddings: Array of embeddings with shape (n_samples, n_features)
            
        Returns:
            ClusteringResult object with clustering information
        """
        if len(embeddings) == 0:
            self.logger.warning("Empty embeddings array provided")
            return ClusteringResult(
                labels=np.array([]),
                algorithm="hdbscan",
                processing_time=0.0
            )
        
        self.logger.info(f"Starting HDBSCAN clustering on {len(embeddings)} items")
        start_time = time.time()
        
        try:
            # Perform clustering
            labels = self.clusterer.fit_predict(embeddings)
            
            # Get additional information
            probabilities = getattr(self.clusterer, 'probabilities_', None)
            cluster_persistence = getattr(self.clusterer, 'cluster_persistence_', None)
            outlier_scores = getattr(self.clusterer, 'outlier_scores_', None)
            
            # Calculate metrics
            processing_time = time.time() - start_time
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            # Calculate quality metrics
            silhouette_avg = None
            calinski_harabasz = None
            
            if n_clusters > 1 and silhouette_score is not None:
                try:
                    # Only calculate silhouette score for non-noise points
                    non_noise_mask = labels != -1
                    if np.sum(non_noise_mask) > 1:
                        silhouette_avg = silhouette_score(
                            embeddings[non_noise_mask], 
                            labels[non_noise_mask]
                        )
                except Exception as e:
                    self.logger.warning(f"Failed to calculate silhouette score: {e}")
            
            if n_clusters > 1 and calinski_harabasz_score is not None:
                try:
                    non_noise_mask = labels != -1
                    if np.sum(non_noise_mask) > 1:
                        calinski_harabasz = calinski_harabasz_score(
                            embeddings[non_noise_mask], 
                            labels[non_noise_mask]
                        )
                except Exception as e:
                    self.logger.warning(f"Failed to calculate Calinski-Harabasz score: {e}")
            
            result = ClusteringResult(
                labels=labels,
                probabilities=probabilities,
                cluster_persistence=cluster_persistence,
                outlier_scores=outlier_scores,
                algorithm="hdbscan",
                processing_time=processing_time,
                n_clusters=n_clusters,
                n_noise=n_noise,
                silhouette_score=silhouette_avg,
                calinski_harabasz_score=calinski_harabasz
            )
            
            self.logger.info(
                f"HDBSCAN clustering completed in {processing_time:.2f}s. "
                f"Found {n_clusters} clusters, {n_noise} noise points "
                f"({result.noise_ratio:.1%} noise ratio)"
            )
            
            if silhouette_avg is not None:
                self.logger.info(f"Silhouette score: {silhouette_avg:.3f}")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"HDBSCAN clustering failed after {processing_time:.2f}s: {e}")
            
            # Return empty result on failure
            return ClusteringResult(
                labels=np.full(len(embeddings), -1),  # All points as noise
                algorithm="hdbscan",
                processing_time=processing_time,
                n_clusters=0,
                n_noise=len(embeddings)
            )
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data points.
        
        Args:
            embeddings: Array of embeddings for new points
            
        Returns:
            Array of predicted cluster labels
        """
        if not hasattr(self.clusterer, 'prediction_data_') or self.clusterer.prediction_data_ is None:
            self.logger.warning("No prediction data available. Clusterer must be fitted with prediction_data=True")
            return np.full(len(embeddings), -1)
        
        try:
            labels, _ = hdbscan.approximate_predict(self.clusterer, embeddings)
            return labels
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return np.full(len(embeddings), -1)
    
    def get_cluster_exemplars(self, embeddings: np.ndarray, result: ClusteringResult, n_exemplars: int = 3) -> Dict[int, List[int]]:
        """
        Get exemplar points for each cluster.
        
        Args:
            embeddings: Original embeddings
            result: Clustering result
            n_exemplars: Number of exemplars per cluster
            
        Returns:
            Dictionary mapping cluster_id to list of exemplar indices
        """
        exemplars = {}
        
        for cluster_id in range(result.n_clusters):
            cluster_mask = result.labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            cluster_embeddings = embeddings[cluster_mask]
            
            # Calculate centroid
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Find points closest to centroid
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            closest_indices = np.argsort(distances)[:n_exemplars]
            
            # Map back to original indices
            exemplars[cluster_id] = cluster_indices[closest_indices].tolist()
        
        return exemplars
    
    def update_config(self, new_config: HDBSCANConfig):
        """Update configuration and reinitialize clusterer."""
        self.config = new_config
        self._initialize_clusterer()
        self.logger.info("HDBSCAN configuration updated")
    
    def get_cluster_hierarchy(self) -> Optional[Dict[str, Any]]:
        """
        Get cluster hierarchy information if available.
        
        Returns:
            Dictionary with hierarchy information or None
        """
        if not hasattr(self.clusterer, 'condensed_tree_'):
            return None
        
        try:
            condensed_tree = self.clusterer.condensed_tree_
            
            return {
                "condensed_tree": condensed_tree.to_pandas() if hasattr(condensed_tree, 'to_pandas') else None,
                "cluster_persistence": getattr(self.clusterer, 'cluster_persistence_', None),
                "cluster_hierarchy": getattr(self.clusterer, 'cluster_hierarchy_', None)
            }
        except Exception as e:
            self.logger.warning(f"Failed to extract hierarchy information: {e}")
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance and configuration statistics."""
        return {
            "algorithm": "hdbscan",
            "config": {
                "min_cluster_size": self.config.min_cluster_size,
                "min_samples": self.config.min_samples,
                "cluster_selection_epsilon": self.config.cluster_selection_epsilon,
                "max_cluster_size": self.config.max_cluster_size,
                "metric": self.config.metric,
                "cluster_selection_method": self.config.cluster_selection_method,
                "allow_single_cluster": self.config.allow_single_cluster,
                "prediction_data": self.config.prediction_data
            },
            "is_fitted": hasattr(self.clusterer, 'labels_') if self.clusterer else False
        }