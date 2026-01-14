"""
Hierarchical Clustering Algorithm Implementation

Provides agglomerative hierarchical clustering for newsletter content.
Serves as a fallback and complementary approach to HDBSCAN.
"""

import logging
import time
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist
except ImportError:
    AgglomerativeClustering = None
    silhouette_score = None
    calinski_harabasz_score = None
    dendrogram = None
    linkage = None
    pdist = None

from ..config.settings import HierarchicalConfig
from .hdbscan_clusterer import ClusteringResult


class HierarchicalClusterer:
    """Hierarchical clustering implementation."""
    
    def __init__(self, config: HierarchicalConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Validate dependencies
        if AgglomerativeClustering is None:
            raise ImportError(
                "scikit-learn is required for hierarchical clustering. "
                "Install with: pip install scikit-learn"
            )
        
        # Initialize clusterer
        self.clusterer = None
        self._initialize_clusterer()
    
    def _initialize_clusterer(self):
        """Initialize the hierarchical clusterer with configuration."""
        try:
            # Use metric instead of deprecated affinity parameter
            self.clusterer = AgglomerativeClustering(
                n_clusters=self.config.n_clusters,
                distance_threshold=self.config.distance_threshold,
                linkage=self.config.linkage,
                metric=self.config.affinity,  # Use metric instead of affinity
                compute_full_tree=self.config.compute_full_tree,
                compute_distances=self.config.compute_distances
            )
            
            self.logger.info("Hierarchical clusterer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize hierarchical clusterer: {e}")
            raise
    
    def fit_predict(self, embeddings: np.ndarray) -> ClusteringResult:
        """
        Perform hierarchical clustering on embeddings.
        
        Args:
            embeddings: Array of embeddings with shape (n_samples, n_features)
            
        Returns:
            ClusteringResult object with clustering information
        """
        if len(embeddings) == 0:
            self.logger.warning("Empty embeddings array provided")
            return ClusteringResult(
                labels=np.array([]),
                algorithm="hierarchical",
                processing_time=0.0
            )
        
        self.logger.info(
            f"Starting hierarchical clustering on {len(embeddings)} items"
        )
        start_time = time.time()
        
        try:
            # Determine optimal number of clusters if not specified
            if self.config.n_clusters is None and self.config.distance_threshold is None:
                n_clusters = self._estimate_optimal_clusters(embeddings)
                self.clusterer.set_params(n_clusters=n_clusters)
                self.logger.info(f"Estimated optimal clusters: {n_clusters}")
            
            # Perform clustering
            labels = self.clusterer.fit_predict(embeddings)
            
            # Calculate metrics
            processing_time = time.time() - start_time
            n_clusters = len(set(labels))
            n_noise = 0  # Hierarchical clustering doesn't produce noise points
            
            # Calculate quality metrics
            silhouette_avg = None
            calinski_harabasz = None
            
            if n_clusters > 1 and silhouette_score is not None:
                try:
                    silhouette_avg = silhouette_score(embeddings, labels)
                except Exception as e:
                    self.logger.warning(f"Failed to calculate silhouette score: {e}")
            
            if n_clusters > 1 and calinski_harabasz_score is not None:
                try:
                    calinski_harabasz = calinski_harabasz_score(embeddings, labels)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to calculate Calinski-Harabasz score: {e}"
                    )
            
            result = ClusteringResult(
                labels=labels,
                probabilities=None,  # Not available for hierarchical clustering
                cluster_persistence=None,
                outlier_scores=None,
                algorithm="hierarchical",
                processing_time=processing_time,
                n_clusters=n_clusters,
                n_noise=n_noise,
                silhouette_score=silhouette_avg,
                calinski_harabasz_score=calinski_harabasz
            )
            
            self.logger.info(
                f"Hierarchical clustering completed in {processing_time:.2f}s. "
                f"Found {n_clusters} clusters"
            )
            
            if silhouette_avg is not None:
                self.logger.info(f"Silhouette score: {silhouette_avg:.3f}")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(
                f"Hierarchical clustering failed after {processing_time:.2f}s: {e}"
            )
            
            # Return single cluster result on failure
            return ClusteringResult(
                labels=np.zeros(len(embeddings)),  # All points in cluster 0
                algorithm="hierarchical",
                processing_time=processing_time,
                n_clusters=1,
                n_noise=0
            )
    
    def _estimate_optimal_clusters(self, embeddings: np.ndarray) -> int:
        """
        Estimate optimal number of clusters using elbow method.
        
        Args:
            embeddings: Array of embeddings
            
        Returns:
            Estimated optimal number of clusters
        """
        if len(embeddings) < 4:
            return 1
        
        max_clusters = min(10, len(embeddings) // 2)
        
        try:
            # Calculate within-cluster sum of squares for different k values
            wcss = []
            k_range = range(1, max_clusters + 1)
            
            for k in k_range:
                if k == 1:
                    wcss.append(np.sum(pdist(embeddings) ** 2) / len(embeddings))
                else:
                    temp_clusterer = AgglomerativeClustering(
                        n_clusters=k,
                        linkage=self.config.linkage,
                        affinity=self.config.affinity
                    )
                    temp_labels = temp_clusterer.fit_predict(embeddings)
                    
                    # Calculate WCSS
                    cluster_wcss = 0
                    for cluster_id in range(k):
                        cluster_points = embeddings[temp_labels == cluster_id]
                        if len(cluster_points) > 0:
                            centroid = np.mean(cluster_points, axis=0)
                            cluster_wcss += np.sum(
                                (cluster_points - centroid) ** 2
                            )
                    wcss.append(cluster_wcss)
            
            # Find elbow point
            if len(wcss) < 3:
                return 2
            
            # Calculate rate of change
            deltas = np.diff(wcss)
            delta_deltas = np.diff(deltas)
            
            # Find the point where the rate of change starts to level off
            elbow_idx = np.argmax(delta_deltas) + 2  # +2 due to double diff
            optimal_k = min(max_clusters, max(2, elbow_idx))
            
            return optimal_k
            
        except Exception as e:
            self.logger.warning(f"Failed to estimate optimal clusters: {e}")
            return min(3, len(embeddings) // 3)  # Conservative fallback
    
    def get_dendrogram_data(self, embeddings: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Generate dendrogram data for visualization.
        
        Args:
            embeddings: Array of embeddings
            
        Returns:
            Dictionary with dendrogram data or None if failed
        """
        if linkage is None or dendrogram is None:
            self.logger.warning("scipy is required for dendrogram generation")
            return None
        
        try:
            # Calculate linkage matrix
            linkage_matrix = linkage(
                embeddings, 
                method=self.config.linkage,
                metric=self.config.affinity
            )
            
            # Generate dendrogram (without plotting)
            dendro_data = dendrogram(
                linkage_matrix,
                no_plot=True,
                truncate_mode='lastp',
                p=min(30, len(embeddings))  # Limit for performance
            )
            
            return {
                "linkage_matrix": linkage_matrix.tolist(),
                "dendrogram": dendro_data,
                "method": self.config.linkage,
                "metric": self.config.affinity
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to generate dendrogram data: {e}")
            return None
    
    def predict_with_threshold(
        self, 
        embeddings: np.ndarray, 
        distance_threshold: float
    ) -> ClusteringResult:
        """
        Perform clustering with a specific distance threshold.
        
        Args:
            embeddings: Array of embeddings
            distance_threshold: Distance threshold for clustering
            
        Returns:
            ClusteringResult object
        """
        # Create temporary clusterer with threshold
        temp_clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage=self.config.linkage,
            affinity=self.config.affinity,
            compute_full_tree=self.config.compute_full_tree,
            compute_distances=self.config.compute_distances
        )
        
        # Store original clusterer
        original_clusterer = self.clusterer
        self.clusterer = temp_clusterer
        
        try:
            result = self.fit_predict(embeddings)
            return result
        finally:
            # Restore original clusterer
            self.clusterer = original_clusterer
    
    def get_cluster_distances(self) -> Optional[np.ndarray]:
        """
        Get distances between clusters if available.
        
        Returns:
            Array of distances or None
        """
        if not hasattr(self.clusterer, 'distances_'):
            return None
        
        return self.clusterer.distances_
    
    def update_config(self, new_config: HierarchicalConfig):
        """Update configuration and reinitialize clusterer."""
        self.config = new_config
        self._initialize_clusterer()
        self.logger.info("Hierarchical clustering configuration updated")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance and configuration statistics."""
        return {
            "algorithm": "hierarchical",
            "config": {
                "n_clusters": self.config.n_clusters,
                "distance_threshold": self.config.distance_threshold,
                "linkage": self.config.linkage,
                "affinity": self.config.affinity,
                "compute_full_tree": self.config.compute_full_tree,
                "compute_distances": self.config.compute_distances
            },
            "is_fitted": hasattr(self.clusterer, 'labels_') if self.clusterer else False
        }
    
    def find_optimal_threshold(
        self, 
        embeddings: np.ndarray, 
        min_clusters: int = 2, 
        max_clusters: int = 10
    ) -> float:
        """
        Find optimal distance threshold for desired cluster range.
        
        Args:
            embeddings: Array of embeddings
            min_clusters: Minimum number of clusters
            max_clusters: Maximum number of clusters
            
        Returns:
            Optimal distance threshold
        """
        if linkage is None:
            self.logger.warning("scipy is required for threshold optimization")
            return 0.5  # Default threshold
        
        try:
            # Calculate linkage matrix
            linkage_matrix = linkage(
                embeddings,
                method=self.config.linkage,
                metric=self.config.affinity
            )
            
            # Extract distances from linkage matrix
            distances = linkage_matrix[:, 2]
            
            # Find thresholds that produce desired cluster counts
            thresholds = []
            
            for target_clusters in range(min_clusters, max_clusters + 1):
                if target_clusters > len(embeddings):
                    break
                
                # The threshold is the distance at which we get target_clusters
                threshold_idx = len(distances) - target_clusters + 1
                if 0 <= threshold_idx < len(distances):
                    thresholds.append(distances[threshold_idx])
            
            if thresholds:
                # Return median threshold
                return float(np.median(thresholds))
            else:
                return float(np.median(distances))
                
        except Exception as e:
            self.logger.warning(f"Failed to find optimal threshold: {e}")
            return 0.5  # Conservative fallback