"""
Hybrid Clustering Algorithm Implementation

Combines HDBSCAN and hierarchical clustering for optimal results.
Uses intelligent fallback mechanisms and quality-based selection.
"""

import logging
import time
from typing import Dict, Any, Optional, List
import numpy as np

from ..config.settings import HybridConfig, HDBSCANConfig, HierarchicalConfig
from .hdbscan_clusterer import HDBSCANClusterer, ClusteringResult
from .hierarchical_clusterer import HierarchicalClusterer


class HybridClusterer:
    """
    Hybrid clustering that combines HDBSCAN and hierarchical approaches.
    
    Strategy:
    1. Try primary algorithm first
    2. Evaluate result quality
    3. Use fallback if quality is poor
    4. Apply post-processing if enabled
    """
    
    def __init__(
        self, 
        config: HybridConfig,
        hdbscan_config: HDBSCANConfig,
        hierarchical_config: HierarchicalConfig
    ):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize both clusterers
        try:
            self.hdbscan_clusterer = HDBSCANClusterer(hdbscan_config)
            self.hierarchical_clusterer = HierarchicalClusterer(hierarchical_config)
            self.logger.info("Hybrid clusterer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize hybrid clusterer: {e}")
            raise
        
        # Performance tracking
        self.stats = {
            "primary_algorithm_used": 0,
            "fallback_algorithm_used": 0,
            "post_processing_applied": 0,
            "total_clustering_time": 0.0
        }
    
    def fit_predict(self, embeddings: np.ndarray) -> ClusteringResult:
        """
        Perform hybrid clustering on embeddings.
        
        Args:
            embeddings: Array of embeddings with shape (n_samples, n_features)
            
        Returns:
            ClusteringResult object with best clustering result
        """
        if len(embeddings) == 0:
            self.logger.warning("Empty embeddings array provided")
            return ClusteringResult(
                labels=np.array([]),
                algorithm="hybrid",
                processing_time=0.0
            )
        
        self.logger.info(
            f"Starting hybrid clustering on {len(embeddings)} items "
            f"(primary: {self.config.primary_algorithm})"
        )
        start_time = time.time()
        
        try:
            # Step 1: Try primary algorithm
            primary_result = self._run_primary_algorithm(embeddings)
            
            # Step 2: Evaluate quality
            quality_score = self._evaluate_clustering_quality(
                embeddings, primary_result
            )
            
            self.logger.info(
                f"Primary algorithm ({self.config.primary_algorithm}) "
                f"quality score: {quality_score:.3f}"
            )
            
            # Step 3: Use fallback if quality is poor
            final_result = primary_result
            
            if quality_score < self.config.min_cluster_quality_score:
                self.logger.info(
                    f"Quality score {quality_score:.3f} below threshold "
                    f"{self.config.min_cluster_quality_score:.3f}, "
                    f"trying fallback algorithm"
                )
                
                fallback_result = self._run_fallback_algorithm(embeddings)
                fallback_quality = self._evaluate_clustering_quality(
                    embeddings, fallback_result
                )
                
                self.logger.info(
                    f"Fallback algorithm ({self.config.fallback_algorithm}) "
                    f"quality score: {fallback_quality:.3f}"
                )
                
                # Use better result
                if fallback_quality > quality_score:
                    final_result = fallback_result
                    self.stats["fallback_algorithm_used"] += 1
                    self.logger.info("Using fallback algorithm result")
                else:
                    self.stats["primary_algorithm_used"] += 1
                    self.logger.info("Keeping primary algorithm result")
            else:
                self.stats["primary_algorithm_used"] += 1
            
            # Step 4: Apply post-processing if enabled
            if self.config.enable_post_processing:
                final_result = self._post_process_result(
                    embeddings, final_result
                )
                self.stats["post_processing_applied"] += 1
            
            # Update timing and algorithm info
            total_time = time.time() - start_time
            final_result.processing_time = total_time
            final_result.algorithm = "hybrid"
            
            self.stats["total_clustering_time"] += total_time
            
            self.logger.info(
                f"Hybrid clustering completed in {total_time:.2f}s. "
                f"Final result: {final_result.n_clusters} clusters, "
                f"{final_result.n_noise} noise points"
            )
            
            return final_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(
                f"Hybrid clustering failed after {processing_time:.2f}s: {e}"
            )
            
            # Return fallback result
            return ClusteringResult(
                labels=np.zeros(len(embeddings)),  # Single cluster fallback
                algorithm="hybrid",
                processing_time=processing_time,
                n_clusters=1,
                n_noise=0
            )
    
    def _run_primary_algorithm(self, embeddings: np.ndarray) -> ClusteringResult:
        """Run the primary clustering algorithm."""
        if self.config.primary_algorithm == "hdbscan":
            return self.hdbscan_clusterer.fit_predict(embeddings)
        elif self.config.primary_algorithm == "hierarchical":
            return self.hierarchical_clusterer.fit_predict(embeddings)
        else:
            raise ValueError(
                f"Unknown primary algorithm: {self.config.primary_algorithm}"
            )
    
    def _run_fallback_algorithm(self, embeddings: np.ndarray) -> ClusteringResult:
        """Run the fallback clustering algorithm."""
        if self.config.fallback_algorithm == "hdbscan":
            return self.hdbscan_clusterer.fit_predict(embeddings)
        elif self.config.fallback_algorithm == "hierarchical":
            return self.hierarchical_clusterer.fit_predict(embeddings)
        else:
            raise ValueError(
                f"Unknown fallback algorithm: {self.config.fallback_algorithm}"
            )
    
    def _evaluate_clustering_quality(
        self, 
        embeddings: np.ndarray, 
        result: ClusteringResult
    ) -> float:
        """
        Evaluate the quality of a clustering result.
        
        Args:
            embeddings: Original embeddings
            result: Clustering result to evaluate
            
        Returns:
            Quality score between 0 and 1 (higher is better)
        """
        if result.n_clusters == 0:
            return 0.0
        
        if result.n_clusters == 1:
            return 0.1  # Single cluster is usually not ideal
        
        quality_components = []
        
        # Component 1: Silhouette score (if available)
        if result.silhouette_score is not None:
            # Normalize silhouette score from [-1, 1] to [0, 1]
            normalized_silhouette = (result.silhouette_score + 1) / 2
            quality_components.append(normalized_silhouette * 0.4)
        
        # Component 2: Noise ratio (lower is better for most cases)
        noise_penalty = min(1.0, result.noise_ratio * 2)  # Penalize high noise
        quality_components.append((1 - noise_penalty) * 0.3)
        
        # Component 3: Cluster count appropriateness
        cluster_score = self._evaluate_cluster_count(result.n_clusters, len(embeddings))
        quality_components.append(cluster_score * 0.2)
        
        # Component 4: Cluster balance (avoid very unbalanced clusters)
        balance_score = self._evaluate_cluster_balance(result)
        quality_components.append(balance_score * 0.1)
        
        # Calculate final quality score
        final_score = sum(quality_components)
        
        return min(1.0, max(0.0, final_score))
    
    def _evaluate_cluster_count(self, n_clusters: int, n_items: int) -> float:
        """Evaluate appropriateness of cluster count."""
        if n_clusters < self.config.min_clusters:
            return 0.0
        
        if n_clusters > self.config.max_clusters:
            return 0.0
        
        # Ideal range: 2-10% of items as clusters
        ideal_min = max(2, n_items * 0.02)
        ideal_max = min(self.config.max_clusters, n_items * 0.1)
        
        if ideal_min <= n_clusters <= ideal_max:
            return 1.0
        elif n_clusters < ideal_min:
            return n_clusters / ideal_min
        else:
            return ideal_max / n_clusters
    
    def _evaluate_cluster_balance(self, result: ClusteringResult) -> float:
        """Evaluate how balanced the cluster sizes are."""
        if result.n_clusters <= 1:
            return 1.0
        
        # Get cluster sizes (excluding noise)
        cluster_sizes = [
            size for label, size in result.cluster_sizes.items() 
            if label != -1
        ]
        
        if not cluster_sizes:
            return 0.0
        
        # Calculate coefficient of variation
        mean_size = np.mean(cluster_sizes)
        std_size = np.std(cluster_sizes)
        
        if mean_size == 0:
            return 0.0
        
        cv = std_size / mean_size
        
        # Lower CV is better (more balanced)
        # CV of 0 = perfect balance, CV > 2 = very unbalanced
        balance_score = max(0.0, 1.0 - cv / 2.0)
        
        return balance_score
    
    def _post_process_result(
        self, 
        embeddings: np.ndarray, 
        result: ClusteringResult
    ) -> ClusteringResult:
        """
        Apply post-processing to improve clustering result.
        
        Args:
            embeddings: Original embeddings
            result: Initial clustering result
            
        Returns:
            Post-processed clustering result
        """
        self.logger.info("Applying post-processing to clustering result")
        
        # Create a copy of the result
        processed_labels = result.labels.copy()
        
        # Post-processing step 1: Merge very small clusters
        processed_labels = self._merge_small_clusters(
            embeddings, processed_labels
        )
        
        # Post-processing step 2: Reassign outliers if possible
        processed_labels = self._reassign_outliers(
            embeddings, processed_labels
        )
        
        # Create new result with processed labels
        n_clusters = len(set(processed_labels)) - (1 if -1 in processed_labels else 0)
        n_noise = list(processed_labels).count(-1)
        
        processed_result = ClusteringResult(
            labels=processed_labels,
            probabilities=result.probabilities,
            cluster_persistence=result.cluster_persistence,
            outlier_scores=result.outlier_scores,
            algorithm=result.algorithm,
            processing_time=result.processing_time,
            n_clusters=n_clusters,
            n_noise=n_noise,
            silhouette_score=result.silhouette_score,
            calinski_harabasz_score=result.calinski_harabasz_score
        )
        
        self.logger.info(
            f"Post-processing completed. "
            f"Clusters: {result.n_clusters} -> {n_clusters}, "
            f"Noise: {result.n_noise} -> {n_noise}"
        )
        
        return processed_result
    
    def _merge_small_clusters(
        self, 
        embeddings: np.ndarray, 
        labels: np.ndarray
    ) -> np.ndarray:
        """Merge clusters that are too small."""
        min_cluster_size = 2  # Minimum viable cluster size
        
        unique_labels = np.unique(labels)
        cluster_labels = unique_labels[unique_labels != -1]
        
        for cluster_id in cluster_labels:
            cluster_mask = labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size < min_cluster_size:
                # Find nearest cluster to merge with
                cluster_points = embeddings[cluster_mask]
                cluster_centroid = np.mean(cluster_points, axis=0)
                
                best_target = -1  # Default to noise
                best_distance = float('inf')
                
                for other_cluster_id in cluster_labels:
                    if other_cluster_id == cluster_id:
                        continue
                    
                    other_mask = labels == other_cluster_id
                    other_points = embeddings[other_mask]
                    other_centroid = np.mean(other_points, axis=0)
                    
                    distance = np.linalg.norm(cluster_centroid - other_centroid)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_target = other_cluster_id
                
                # Merge with best target
                labels[cluster_mask] = best_target
        
        return labels
    
    def _reassign_outliers(
        self, 
        embeddings: np.ndarray, 
        labels: np.ndarray
    ) -> np.ndarray:
        """Reassign outliers to nearest cluster if similarity is high enough."""
        outlier_mask = labels == -1
        outlier_indices = np.where(outlier_mask)[0]
        
        if len(outlier_indices) == 0:
            return labels
        
        unique_labels = np.unique(labels)
        cluster_labels = unique_labels[unique_labels != -1]
        
        if len(cluster_labels) == 0:
            return labels
        
        # Calculate cluster centroids
        cluster_centroids = {}
        for cluster_id in cluster_labels:
            cluster_mask = labels == cluster_id
            cluster_points = embeddings[cluster_mask]
            cluster_centroids[cluster_id] = np.mean(cluster_points, axis=0)
        
        # Reassign outliers
        for outlier_idx in outlier_indices:
            outlier_point = embeddings[outlier_idx]
            
            best_cluster = -1
            best_similarity = 0.0
            
            for cluster_id, centroid in cluster_centroids.items():
                # Calculate cosine similarity
                similarity = np.dot(outlier_point, centroid) / (
                    np.linalg.norm(outlier_point) * np.linalg.norm(centroid)
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = cluster_id
            
            # Reassign if similarity is above threshold
            if best_similarity > self.config.similarity_threshold:
                labels[outlier_idx] = best_cluster
        
        return labels
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance and usage statistics."""
        total_runs = (
            self.stats["primary_algorithm_used"] + 
            self.stats["fallback_algorithm_used"]
        )
        
        return {
            "algorithm": "hybrid",
            "config": {
                "primary_algorithm": self.config.primary_algorithm,
                "fallback_algorithm": self.config.fallback_algorithm,
                "min_cluster_quality_score": self.config.min_cluster_quality_score,
                "max_clusters": self.config.max_clusters,
                "min_clusters": self.config.min_clusters,
                "similarity_threshold": self.config.similarity_threshold,
                "enable_post_processing": self.config.enable_post_processing
            },
            "usage_stats": {
                "total_runs": total_runs,
                "primary_algorithm_used": self.stats["primary_algorithm_used"],
                "fallback_algorithm_used": self.stats["fallback_algorithm_used"],
                "fallback_usage_rate": (
                    self.stats["fallback_algorithm_used"] / total_runs 
                    if total_runs > 0 else 0
                ),
                "post_processing_applied": self.stats["post_processing_applied"],
                "total_clustering_time": self.stats["total_clustering_time"],
                "average_clustering_time": (
                    self.stats["total_clustering_time"] / total_runs 
                    if total_runs > 0 else 0
                )
            }
        }
    
    def update_config(self, new_config: HybridConfig):
        """Update hybrid configuration."""
        self.config = new_config
        self.logger.info("Hybrid clustering configuration updated")