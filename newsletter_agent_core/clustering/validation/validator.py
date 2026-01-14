"""
Clustering Validation System

Provides comprehensive validation and quality assessment for clustering results.
Includes diversity checks, quality metrics, and performance validation.
"""

import logging
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from collections import Counter

try:
    from sklearn.metrics import (
        silhouette_score, 
        calinski_harabasz_score, 
        davies_bouldin_score,
        adjusted_rand_score,
        normalized_mutual_info_score
    )
except ImportError:
    silhouette_score = None
    calinski_harabasz_score = None
    davies_bouldin_score = None
    adjusted_rand_score = None
    normalized_mutual_info_score = None

from ..config.settings import ValidationConfig
from ..algorithms.hdbscan_clusterer import ClusteringResult


class ValidationResult:
    """Container for validation results."""
    
    def __init__(
        self,
        is_valid: bool,
        quality_score: float,
        validation_metrics: Dict[str, Any],
        diversity_metrics: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        issues: List[str],
        recommendations: List[str]
    ):
        self.is_valid = is_valid
        self.quality_score = quality_score
        self.validation_metrics = validation_metrics
        self.diversity_metrics = diversity_metrics
        self.performance_metrics = performance_metrics
        self.issues = issues
        self.recommendations = recommendations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary."""
        return {
            "is_valid": self.is_valid,
            "quality_score": self.quality_score,
            "validation_metrics": self.validation_metrics,
            "diversity_metrics": self.diversity_metrics,
            "performance_metrics": self.performance_metrics,
            "issues": self.issues,
            "recommendations": self.recommendations
        }


class ClusteringValidator:
    """Comprehensive clustering validation system."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Validate dependencies
        if silhouette_score is None:
            self.logger.warning(
                "scikit-learn is required for full validation metrics. "
                "Some metrics will be unavailable."
            )
    
    def validate_clustering(
        self,
        embeddings: np.ndarray,
        clustering_result: ClusteringResult,
        original_items: Optional[List[Dict[str, Any]]] = None
    ) -> ValidationResult:
        """
        Perform comprehensive validation of clustering results.
        
        Args:
            embeddings: Original embeddings used for clustering
            clustering_result: Result from clustering algorithm
            original_items: Original newsletter items (for diversity analysis)
            
        Returns:
            ValidationResult with comprehensive assessment
        """
        self.logger.info("Starting clustering validation")
        start_time = time.time()
        
        issues = []
        recommendations = []
        
        # Validate basic requirements
        basic_validation = self._validate_basic_requirements(
            embeddings, clustering_result
        )
        if not basic_validation["is_valid"]:
            issues.extend(basic_validation["issues"])
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            embeddings, clustering_result
        )
        
        # Calculate diversity metrics
        diversity_metrics = self._calculate_diversity_metrics(
            clustering_result, original_items
        )
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            clustering_result, len(embeddings)
        )
        
        # Validate against thresholds
        threshold_validation = self._validate_against_thresholds(
            quality_metrics, diversity_metrics
        )
        issues.extend(threshold_validation["issues"])
        recommendations.extend(threshold_validation["recommendations"])
        
        # Calculate overall quality score
        quality_score = self._calculate_overall_quality_score(
            quality_metrics, diversity_metrics, performance_metrics
        )
        
        # Determine if clustering is valid
        is_valid = (
            basic_validation["is_valid"] and 
            len(threshold_validation["issues"]) == 0 and
            quality_score >= 0.3  # Minimum acceptable quality
        )
        
        validation_time = time.time() - start_time
        performance_metrics["validation_time"] = validation_time
        
        result = ValidationResult(
            is_valid=is_valid,
            quality_score=quality_score,
            validation_metrics=quality_metrics,
            diversity_metrics=diversity_metrics,
            performance_metrics=performance_metrics,
            issues=issues,
            recommendations=recommendations
        )
        
        self.logger.info(
            f"Validation completed in {validation_time:.2f}s. "
            f"Valid: {is_valid}, Quality: {quality_score:.3f}"
        )
        
        return result
    
    def _validate_basic_requirements(
        self,
        embeddings: np.ndarray,
        clustering_result: ClusteringResult
    ) -> Dict[str, Any]:
        """Validate basic clustering requirements."""
        issues = []
        
        # Check if embeddings and labels have same length
        if len(embeddings) != len(clustering_result.labels):
            issues.append(
                f"Embeddings length ({len(embeddings)}) != "
                f"labels length ({len(clustering_result.labels)})"
            )
        
        # Check for empty result
        if len(clustering_result.labels) == 0:
            issues.append("Empty clustering result")
        
        # Check for reasonable number of clusters
        if clustering_result.n_clusters == 0:
            issues.append("No clusters found")
        elif clustering_result.n_clusters == len(clustering_result.labels):
            issues.append("Each item in its own cluster (over-clustering)")
        
        # Check noise ratio
        if clustering_result.noise_ratio > self.config.max_noise_ratio:
            issues.append(
                f"Noise ratio ({clustering_result.noise_ratio:.1%}) exceeds "
                f"maximum ({self.config.max_noise_ratio:.1%})"
            )
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues
        }
    
    def _calculate_quality_metrics(
        self,
        embeddings: np.ndarray,
        clustering_result: ClusteringResult
    ) -> Dict[str, Any]:
        """Calculate clustering quality metrics."""
        metrics = {}
        
        # Use existing metrics from clustering result if available
        if clustering_result.silhouette_score is not None:
            metrics["silhouette_score"] = clustering_result.silhouette_score
        elif (silhouette_score is not None and 
              clustering_result.n_clusters > 1 and 
              clustering_result.n_noise < len(clustering_result.labels)):
            try:
                # Calculate for non-noise points only
                non_noise_mask = clustering_result.labels != -1
                if np.sum(non_noise_mask) > 1:
                    metrics["silhouette_score"] = silhouette_score(
                        embeddings[non_noise_mask],
                        clustering_result.labels[non_noise_mask]
                    )
            except Exception as e:
                self.logger.warning(f"Failed to calculate silhouette score: {e}")
                metrics["silhouette_score"] = None
        
        # Calinski-Harabasz score
        if clustering_result.calinski_harabasz_score is not None:
            metrics["calinski_harabasz_score"] = clustering_result.calinski_harabasz_score
        elif (calinski_harabasz_score is not None and 
              clustering_result.n_clusters > 1):
            try:
                non_noise_mask = clustering_result.labels != -1
                if np.sum(non_noise_mask) > 1:
                    metrics["calinski_harabasz_score"] = calinski_harabasz_score(
                        embeddings[non_noise_mask],
                        clustering_result.labels[non_noise_mask]
                    )
            except Exception as e:
                self.logger.warning(f"Failed to calculate Calinski-Harabasz score: {e}")
                metrics["calinski_harabasz_score"] = None
        
        # Davies-Bouldin score (lower is better)
        if (davies_bouldin_score is not None and 
            clustering_result.n_clusters > 1):
            try:
                non_noise_mask = clustering_result.labels != -1
                if np.sum(non_noise_mask) > 1:
                    metrics["davies_bouldin_score"] = davies_bouldin_score(
                        embeddings[non_noise_mask],
                        clustering_result.labels[non_noise_mask]
                    )
            except Exception as e:
                self.logger.warning(f"Failed to calculate Davies-Bouldin score: {e}")
                metrics["davies_bouldin_score"] = None
        
        # Inertia (within-cluster sum of squares)
        metrics["inertia"] = self._calculate_inertia(embeddings, clustering_result)
        
        # Cluster coherence
        metrics["cluster_coherence"] = self._calculate_cluster_coherence(
            embeddings, clustering_result
        )
        
        return metrics
    
    def _calculate_diversity_metrics(
        self,
        clustering_result: ClusteringResult,
        original_items: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Calculate diversity metrics for clustering."""
        metrics = {
            "cluster_size_diversity": self._calculate_cluster_size_diversity(
                clustering_result
            ),
            "source_diversity": None,
            "temporal_diversity": None,
            "topic_diversity": None
        }
        
        if original_items and len(original_items) == len(clustering_result.labels):
            # Source diversity
            metrics["source_diversity"] = self._calculate_source_diversity(
                clustering_result, original_items
            )
            
            # Temporal diversity
            metrics["temporal_diversity"] = self._calculate_temporal_diversity(
                clustering_result, original_items
            )
            
            # Topic diversity (based on technologies/companies)
            metrics["topic_diversity"] = self._calculate_topic_diversity(
                clustering_result, original_items
            )
        
        return metrics
    
    def _calculate_performance_metrics(
        self,
        clustering_result: ClusteringResult,
        n_items: int
    ) -> Dict[str, Any]:
        """Calculate performance-related metrics."""
        return {
            "processing_time": clustering_result.processing_time,
            "items_per_second": (
                n_items / clustering_result.processing_time 
                if clustering_result.processing_time > 0 else 0
            ),
            "noise_ratio": clustering_result.noise_ratio,
            "largest_cluster_ratio": (
                clustering_result.largest_cluster_size / n_items 
                if n_items > 0 else 0
            ),
            "cluster_count": clustering_result.n_clusters,
            "algorithm": clustering_result.algorithm
        }
    
    def _validate_against_thresholds(
        self,
        quality_metrics: Dict[str, Any],
        diversity_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate metrics against configured thresholds."""
        issues = []
        recommendations = []
        
        # Silhouette score validation
        silhouette = quality_metrics.get("silhouette_score")
        if silhouette is not None:
            if silhouette < self.config.min_silhouette_score:
                issues.append(
                    f"Silhouette score ({silhouette:.3f}) below minimum "
                    f"({self.config.min_silhouette_score:.3f})"
                )
                recommendations.append(
                    "Consider adjusting clustering parameters or using "
                    "different algorithm"
                )
        
        # Cluster coherence validation
        coherence = quality_metrics.get("cluster_coherence")
        if coherence is not None:
            if coherence < self.config.min_cluster_coherence:
                issues.append(
                    f"Cluster coherence ({coherence:.3f}) below minimum "
                    f"({self.config.min_cluster_coherence:.3f})"
                )
                recommendations.append(
                    "Clusters may be too loose. Consider stricter parameters"
                )
        
        # Source diversity validation
        if self.config.enable_diversity_check:
            source_diversity = diversity_metrics.get("source_diversity")
            if source_diversity is not None:
                if source_diversity < self.config.min_source_diversity:
                    issues.append(
                        f"Source diversity ({source_diversity:.3f}) below minimum "
                        f"({self.config.min_source_diversity:.3f})"
                    )
                    recommendations.append(
                        "Clusters may be biased toward specific sources"
                    )
        
        return {
            "issues": issues,
            "recommendations": recommendations
        }
    
    def _calculate_overall_quality_score(
        self,
        quality_metrics: Dict[str, Any],
        diversity_metrics: Dict[str, Any],
        performance_metrics: Dict[str, Any]
    ) -> float:
        """Calculate overall quality score (0-1, higher is better)."""
        score_components = []
        
        # Quality component (40% weight)
        silhouette = quality_metrics.get("silhouette_score")
        if silhouette is not None:
            # Normalize silhouette from [-1, 1] to [0, 1]
            normalized_silhouette = (silhouette + 1) / 2
            score_components.append(normalized_silhouette * 0.4)
        
        # Coherence component (30% weight)
        coherence = quality_metrics.get("cluster_coherence")
        if coherence is not None:
            score_components.append(coherence * 0.3)
        
        # Diversity component (20% weight)
        source_diversity = diversity_metrics.get("source_diversity")
        if source_diversity is not None:
            score_components.append(source_diversity * 0.2)
        
        # Performance component (10% weight)
        noise_ratio = performance_metrics.get("noise_ratio", 0)
        noise_score = max(0, 1 - noise_ratio * 2)  # Penalize high noise
        score_components.append(noise_score * 0.1)
        
        # Calculate final score
        if score_components:
            return sum(score_components)
        else:
            return 0.5  # Default neutral score
    
    def _calculate_inertia(
        self,
        embeddings: np.ndarray,
        clustering_result: ClusteringResult
    ) -> float:
        """Calculate within-cluster sum of squares."""
        if clustering_result.n_clusters == 0:
            return float('inf')
        
        total_inertia = 0.0
        
        for cluster_id in range(clustering_result.n_clusters):
            cluster_mask = clustering_result.labels == cluster_id
            cluster_points = embeddings[cluster_mask]
            
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
                cluster_inertia = np.sum((cluster_points - centroid) ** 2)
                total_inertia += cluster_inertia
        
        return total_inertia
    
    def _calculate_cluster_coherence(
        self,
        embeddings: np.ndarray,
        clustering_result: ClusteringResult
    ) -> float:
        """Calculate average intra-cluster similarity."""
        if clustering_result.n_clusters == 0:
            return 0.0
        
        coherence_scores = []
        
        for cluster_id in range(clustering_result.n_clusters):
            cluster_mask = clustering_result.labels == cluster_id
            cluster_points = embeddings[cluster_mask]
            
            if len(cluster_points) < 2:
                continue
            
            # Calculate pairwise similarities within cluster
            similarities = []
            for i in range(len(cluster_points)):
                for j in range(i + 1, len(cluster_points)):
                    similarity = np.dot(cluster_points[i], cluster_points[j])
                    similarities.append(similarity)
            
            if similarities:
                coherence_scores.append(np.mean(similarities))
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def _calculate_cluster_size_diversity(
        self,
        clustering_result: ClusteringResult
    ) -> float:
        """Calculate diversity of cluster sizes (higher is more balanced)."""
        if clustering_result.n_clusters <= 1:
            return 1.0
        
        # Get cluster sizes (excluding noise)
        cluster_sizes = [
            size for label, size in clustering_result.cluster_sizes.items()
            if label != -1
        ]
        
        if not cluster_sizes:
            return 0.0
        
        # Calculate entropy of cluster size distribution
        total_items = sum(cluster_sizes)
        proportions = [size / total_items for size in cluster_sizes]
        
        entropy = -sum(p * np.log2(p) for p in proportions if p > 0)
        max_entropy = np.log2(len(cluster_sizes))
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _calculate_source_diversity(
        self,
        clustering_result: ClusteringResult,
        original_items: List[Dict[str, Any]]
    ) -> float:
        """Calculate source diversity within clusters."""
        if clustering_result.n_clusters == 0:
            return 0.0
        
        cluster_diversities = []
        
        for cluster_id in range(clustering_result.n_clusters):
            cluster_indices = np.where(clustering_result.labels == cluster_id)[0]
            
            if len(cluster_indices) < 2:
                continue
            
            # Get sources for this cluster
            cluster_sources = [
                original_items[i].get("source", "unknown")
                for i in cluster_indices
            ]
            
            # Calculate source diversity (entropy)
            source_counts = Counter(cluster_sources)
            total_items = len(cluster_sources)
            
            if total_items <= 1:
                continue
            
            proportions = [count / total_items for count in source_counts.values()]
            entropy = -sum(p * np.log2(p) for p in proportions if p > 0)
            max_entropy = np.log2(len(source_counts))
            
            diversity = entropy / max_entropy if max_entropy > 0 else 0.0
            cluster_diversities.append(diversity)
        
        return np.mean(cluster_diversities) if cluster_diversities else 0.0
    
    def _calculate_temporal_diversity(
        self,
        clustering_result: ClusteringResult,
        original_items: List[Dict[str, Any]]
    ) -> float:
        """Calculate temporal diversity within clusters."""
        # This is a placeholder - would need actual date parsing
        # For now, return a neutral score
        return 0.5
    
    def _calculate_topic_diversity(
        self,
        clustering_result: ClusteringResult,
        original_items: List[Dict[str, Any]]
    ) -> float:
        """Calculate topic diversity based on technologies/companies."""
        if clustering_result.n_clusters == 0:
            return 0.0
        
        cluster_diversities = []
        
        for cluster_id in range(clustering_result.n_clusters):
            cluster_indices = np.where(clustering_result.labels == cluster_id)[0]
            
            if len(cluster_indices) < 2:
                continue
            
            # Collect all technologies and companies for this cluster
            all_topics = []
            for i in cluster_indices:
                item = original_items[i]
                technologies = item.get("technologies", [])
                companies = item.get("companies", [])
                
                if isinstance(technologies, str):
                    technologies = [t.strip() for t in technologies.split(",") if t.strip()]
                if isinstance(companies, str):
                    companies = [c.strip() for c in companies.split(",") if c.strip()]
                
                all_topics.extend(technologies)
                all_topics.extend(companies)
            
            if not all_topics:
                continue
            
            # Calculate topic diversity
            topic_counts = Counter(all_topics)
            total_topics = len(all_topics)
            
            proportions = [count / total_topics for count in topic_counts.values()]
            entropy = -sum(p * np.log2(p) for p in proportions if p > 0)
            max_entropy = np.log2(len(topic_counts))
            
            diversity = entropy / max_entropy if max_entropy > 0 else 0.0
            cluster_diversities.append(diversity)
        
        return np.mean(cluster_diversities) if cluster_diversities else 0.0
    
    def compare_clustering_results(
        self,
        result1: ClusteringResult,
        result2: ClusteringResult
    ) -> Dict[str, Any]:
        """Compare two clustering results."""
        comparison = {
            "algorithm_1": result1.algorithm,
            "algorithm_2": result2.algorithm,
            "clusters_1": result1.n_clusters,
            "clusters_2": result2.n_clusters,
            "noise_1": result1.n_noise,
            "noise_2": result2.n_noise,
            "processing_time_1": result1.processing_time,
            "processing_time_2": result2.processing_time,
            "silhouette_1": result1.silhouette_score,
            "silhouette_2": result2.silhouette_score
        }
        
        # Calculate agreement metrics if possible
        if (adjusted_rand_score is not None and 
            len(result1.labels) == len(result2.labels)):
            try:
                comparison["adjusted_rand_score"] = adjusted_rand_score(
                    result1.labels, result2.labels
                )
            except Exception as e:
                self.logger.warning(f"Failed to calculate ARI: {e}")
        
        if (normalized_mutual_info_score is not None and 
            len(result1.labels) == len(result2.labels)):
            try:
                comparison["normalized_mutual_info"] = normalized_mutual_info_score(
                    result1.labels, result2.labels
                )
            except Exception as e:
                self.logger.warning(f"Failed to calculate NMI: {e}")
        
        return comparison