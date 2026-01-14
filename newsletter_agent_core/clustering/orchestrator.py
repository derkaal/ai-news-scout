"""
Clustering Orchestrator

Main coordination service for the newsletter clustering engine.
Manages the complete clustering workflow from embeddings to validation.
"""

import logging
import time
import psutil
import threading
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import numpy as np

from .config.settings import ClusteringConfig
from .embedding.service import EmbeddingService
from .algorithms.hdbscan_clusterer import HDBSCANClusterer
from .algorithms.hierarchical_clusterer import HierarchicalClusterer
from .algorithms.hybrid_clusterer import HybridClusterer
from .algorithms.hdbscan_clusterer import ClusteringResult
from .validation.validator import ClusteringValidator, ValidationResult


class PerformanceMonitor:
    """Monitor performance metrics during clustering."""
    
    def __init__(self, max_memory_gb: float):
        self.max_memory_gb = max_memory_gb
        self.start_time = None
        self.peak_memory_mb = 0
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.peak_memory_mb = 0
        self.monitoring = True
        
        # Start memory monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_memory)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return metrics."""
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            "elapsed_time": elapsed_time,
            "peak_memory_mb": self.peak_memory_mb,
            "peak_memory_gb": self.peak_memory_mb / 1024,
            "memory_limit_exceeded": self.peak_memory_mb / 1024 > self.max_memory_gb
        }
    
    def _monitor_memory(self):
        """Monitor memory usage in background thread."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                memory_mb = process.memory_info().rss / (1024 * 1024)
                self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
                time.sleep(0.1)  # Check every 100ms
            except Exception:
                break


class ClusteringOrchestrator:
    """
    Main orchestrator for the clustering engine.
    
    Coordinates embedding generation, clustering, validation, and result processing.
    """
    
    def __init__(self, config: Optional[ClusteringConfig] = None):
        self.config = config or ClusteringConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
        # Performance tracking
        self.stats = {
            "total_items_processed": 0,
            "total_processing_time": 0.0,
            "successful_clusterings": 0,
            "failed_clusterings": 0,
            "cache_hit_rate": 0.0
        }
    
    def _initialize_components(self):
        """Initialize all clustering components."""
        try:
            # Initialize embedding service
            self.embedding_service = EmbeddingService(
                self.config.embedding,
                self.config.cache
            )
            
            # Initialize clustering algorithms conditionally
            self.hdbscan_clusterer = None
            self.hierarchical_clusterer = None
            self.hybrid_clusterer = None
            
            # Only initialize the algorithms we can actually use
            try:
                self.hdbscan_clusterer = HDBSCANClusterer(self.config.hdbscan)
                self.logger.info("HDBSCAN clusterer initialized successfully")
            except ImportError as e:
                self.logger.warning(f"HDBSCAN not available: {e}")
            
            try:
                self.hierarchical_clusterer = HierarchicalClusterer(self.config.hierarchical)
                self.logger.info("Hierarchical clusterer initialized successfully")
            except ImportError as e:
                self.logger.warning(f"Hierarchical clustering not available: {e}")
            
            try:
                if self.hdbscan_clusterer and self.hierarchical_clusterer:
                    self.hybrid_clusterer = HybridClusterer(
                        self.config.hybrid,
                        self.config.hdbscan,
                        self.config.hierarchical
                    )
                    self.logger.info("Hybrid clusterer initialized successfully")
                else:
                    self.logger.warning("Hybrid clustering not available (requires both HDBSCAN and hierarchical)")
            except ImportError as e:
                self.logger.warning(f"Hybrid clustering not available: {e}")
            
            # Initialize validator
            self.validator = ClusteringValidator(self.config.validation)
            
            self.logger.info("Clustering orchestrator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize clustering orchestrator: {e}")
            raise
    
    def cluster_items(
        self,
        items: List[Dict[str, Any]],
        text_field: str = "short_description",
        algorithm: Optional[str] = None,
        validate_results: bool = True
    ) -> Dict[str, Any]:
        """
        Perform complete clustering workflow on newsletter items.
        
        Args:
            items: List of newsletter items to cluster
            text_field: Field containing text to cluster on
            algorithm: Specific algorithm to use (overrides config)
            validate_results: Whether to validate clustering results
            
        Returns:
            Dictionary with clustering results and metadata
        """
        if not items:
            self.logger.warning("No items provided for clustering")
            return self._create_empty_result()
        
        self.logger.info(
            f"Starting clustering workflow for {len(items)} items "
            f"using {algorithm or self.config.default_algorithm} algorithm"
        )
        
        # Start performance monitoring
        monitor = PerformanceMonitor(self.config.performance.max_memory_usage_gb)
        monitor.start_monitoring()
        
        try:
            # Execute clustering with timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self._execute_clustering_workflow,
                    items, text_field, algorithm, validate_results
                )
                
                try:
                    result = future.result(
                        timeout=self.config.performance.max_processing_time_seconds
                    )
                except TimeoutError:
                    self.logger.error(
                        f"Clustering timed out after "
                        f"{self.config.performance.max_processing_time_seconds}s"
                    )
                    self.stats["failed_clusterings"] += 1
                    return self._create_timeout_result(items)
            
            # Stop monitoring and add performance metrics
            performance_metrics = monitor.stop_monitoring()
            result["performance_metrics"] = performance_metrics
            
            # Check memory constraints
            if performance_metrics["memory_limit_exceeded"]:
                self.logger.warning(
                    f"Memory limit exceeded: {performance_metrics['peak_memory_gb']:.2f}GB "
                    f"> {self.config.performance.max_memory_usage_gb}GB"
                )
                result["warnings"] = result.get("warnings", [])
                result["warnings"].append("Memory limit exceeded during processing")
            
            # Update stats
            self.stats["total_items_processed"] += len(items)
            self.stats["total_processing_time"] += performance_metrics["elapsed_time"]
            self.stats["successful_clusterings"] += 1
            
            # Update cache hit rate
            embedding_stats = self.embedding_service.get_cache_stats()
            self.stats["cache_hit_rate"] = embedding_stats.get("hit_rate", 0.0)
            
            self.logger.info(
                f"Clustering workflow completed successfully in "
                f"{performance_metrics['elapsed_time']:.2f}s"
            )
            
            return result
            
        except Exception as e:
            monitor.stop_monitoring()
            self.logger.error(f"Clustering workflow failed: {e}")
            self.stats["failed_clusterings"] += 1
            return self._create_error_result(items, str(e))
    
    def _execute_clustering_workflow(
        self,
        items: List[Dict[str, Any]],
        text_field: str,
        algorithm: Optional[str],
        validate_results: bool
    ) -> Dict[str, Any]:
        """Execute the main clustering workflow."""
        
        # Step 1: Extract texts for embedding
        texts = self._extract_texts(items, text_field)
        if not texts:
            raise ValueError(f"No valid texts found in field '{text_field}'")
        
        # Step 2: Generate embeddings
        self.logger.info("Generating embeddings...")
        embeddings = self.embedding_service.generate_embeddings(
            texts,
            show_progress=self.config.performance.enable_progress_tracking
        )
        
        if len(embeddings) == 0:
            raise ValueError("Failed to generate embeddings")
        
        # Step 3: Perform clustering
        self.logger.info("Performing clustering...")
        clustering_result = self._perform_clustering(
            embeddings, algorithm or self.config.default_algorithm
        )
        
        # Step 4: Validate results if requested
        validation_result = None
        if validate_results:
            self.logger.info("Validating clustering results...")
            validation_result = self.validator.validate_clustering(
                embeddings, clustering_result, items
            )
        
        # Step 5: Process and enhance results
        self.logger.info("Processing results...")
        processed_result = self._process_clustering_results(
            items, clustering_result, validation_result, embeddings
        )
        
        return processed_result
    
    def _extract_texts(
        self, 
        items: List[Dict[str, Any]], 
        text_field: str
    ) -> List[str]:
        """Extract texts from items for embedding."""
        texts = []
        
        for item in items:
            text = item.get(text_field, "")
            
            # Handle different text field formats
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
            elif isinstance(text, list) and text:
                # Join list elements
                texts.append(" ".join(str(t) for t in text if t))
            else:
                # Fallback to other fields
                fallback_text = (
                    item.get("headline", "") + " " + 
                    item.get("master_headline", "")
                ).strip()
                
                if fallback_text:
                    texts.append(fallback_text)
                else:
                    texts.append("No content available")
        
        return texts
    
    def _perform_clustering(
        self,
        embeddings: np.ndarray,
        algorithm: str
    ) -> ClusteringResult:
        """Perform clustering using specified algorithm."""
        
        if algorithm == "hdbscan":
            if self.hdbscan_clusterer is None:
                raise ValueError("HDBSCAN clustering not available. Install hdbscan package.")
            return self.hdbscan_clusterer.fit_predict(embeddings)
        elif algorithm == "hierarchical":
            if self.hierarchical_clusterer is None:
                raise ValueError("Hierarchical clustering not available. Install scikit-learn package.")
            return self.hierarchical_clusterer.fit_predict(embeddings)
        elif algorithm == "hybrid":
            if self.hybrid_clusterer is None:
                raise ValueError("Hybrid clustering not available. Requires both HDBSCAN and hierarchical clustering.")
            return self.hybrid_clusterer.fit_predict(embeddings)
        else:
            raise ValueError(f"Unknown clustering algorithm: {algorithm}")
    
    def _process_clustering_results(
        self,
        items: List[Dict[str, Any]],
        clustering_result: ClusteringResult,
        validation_result: Optional[ValidationResult],
        embeddings: np.ndarray
    ) -> Dict[str, Any]:
        """Process and enhance clustering results."""
        
        # Create enhanced items with cluster information
        enhanced_items = []
        for i, item in enumerate(items):
            enhanced_item = item.copy() if self.config.preserve_original_data else {}
            
            # Add clustering metadata
            if self.config.add_clustering_metadata:
                cluster_id = int(clustering_result.labels[i])
                
                enhanced_item.update({
                    "cluster_id": cluster_id,
                    "is_noise": cluster_id == -1,
                    "cluster_probability": (
                        float(clustering_result.probabilities[i])
                        if clustering_result.probabilities is not None else None
                    ),
                    "outlier_score": (
                        float(clustering_result.outlier_scores[i])
                        if clustering_result.outlier_scores is not None else None
                    )
                })
            
            enhanced_items.append(enhanced_item)
        
        # Create cluster summaries
        cluster_summaries = self._create_cluster_summaries(
            enhanced_items, clustering_result, embeddings
        )
        
        # Prepare result
        result = {
            "items": enhanced_items,
            "clustering_result": clustering_result.to_dict(),
            "cluster_summaries": cluster_summaries,
            "algorithm_used": clustering_result.algorithm,
            "total_items": len(items),
            "total_clusters": clustering_result.n_clusters,
            "noise_items": clustering_result.n_noise,
            "processing_time": clustering_result.processing_time
        }
        
        # Add validation results if available
        if validation_result:
            result["validation"] = validation_result.to_dict()
            result["is_valid"] = validation_result.is_valid
            result["quality_score"] = validation_result.quality_score
            
            if validation_result.issues:
                result["issues"] = validation_result.issues
            
            if validation_result.recommendations:
                result["recommendations"] = validation_result.recommendations
        
        return result
    
    def _create_cluster_summaries(
        self,
        items: List[Dict[str, Any]],
        clustering_result: ClusteringResult,
        embeddings: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Create summaries for each cluster."""
        summaries = []
        
        for cluster_id in range(clustering_result.n_clusters):
            cluster_indices = np.where(clustering_result.labels == cluster_id)[0]
            cluster_items = [items[i] for i in cluster_indices]
            cluster_embeddings = embeddings[cluster_indices]
            
            # Calculate cluster centroid
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Find representative items (closest to centroid)
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            representative_indices = np.argsort(distances)[:3]  # Top 3
            representative_items = [cluster_items[i] for i in representative_indices]
            
            # Extract common themes
            sources = [item.get("source", "Unknown") for item in cluster_items]
            technologies = []
            companies = []
            
            for item in cluster_items:
                item_techs = item.get("technologies", [])
                item_companies = item.get("companies", [])
                
                if isinstance(item_techs, str):
                    item_techs = [t.strip() for t in item_techs.split(",") if t.strip()]
                if isinstance(item_companies, str):
                    item_companies = [c.strip() for c in item_companies.split(",") if c.strip()]
                
                technologies.extend(item_techs)
                companies.extend(item_companies)
            
            # Count occurrences
            from collections import Counter
            source_counts = Counter(sources)
            tech_counts = Counter(technologies)
            company_counts = Counter(companies)
            
            summary = {
                "cluster_id": cluster_id,
                "size": len(cluster_items),
                "representative_items": [
                    {
                        "headline": item.get("headline", ""),
                        "source": item.get("source", ""),
                        "short_description": item.get("short_description", "")[:100] + "..."
                    }
                    for item in representative_items
                ],
                "top_sources": dict(source_counts.most_common(3)),
                "top_technologies": dict(tech_counts.most_common(5)),
                "top_companies": dict(company_counts.most_common(5)),
                "avg_cluster_distance": float(np.mean(distances))
            }
            
            summaries.append(summary)
        
        # Sort by cluster size (largest first)
        summaries.sort(key=lambda x: x["size"], reverse=True)
        
        return summaries
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create result for empty input."""
        return {
            "items": [],
            "clustering_result": {
                "algorithm": "none",
                "n_clusters": 0,
                "n_noise": 0,
                "processing_time": 0.0,
                "labels": []
            },
            "cluster_summaries": [],
            "total_items": 0,
            "total_clusters": 0,
            "noise_items": 0,
            "processing_time": 0.0,
            "is_valid": False,
            "quality_score": 0.0
        }
    
    def _create_timeout_result(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create result for timeout scenario."""
        return {
            "items": items,
            "clustering_result": {
                "algorithm": "timeout",
                "n_clusters": 1,
                "n_noise": 0,
                "processing_time": self.config.performance.max_processing_time_seconds,
                "labels": [0] * len(items)  # Single cluster fallback
            },
            "cluster_summaries": [{
                "cluster_id": 0,
                "size": len(items),
                "representative_items": items[:3],
                "top_sources": {},
                "top_technologies": {},
                "top_companies": {},
                "avg_cluster_distance": 0.0
            }],
            "total_items": len(items),
            "total_clusters": 1,
            "noise_items": 0,
            "processing_time": self.config.performance.max_processing_time_seconds,
            "is_valid": False,
            "quality_score": 0.0,
            "error": "Clustering timed out"
        }
    
    def _create_error_result(
        self, 
        items: List[Dict[str, Any]], 
        error_message: str
    ) -> Dict[str, Any]:
        """Create result for error scenario."""
        return {
            "items": items,
            "clustering_result": {
                "algorithm": "error",
                "n_clusters": 0,
                "n_noise": len(items),
                "processing_time": 0.0,
                "labels": [-1] * len(items)  # All noise
            },
            "cluster_summaries": [],
            "total_items": len(items),
            "total_clusters": 0,
            "noise_items": len(items),
            "processing_time": 0.0,
            "is_valid": False,
            "quality_score": 0.0,
            "error": error_message
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        embedding_stats = self.embedding_service.get_performance_stats()
        cache_stats = self.embedding_service.get_cache_stats()
        
        # Only get stats from available clusterers
        algorithm_stats = {}
        if self.hdbscan_clusterer is not None:
            algorithm_stats["hdbscan"] = self.hdbscan_clusterer.get_performance_stats()
        if self.hierarchical_clusterer is not None:
            algorithm_stats["hierarchical"] = self.hierarchical_clusterer.get_performance_stats()
        if self.hybrid_clusterer is not None:
            algorithm_stats["hybrid"] = self.hybrid_clusterer.get_performance_stats()
        
        return {
            "orchestrator_stats": self.stats,
            "embedding_stats": embedding_stats,
            "cache_stats": cache_stats,
            "algorithm_stats": algorithm_stats
        }
    
    def clear_cache(self):
        """Clear all caches."""
        self.embedding_service.clear_cache()
        self.logger.info("All caches cleared")
    
    def update_config(self, new_config: ClusteringConfig):
        """Update configuration and reinitialize components."""
        self.config = new_config
        self._initialize_components()
        self.logger.info("Configuration updated and components reinitialized")