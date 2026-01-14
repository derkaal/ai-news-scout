"""
Metrics Collection System

Centralized metrics collection for performance, quality, and business metrics.
"""

import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from collections import defaultdict, deque
from pathlib import Path
import psutil

from .config import MonitoringConfig


class MetricsCollector:
    """
    Centralized metrics collection and storage system.
    
    Collects performance, quality, and business metrics with configurable
    retention and export capabilities.
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self._metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.metrics.max_data_points)
        )
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Collection state
        self._collection_active = False
        self._collection_thread: Optional[threading.Thread] = None
        
        # System metrics
        self._process = psutil.Process()
        
        self.logger.info("MetricsCollector initialized")
    
    def start_collection(self):
        """Start automatic metrics collection."""
        if self._collection_active:
            self.logger.warning("Metrics collection already active")
            return
        
        self._collection_active = True
        self._collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True
        )
        self._collection_thread.start()
        self.logger.info("Started metrics collection")
    
    def stop_collection(self):
        """Stop automatic metrics collection."""
        self._collection_active = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5.0)
        self.logger.info("Stopped metrics collection")
    
    def record_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Record a counter metric."""
        with self._lock:
            metric_key = self._build_metric_key(name, tags)
            self._counters[metric_key] += value
            
            self._record_timestamped_metric(metric_key, {
                'type': 'counter',
                'value': self._counters[metric_key],
                'increment': value,
                'tags': tags or {}
            })
    
    def record_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a gauge metric."""
        with self._lock:
            metric_key = self._build_metric_key(name, tags)
            self._gauges[metric_key] = value
            
            self._record_timestamped_metric(metric_key, {
                'type': 'gauge',
                'value': value,
                'tags': tags or {}
            })
    
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram metric."""
        with self._lock:
            metric_key = self._build_metric_key(name, tags)
            self._histograms[metric_key].append(value)
            
            # Keep only recent values for memory efficiency
            if len(self._histograms[metric_key]) > 1000:
                self._histograms[metric_key] = self._histograms[metric_key][-1000:]
            
            self._record_timestamped_metric(metric_key, {
                'type': 'histogram',
                'value': value,
                'tags': tags or {}
            })
    
    def record_timing(self, name: str, duration_seconds: float, tags: Dict[str, str] = None):
        """Record a timing metric."""
        self.record_histogram(f"{name}.duration", duration_seconds, tags)
        self.record_counter(f"{name}.count", 1, tags)
    
    def record_clustering_metrics(self, clustering_result: Dict[str, Any]):
        """Record clustering-specific metrics."""
        tags = {
            'algorithm': clustering_result.get('algorithm_used', 'unknown')
        }
        
        # Performance metrics
        processing_time = clustering_result.get('processing_time', 0)
        self.record_timing('clustering.processing', processing_time, tags)
        
        # Quality metrics
        if 'validation' in clustering_result:
            validation = clustering_result['validation']
            
            if 'validation_metrics' in validation:
                metrics = validation['validation_metrics']
                
                if 'silhouette_score' in metrics and metrics['silhouette_score'] is not None:
                    self.record_gauge(
                        'clustering.quality.silhouette_score',
                        metrics['silhouette_score'],
                        tags
                    )
                
                if 'cluster_coherence' in metrics and metrics['cluster_coherence'] is not None:
                    self.record_gauge(
                        'clustering.quality.coherence',
                        metrics['cluster_coherence'],
                        tags
                    )
            
            # Overall quality score
            if 'quality_score' in validation:
                self.record_gauge(
                    'clustering.quality.overall_score',
                    validation['quality_score'],
                    tags
                )
        
        # Cluster statistics
        self.record_gauge('clustering.clusters.count', clustering_result.get('total_clusters', 0), tags)
        self.record_gauge('clustering.items.total', clustering_result.get('total_items', 0), tags)
        self.record_gauge('clustering.items.noise', clustering_result.get('noise_items', 0), tags)
        
        # Noise ratio
        if clustering_result.get('total_items', 0) > 0:
            noise_ratio = clustering_result.get('noise_items', 0) / clustering_result['total_items']
            self.record_gauge('clustering.quality.noise_ratio', noise_ratio, tags)
    
    def record_performance_metrics(self, performance_data: Dict[str, Any]):
        """Record system performance metrics."""
        # Memory metrics
        if 'peak_memory_mb' in performance_data:
            self.record_gauge('system.memory.peak_mb', performance_data['peak_memory_mb'])
        
        if 'peak_memory_gb' in performance_data:
            self.record_gauge('system.memory.peak_gb', performance_data['peak_memory_gb'])
        
        # Processing time
        if 'elapsed_time' in performance_data:
            self.record_timing('system.processing', performance_data['elapsed_time'])
        
        # Memory limit violations
        if performance_data.get('memory_limit_exceeded', False):
            self.record_counter('system.memory.limit_exceeded')
    
    def record_business_metrics(self, business_data: Dict[str, Any]):
        """Record business-related metrics."""
        # Success/failure rates
        if 'successful_clusterings' in business_data:
            self.record_counter('business.clustering.success', business_data['successful_clusterings'])
        
        if 'failed_clusterings' in business_data:
            self.record_counter('business.clustering.failure', business_data['failed_clusterings'])
        
        # Cache performance
        if 'cache_hit_rate' in business_data:
            self.record_gauge('business.cache.hit_rate', business_data['cache_hit_rate'])
        
        # Processing throughput
        if 'total_items_processed' in business_data and 'total_processing_time' in business_data:
            if business_data['total_processing_time'] > 0:
                throughput = business_data['total_items_processed'] / business_data['total_processing_time']
                self.record_gauge('business.throughput.items_per_second', throughput)
    
    def get_metrics_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get summary of metrics within a time window."""
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        
        with self._lock:
            summary = {
                'counters': dict(self._counters),
                'gauges': dict(self._gauges),
                'histograms': {},
                'recent_metrics': {},
                'time_window_minutes': time_window_minutes
            }
            
            # Histogram summaries
            for name, values in self._histograms.items():
                if values:
                    summary['histograms'][name] = {
                        'count': len(values),
                        'min': min(values),
                        'max': max(values),
                        'avg': sum(values) / len(values),
                        'p50': self._percentile(values, 50),
                        'p95': self._percentile(values, 95),
                        'p99': self._percentile(values, 99)
                    }
            
            # Recent metrics within time window
            for metric_name, metric_data in self._metrics.items():
                recent_data = [
                    entry for entry in metric_data
                    if datetime.fromisoformat(entry['timestamp']) >= cutoff_time
                ]
                if recent_data:
                    summary['recent_metrics'][metric_name] = recent_data
        
        return summary
    
    def export_metrics(self, format_type: str = 'json') -> str:
        """Export metrics in specified format."""
        summary = self.get_metrics_summary()
        
        if format_type.lower() == 'json':
            return json.dumps(summary, indent=2, default=str)
        elif format_type.lower() == 'prometheus':
            return self._export_prometheus_format(summary)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def clear_metrics(self):
        """Clear all collected metrics."""
        with self._lock:
            self._metrics.clear()
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
        self.logger.info("Cleared all metrics")
    
    def _collection_loop(self):
        """Main collection loop for system metrics."""
        while self._collection_active:
            try:
                self._collect_system_metrics()
                time.sleep(self.config.metrics.collection_interval_seconds)
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(5)  # Brief pause before retrying
    
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_gauge('system.cpu.usage_percent', cpu_percent)
            
            # Memory usage
            memory_info = self._process.memory_info()
            self.record_gauge('system.memory.rss_mb', memory_info.rss / (1024 * 1024))
            self.record_gauge('system.memory.vms_mb', memory_info.vms / (1024 * 1024))
            
            # System memory
            system_memory = psutil.virtual_memory()
            self.record_gauge('system.memory.system_usage_percent', system_memory.percent)
            
            # Disk usage
            disk_usage = psutil.disk_usage('/')
            self.record_gauge('system.disk.usage_percent', 
                            (disk_usage.used / disk_usage.total) * 100)
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def _build_metric_key(self, name: str, tags: Dict[str, str] = None) -> str:
        """Build a unique metric key including tags."""
        if not tags:
            return name
        
        tag_string = ','.join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_string}]"
    
    def _record_timestamped_metric(self, metric_key: str, data: Dict[str, Any]):
        """Record a metric with timestamp."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            **data
        }
        self._metrics[metric_key].append(entry)
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int((percentile / 100.0) * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]
    
    def _export_prometheus_format(self, summary: Dict[str, Any]) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Counters
        for name, value in summary['counters'].items():
            clean_name = name.replace('.', '_').replace('[', '_').replace(']', '').replace('=', '_').replace(',', '_')
            lines.append(f"# TYPE {clean_name} counter")
            lines.append(f"{clean_name} {value}")
        
        # Gauges
        for name, value in summary['gauges'].items():
            clean_name = name.replace('.', '_').replace('[', '_').replace(']', '').replace('=', '_').replace(',', '_')
            lines.append(f"# TYPE {clean_name} gauge")
            lines.append(f"{clean_name} {value}")
        
        return '\n'.join(lines)