"""
Performance Monitoring System

Monitors system performance, processing times, memory usage, and cache efficiency.
"""

import time
import logging
import threading
import psutil
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager
from datetime import datetime, timedelta

from .config import MonitoringConfig
from .metrics_collector import MetricsCollector


class PerformanceMonitor:
    """
    Performance monitoring system for the newsletter clustering agent.
    
    Tracks processing times, memory usage, cache performance, and system
    resource utilization with threshold-based alerting.
    """
    
    def __init__(self, config: MonitoringConfig, metrics_collector: MetricsCollector):
        self.config = config
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self._active_operations: Dict[str, Dict[str, Any]] = {}
        self._performance_history: Dict[str, list] = {}
        self._lock = threading.RLock()
        
        # System monitoring
        self._process = psutil.Process()
        self._baseline_memory_mb = self._get_current_memory_mb()
        
        # Alert callbacks
        self._alert_callbacks: Dict[str, Callable] = {}
        
        self.logger.info("PerformanceMonitor initialized")
    
    def register_alert_callback(self, alert_type: str, callback: Callable):
        """Register callback for performance alerts."""
        self._alert_callbacks[alert_type] = callback
        self.logger.info(f"Registered alert callback for {alert_type}")
    
    @contextmanager
    def monitor_operation(self, operation_name: str, tags: Dict[str, str] = None):
        """
        Context manager for monitoring operation performance.
        
        Usage:
            with performance_monitor.monitor_operation('clustering'):
                # perform clustering operation
                pass
        """
        operation_id = f"{operation_name}_{int(time.time() * 1000)}"
        start_time = time.time()
        start_memory = self._get_current_memory_mb()
        
        # Record operation start
        with self._lock:
            self._active_operations[operation_id] = {
                'name': operation_name,
                'start_time': start_time,
                'start_memory_mb': start_memory,
                'tags': tags or {}
            }
        
        try:
            yield operation_id
        finally:
            # Record operation completion
            end_time = time.time()
            end_memory = self._get_current_memory_mb()
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Remove from active operations
            with self._lock:
                operation_data = self._active_operations.pop(operation_id, {})
            
            # Record metrics
            self._record_operation_metrics(
                operation_name, duration, memory_delta, tags
            )
            
            # Check thresholds and alert if necessary
            self._check_performance_thresholds(
                operation_name, duration, end_memory, tags
            )
            
            self.logger.debug(
                f"Operation {operation_name} completed in {duration:.2f}s, "
                f"memory delta: {memory_delta:.1f}MB"
            )
    
    def monitor_cache_performance(self, cache_stats: Dict[str, Any]):
        """Monitor cache performance metrics."""
        hit_rate = cache_stats.get('hit_rate', 0.0)
        total_requests = cache_stats.get('total_requests', 0)
        cache_size = cache_stats.get('cache_size', 0)
        
        # Record cache metrics
        self.metrics_collector.record_gauge('cache.hit_rate', hit_rate)
        self.metrics_collector.record_gauge('cache.total_requests', total_requests)
        self.metrics_collector.record_gauge('cache.size', cache_size)
        
        # Check cache hit rate threshold
        min_hit_rate = self.config.performance_thresholds.min_cache_hit_rate
        if hit_rate < min_hit_rate:
            self._trigger_alert(
                'cache_hit_rate_low',
                f"Cache hit rate ({hit_rate:.1%}) below threshold ({min_hit_rate:.1%})",
                {'hit_rate': hit_rate, 'threshold': min_hit_rate}
            )
    
    def monitor_queue_performance(self, queue_size: int, queue_name: str = 'default'):
        """Monitor queue performance metrics."""
        tags = {'queue': queue_name}
        
        self.metrics_collector.record_gauge('queue.size', queue_size, tags)
        
        # Check queue size threshold
        max_queue_size = self.config.performance_thresholds.max_queue_size
        if queue_size > max_queue_size:
            self._trigger_alert(
                'queue_size_high',
                f"Queue {queue_name} size ({queue_size}) exceeds threshold ({max_queue_size})",
                {'queue_name': queue_name, 'size': queue_size, 'threshold': max_queue_size}
            )
    
    def monitor_throughput(self, items_processed: int, time_window_seconds: float):
        """Monitor processing throughput."""
        if time_window_seconds > 0:
            throughput = items_processed / time_window_seconds
            throughput_per_minute = throughput * 60
            
            self.metrics_collector.record_gauge('throughput.items_per_second', throughput)
            self.metrics_collector.record_gauge('throughput.items_per_minute', throughput_per_minute)
            
            # Check throughput threshold
            min_throughput = self.config.performance_thresholds.min_throughput_items_per_minute
            if throughput_per_minute < min_throughput:
                self._trigger_alert(
                    'throughput_low',
                    f"Throughput ({throughput_per_minute:.1f} items/min) below threshold ({min_throughput} items/min)",
                    {'throughput': throughput_per_minute, 'threshold': min_throughput}
                )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        current_memory = self._get_current_memory_mb()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Get recent operation performance
        recent_operations = self._get_recent_operation_stats()
        
        # System resource usage
        system_memory = psutil.virtual_memory()
        disk_usage = psutil.disk_usage('/')
        
        return {
            'timestamp': datetime.now().isoformat(),
            'memory': {
                'current_mb': current_memory,
                'baseline_mb': self._baseline_memory_mb,
                'delta_mb': current_memory - self._baseline_memory_mb,
                'system_usage_percent': system_memory.percent,
                'system_available_gb': system_memory.available / (1024**3)
            },
            'cpu': {
                'usage_percent': cpu_percent,
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            },
            'disk': {
                'usage_percent': (disk_usage.used / disk_usage.total) * 100,
                'free_gb': disk_usage.free / (1024**3)
            },
            'operations': recent_operations,
            'active_operations': len(self._active_operations),
            'thresholds': {
                'max_processing_time_seconds': self.config.performance_thresholds.max_processing_time_seconds,
                'max_memory_usage_gb': self.config.performance_thresholds.max_memory_usage_gb,
                'min_cache_hit_rate': self.config.performance_thresholds.min_cache_hit_rate
            }
        }
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health status."""
        health_status = {
            'healthy': True,
            'issues': [],
            'warnings': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Check memory usage
        current_memory_gb = self._get_current_memory_mb() / 1024
        max_memory_gb = self.config.performance_thresholds.max_memory_usage_gb
        
        if current_memory_gb > max_memory_gb:
            health_status['healthy'] = False
            health_status['issues'].append(
                f"Memory usage ({current_memory_gb:.2f}GB) exceeds limit ({max_memory_gb}GB)"
            )
        elif current_memory_gb > max_memory_gb * 0.8:
            health_status['warnings'].append(
                f"Memory usage ({current_memory_gb:.2f}GB) approaching limit ({max_memory_gb}GB)"
            )
        
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 90:
            health_status['warnings'].append(f"High CPU usage: {cpu_percent:.1f}%")
        
        # Check disk space
        disk_usage = psutil.disk_usage('/')
        disk_percent = (disk_usage.used / disk_usage.total) * 100
        if disk_percent > 90:
            health_status['healthy'] = False
            health_status['issues'].append(f"Low disk space: {disk_percent:.1f}% used")
        elif disk_percent > 80:
            health_status['warnings'].append(f"Disk space warning: {disk_percent:.1f}% used")
        
        # Check for long-running operations
        current_time = time.time()
        max_processing_time = self.config.performance_thresholds.max_processing_time_seconds
        
        with self._lock:
            for op_id, op_data in self._active_operations.items():
                duration = current_time - op_data['start_time']
                if duration > max_processing_time:
                    health_status['warnings'].append(
                        f"Long-running operation: {op_data['name']} ({duration:.1f}s)"
                    )
        
        return health_status
    
    def _get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            return self._process.memory_info().rss / (1024 * 1024)
        except Exception as e:
            self.logger.error(f"Error getting memory usage: {e}")
            return 0.0
    
    def _record_operation_metrics(self, operation_name: str, duration: float, 
                                memory_delta: float, tags: Dict[str, str] = None):
        """Record operation performance metrics."""
        # Record timing metrics
        self.metrics_collector.record_timing(f"operation.{operation_name}", duration, tags)
        
        # Record memory metrics
        self.metrics_collector.record_histogram(
            f"operation.{operation_name}.memory_delta_mb", memory_delta, tags
        )
        
        # Store in performance history
        with self._lock:
            if operation_name not in self._performance_history:
                self._performance_history[operation_name] = []
            
            self._performance_history[operation_name].append({
                'timestamp': datetime.now().isoformat(),
                'duration': duration,
                'memory_delta_mb': memory_delta,
                'tags': tags or {}
            })
            
            # Keep only recent history
            if len(self._performance_history[operation_name]) > 1000:
                self._performance_history[operation_name] = \
                    self._performance_history[operation_name][-1000:]
    
    def _check_performance_thresholds(self, operation_name: str, duration: float,
                                    current_memory_mb: float, tags: Dict[str, str] = None):
        """Check performance against thresholds and trigger alerts."""
        # Check processing time threshold
        max_time = self.config.performance_thresholds.max_processing_time_seconds
        if duration > max_time:
            self._trigger_alert(
                'processing_time_exceeded',
                f"Operation {operation_name} took {duration:.2f}s (threshold: {max_time}s)",
                {
                    'operation': operation_name,
                    'duration': duration,
                    'threshold': max_time,
                    'tags': tags or {}
                }
            )
        
        # Check memory threshold
        max_memory_gb = self.config.performance_thresholds.max_memory_usage_gb
        current_memory_gb = current_memory_mb / 1024
        if current_memory_gb > max_memory_gb:
            self._trigger_alert(
                'memory_limit_exceeded',
                f"Memory usage {current_memory_gb:.2f}GB exceeds limit {max_memory_gb}GB",
                {
                    'operation': operation_name,
                    'memory_gb': current_memory_gb,
                    'threshold': max_memory_gb,
                    'tags': tags or {}
                }
            )
    
    def _trigger_alert(self, alert_type: str, message: str, context: Dict[str, Any]):
        """Trigger performance alert."""
        self.logger.warning(f"Performance alert [{alert_type}]: {message}")
        
        # Record alert metric
        self.metrics_collector.record_counter(f"alerts.{alert_type}")
        
        # Call registered callback if available
        if alert_type in self._alert_callbacks:
            try:
                self._alert_callbacks[alert_type](message, context)
            except Exception as e:
                self.logger.error(f"Error in alert callback for {alert_type}: {e}")
    
    def _get_recent_operation_stats(self, hours: int = 1) -> Dict[str, Any]:
        """Get statistics for recent operations."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        stats = {}
        
        with self._lock:
            for operation_name, history in self._performance_history.items():
                recent_ops = [
                    op for op in history
                    if datetime.fromisoformat(op['timestamp']) >= cutoff_time
                ]
                
                if recent_ops:
                    durations = [op['duration'] for op in recent_ops]
                    memory_deltas = [op['memory_delta_mb'] for op in recent_ops]
                    
                    stats[operation_name] = {
                        'count': len(recent_ops),
                        'avg_duration': sum(durations) / len(durations),
                        'max_duration': max(durations),
                        'min_duration': min(durations),
                        'avg_memory_delta_mb': sum(memory_deltas) / len(memory_deltas),
                        'max_memory_delta_mb': max(memory_deltas)
                    }
        
        return stats