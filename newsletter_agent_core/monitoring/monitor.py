"""
Main Monitoring Orchestrator

Coordinates all monitoring components and provides a unified interface.
"""

import logging
import signal
import sys
import threading
from typing import Dict, Any, Optional
from datetime import datetime

from .config import MonitoringConfig, get_monitoring_config
from .metrics_collector import MetricsCollector
from .performance_monitor import PerformanceMonitor
from .quality_monitor import QualityMonitor
from .health_monitor import HealthMonitor
from .alerting import AlertManager
from .health_endpoints import HealthEndpoints


class MonitoringOrchestrator:
    """
    Main monitoring orchestrator.
    
    Coordinates all monitoring components, manages lifecycle,
    and provides unified monitoring interface.
    """
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or get_monitoring_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.metrics_collector = MetricsCollector(self.config)
        self.performance_monitor = PerformanceMonitor(
            self.config, self.metrics_collector
        )
        self.quality_monitor = QualityMonitor(
            self.config, self.metrics_collector
        )
        self.health_monitor = HealthMonitor(
            self.config, self.metrics_collector
        )
        self.alert_manager = AlertManager(self.config)
        
        # Health endpoints
        self.health_endpoints = HealthEndpoints(
            self.config,
            self.health_monitor,
            self.performance_monitor,
            self.quality_monitor,
            self.metrics_collector
        )
        
        # State tracking
        self._running = False
        self._shutdown_event = threading.Event()
        
        # Setup alert callbacks
        self._setup_alert_callbacks()
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        self.logger.info("MonitoringOrchestrator initialized")
    
    def start(self):
        """Start all monitoring components."""
        if self._running:
            self.logger.warning("Monitoring already running")
            return
        
        try:
            self.logger.info("Starting monitoring system...")
            
            # Start metrics collection
            self.metrics_collector.start_collection()
            
            # Start health monitoring
            self.health_monitor.start_monitoring()
            
            # Start health endpoints server
            self.health_endpoints.start_server(
                port=self.config.dashboard.dashboard_port
            )
            
            self._running = True
            
            self.logger.info("Monitoring system started successfully")
            
            # Send startup notification
            self.alert_manager.send_alert(
                'monitoring_started',
                'Newsletter clustering monitoring system started',
                'info',
                {
                    'environment': self.config.environment,
                    'version': self.config.version,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring system: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Stop all monitoring components."""
        if not self._running:
            return
        
        self.logger.info("Stopping monitoring system...")
        
        try:
            # Send shutdown notification
            self.alert_manager.send_alert(
                'monitoring_stopping',
                'Newsletter clustering monitoring system stopping',
                'info',
                {'timestamp': datetime.now().isoformat()}
            )
            
            # Stop components in reverse order
            self.health_endpoints.stop_server()
            self.health_monitor.stop_monitoring()
            self.metrics_collector.stop_collection()
            
            self._running = False
            self._shutdown_event.set()
            
            self.logger.info("Monitoring system stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring system: {e}")
    
    def is_running(self) -> bool:
        """Check if monitoring is running."""
        return self._running
    
    def wait_for_shutdown(self):
        """Wait for shutdown signal."""
        self._shutdown_event.wait()
    
    def monitor_clustering_operation(self, clustering_result: Dict[str, Any]):
        """Monitor a clustering operation."""
        try:
            # Record clustering metrics
            self.metrics_collector.record_clustering_metrics(clustering_result)
            
            # Monitor quality
            self.quality_monitor.monitor_clustering_quality(clustering_result)
            
            # Monitor validation results
            if 'validation' in clustering_result:
                self.quality_monitor.monitor_validation_results(
                    clustering_result['validation']
                )
            
            # Monitor diversity metrics
            if 'validation' in clustering_result and 'diversity_metrics' in clustering_result['validation']:
                self.quality_monitor.monitor_diversity_metrics(
                    clustering_result['validation']['diversity_metrics']
                )
            
            # Monitor performance metrics
            if 'performance_metrics' in clustering_result:
                self.performance_monitor.monitor_performance_metrics(
                    clustering_result['performance_metrics']
                )
            
            self.logger.debug("Clustering operation monitoring completed")
            
        except Exception as e:
            self.logger.error(f"Error monitoring clustering operation: {e}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status."""
        try:
            health_summary = self.health_monitor.get_health_summary()
            performance_summary = self.performance_monitor.get_performance_summary()
            quality_summary = self.quality_monitor.get_quality_summary()
            alert_stats = self.alert_manager.get_alert_statistics()
            
            return {
                'monitoring_active': self._running,
                'timestamp': datetime.now().isoformat(),
                'health': health_summary,
                'performance': performance_summary,
                'quality': quality_summary,
                'alerts': alert_stats,
                'configuration': {
                    'environment': self.config.environment,
                    'service_name': self.config.service_name,
                    'version': self.config.version
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting monitoring status: {e}")
            return {
                'monitoring_active': self._running,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def register_dependency_check(self, name: str, check_func):
        """Register a dependency health check."""
        self.health_monitor.register_dependency_check(name, check_func)
    
    def set_quality_baseline(self, metric_name: str, baseline_value: float):
        """Set quality baseline for monitoring."""
        self.quality_monitor.set_quality_baseline(metric_name, baseline_value)
    
    def _setup_alert_callbacks(self):
        """Setup alert callbacks for monitoring components."""
        
        def performance_alert_handler(message: str, context: Dict[str, Any]):
            """Handle performance alerts."""
            severity = 'critical' if 'exceeded' in message.lower() else 'warning'
            self.alert_manager.send_alert(
                'performance_issue',
                message,
                severity,
                context
            )
        
        def quality_alert_handler(message: str, context: Dict[str, Any]):
            """Handle quality alerts."""
            severity = 'warning' if 'low' in message.lower() else 'info'
            self.alert_manager.send_alert(
                'quality_issue',
                message,
                severity,
                context
            )
        
        def health_alert_handler(message: str, context: Dict[str, Any]):
            """Handle health alerts."""
            severity = 'critical' if 'unhealthy' in message.lower() else 'warning'
            self.alert_manager.send_alert(
                'health_issue',
                message,
                severity,
                context
            )
        
        # Register callbacks
        self.performance_monitor.register_alert_callback(
            'performance_alert', performance_alert_handler
        )
        self.quality_monitor.register_alert_callback(
            'quality_alert', quality_alert_handler
        )
        self.health_monitor.register_alert_callback(
            'health_alert', health_alert_handler
        )
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            self.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


# Global monitoring instance
_monitoring_instance: Optional[MonitoringOrchestrator] = None


def get_monitoring_instance(config: Optional[MonitoringConfig] = None) -> MonitoringOrchestrator:
    """Get or create global monitoring instance."""
    global _monitoring_instance
    
    if _monitoring_instance is None:
        _monitoring_instance = MonitoringOrchestrator(config)
    
    return _monitoring_instance


def start_monitoring(config: Optional[MonitoringConfig] = None):
    """Start global monitoring system."""
    monitor = get_monitoring_instance(config)
    monitor.start()
    return monitor


def stop_monitoring():
    """Stop global monitoring system."""
    global _monitoring_instance
    
    if _monitoring_instance:
        _monitoring_instance.stop()
        _monitoring_instance = None


def monitor_clustering_operation(clustering_result: Dict[str, Any]):
    """Monitor a clustering operation using global instance."""
    monitor = get_monitoring_instance()
    monitor.monitor_clustering_operation(clustering_result)


def get_monitoring_status() -> Dict[str, Any]:
    """Get monitoring status using global instance."""
    monitor = get_monitoring_instance()
    return monitor.get_monitoring_status()


# Context manager for monitoring operations
class MonitoringContext:
    """Context manager for monitoring operations."""
    
    def __init__(self, operation_name: str, tags: Dict[str, str] = None):
        self.operation_name = operation_name
        self.tags = tags or {}
        self.monitor = get_monitoring_instance()
    
    def __enter__(self):
        return self.monitor.performance_monitor.monitor_operation(
            self.operation_name, self.tags
        ).__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.monitor.performance_monitor.monitor_operation(
            self.operation_name, self.tags
        ).__exit__(exc_type, exc_val, exc_tb)


def monitor_operation(operation_name: str, tags: Dict[str, str] = None):
    """Decorator/context manager for monitoring operations."""
    return MonitoringContext(operation_name, tags)