"""
Health Check Endpoints

HTTP endpoints for health checks, readiness probes, and liveness probes.
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
from flask import Flask, jsonify, request
import threading

from .health_monitor import HealthMonitor, HealthStatus
from .performance_monitor import PerformanceMonitor
from .quality_monitor import QualityMonitor
from .metrics_collector import MetricsCollector
from .config import MonitoringConfig


class HealthEndpoints:
    """
    HTTP endpoints for health checks and monitoring.
    
    Provides standard endpoints for load balancers, orchestrators,
    and monitoring systems.
    """
    
    def __init__(self, config: MonitoringConfig,
                 health_monitor: HealthMonitor,
                 performance_monitor: PerformanceMonitor,
                 quality_monitor: QualityMonitor,
                 metrics_collector: MetricsCollector):
        self.config = config
        self.health_monitor = health_monitor
        self.performance_monitor = performance_monitor
        self.quality_monitor = quality_monitor
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)
        
        # Flask app for endpoints
        self.app = Flask(__name__)
        self.app.config['JSON_SORT_KEYS'] = False
        
        # Register routes
        self._register_routes()
        
        # Server state
        self._server_thread: Optional[threading.Thread] = None
        self._server_running = False
        
        self.logger.info("HealthEndpoints initialized")
    
    def start_server(self, host: str = '0.0.0.0', port: int = 8080):
        """Start the health check server."""
        if self._server_running:
            self.logger.warning("Health check server already running")
            return
        
        def run_server():
            self.app.run(
                host=host,
                port=port,
                debug=False,
                use_reloader=False,
                threaded=True
            )
        
        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        self._server_running = True
        
        self.logger.info(f"Health check server started on {host}:{port}")
    
    def stop_server(self):
        """Stop the health check server."""
        self._server_running = False
        self.logger.info("Health check server stopped")
    
    def _register_routes(self):
        """Register all health check routes."""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Basic health check endpoint."""
            try:
                health_report = self.health_monitor.check_health()
                
                status_code = 200
                if health_report['status'] == HealthStatus.UNHEALTHY.value:
                    status_code = 503
                elif health_report['status'] == HealthStatus.DEGRADED.value:
                    status_code = 200  # Still serving traffic
                
                return jsonify({
                    'status': health_report['status'],
                    'timestamp': health_report['timestamp'],
                    'uptime_seconds': health_report['uptime_seconds'],
                    'service': 'newsletter-clustering-agent',
                    'version': '1.0.0'
                }), status_code
                
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
                return jsonify({
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/health/ready', methods=['GET'])
        def readiness_check():
            """Kubernetes readiness probe endpoint."""
            try:
                is_ready = self.health_monitor.is_ready()
                
                if is_ready:
                    return jsonify({
                        'ready': True,
                        'timestamp': datetime.now().isoformat(),
                        'message': 'Service is ready to accept traffic'
                    }), 200
                else:
                    return jsonify({
                        'ready': False,
                        'timestamp': datetime.now().isoformat(),
                        'message': 'Service is not ready'
                    }), 503
                    
            except Exception as e:
                self.logger.error(f"Readiness check failed: {e}")
                return jsonify({
                    'ready': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/health/live', methods=['GET'])
        def liveness_check():
            """Kubernetes liveness probe endpoint."""
            try:
                is_healthy = self.health_monitor.is_healthy()
                
                if is_healthy:
                    return jsonify({
                        'alive': True,
                        'timestamp': datetime.now().isoformat(),
                        'message': 'Service is alive'
                    }), 200
                else:
                    return jsonify({
                        'alive': False,
                        'timestamp': datetime.now().isoformat(),
                        'message': 'Service is not healthy'
                    }), 503
                    
            except Exception as e:
                self.logger.error(f"Liveness check failed: {e}")
                return jsonify({
                    'alive': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/health/detailed', methods=['GET'])
        def detailed_health():
            """Detailed health information."""
            try:
                health_report = self.health_monitor.check_health()
                performance_summary = self.performance_monitor.get_performance_summary()
                quality_summary = self.quality_monitor.get_quality_summary()
                
                return jsonify({
                    'health': health_report,
                    'performance': performance_summary,
                    'quality': quality_summary,
                    'timestamp': datetime.now().isoformat()
                }), 200
                
            except Exception as e:
                self.logger.error(f"Detailed health check failed: {e}")
                return jsonify({
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/metrics', methods=['GET'])
        def metrics_endpoint():
            """Prometheus-compatible metrics endpoint."""
            try:
                format_type = request.args.get('format', 'prometheus')
                metrics_data = self.metrics_collector.export_metrics(format_type)
                
                if format_type.lower() == 'json':
                    return jsonify(json.loads(metrics_data)), 200
                else:
                    return metrics_data, 200, {'Content-Type': 'text/plain'}
                    
            except Exception as e:
                self.logger.error(f"Metrics export failed: {e}")
                return jsonify({
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/metrics/summary', methods=['GET'])
        def metrics_summary():
            """Metrics summary endpoint."""
            try:
                time_window = int(request.args.get('window_minutes', 60))
                summary = self.metrics_collector.get_metrics_summary(time_window)
                
                return jsonify(summary), 200
                
            except Exception as e:
                self.logger.error(f"Metrics summary failed: {e}")
                return jsonify({
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/performance', methods=['GET'])
        def performance_status():
            """Performance monitoring endpoint."""
            try:
                performance_summary = self.performance_monitor.get_performance_summary()
                system_health = self.performance_monitor.check_system_health()
                
                return jsonify({
                    'performance': performance_summary,
                    'system_health': system_health,
                    'timestamp': datetime.now().isoformat()
                }), 200
                
            except Exception as e:
                self.logger.error(f"Performance status failed: {e}")
                return jsonify({
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/quality', methods=['GET'])
        def quality_status():
            """Quality monitoring endpoint."""
            try:
                quality_summary = self.quality_monitor.get_quality_summary()
                
                # Get quality trends if requested
                include_trends = request.args.get('trends', 'false').lower() == 'true'
                if include_trends:
                    trends_window = int(request.args.get('trends_window', 24))
                    quality_trends = self.quality_monitor.analyze_quality_trends(trends_window)
                    quality_summary['trends'] = quality_trends
                
                return jsonify(quality_summary), 200
                
            except Exception as e:
                self.logger.error(f"Quality status failed: {e}")
                return jsonify({
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/status', methods=['GET'])
        def overall_status():
            """Overall system status endpoint."""
            try:
                health_summary = self.health_monitor.get_health_summary()
                performance_summary = self.performance_monitor.get_performance_summary()
                quality_summary = self.quality_monitor.get_quality_summary()
                
                # Determine overall status
                overall_healthy = (
                    health_summary['current_status'] in ['healthy', 'degraded'] and
                    performance_summary.get('memory', {}).get('current_mb', 0) < 
                    (self.config.performance_thresholds.max_memory_usage_gb * 1024) and
                    quality_summary.get('current_metrics', {}).get('overall_quality_score', 0) > 0.3
                )
                
                return jsonify({
                    'overall_status': 'healthy' if overall_healthy else 'unhealthy',
                    'health': health_summary,
                    'performance': {
                        'memory_usage_mb': performance_summary.get('memory', {}).get('current_mb', 0),
                        'cpu_usage_percent': performance_summary.get('cpu', {}).get('usage_percent', 0),
                        'active_operations': performance_summary.get('active_operations', 0)
                    },
                    'quality': {
                        'overall_score': quality_summary.get('current_metrics', {}).get('overall_quality_score', 0),
                        'silhouette_score': quality_summary.get('current_metrics', {}).get('silhouette_score'),
                        'noise_ratio': quality_summary.get('current_metrics', {}).get('noise_ratio', 0)
                    },
                    'timestamp': datetime.now().isoformat()
                }), 200
                
            except Exception as e:
                self.logger.error(f"Overall status failed: {e}")
                return jsonify({
                    'overall_status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/config', methods=['GET'])
        def config_info():
            """Configuration information endpoint."""
            try:
                return jsonify({
                    'service': 'newsletter-clustering-agent',
                    'version': '1.0.0',
                    'environment': self.config.environment,
                    'monitoring_config': {
                        'collection_interval_seconds': self.config.metrics.collection_interval_seconds,
                        'health_check_interval_seconds': self.config.health_check.check_interval_seconds,
                        'performance_thresholds': {
                            'max_processing_time_seconds': self.config.performance_thresholds.max_processing_time_seconds,
                            'max_memory_usage_gb': self.config.performance_thresholds.max_memory_usage_gb,
                            'min_cache_hit_rate': self.config.performance_thresholds.min_cache_hit_rate
                        },
                        'quality_thresholds': {
                            'min_silhouette_score': self.config.quality_thresholds.min_silhouette_score,
                            'max_noise_ratio': self.config.quality_thresholds.max_noise_ratio,
                            'min_cluster_coherence': self.config.quality_thresholds.min_cluster_coherence
                        }
                    },
                    'timestamp': datetime.now().isoformat()
                }), 200
                
            except Exception as e:
                self.logger.error(f"Config info failed: {e}")
                return jsonify({
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500


def create_health_endpoints(config: MonitoringConfig,
                          health_monitor: HealthMonitor,
                          performance_monitor: PerformanceMonitor,
                          quality_monitor: QualityMonitor,
                          metrics_collector: MetricsCollector) -> HealthEndpoints:
    """Create health endpoints instance."""
    return HealthEndpoints(
        config, health_monitor, performance_monitor,
        quality_monitor, metrics_collector
    )