"""
Health Monitoring System

Monitors system health, uptime, dependencies, and service availability.
"""

import logging
import time
import threading
import requests
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import psutil

from .config import MonitoringConfig
from .metrics_collector import MetricsCollector


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class DependencyCheck:
    """Represents a dependency health check."""
    
    def __init__(self, name: str, check_func: Callable, timeout: int = 10):
        self.name = name
        self.check_func = check_func
        self.timeout = timeout
        self.last_check_time: Optional[datetime] = None
        self.last_status: HealthStatus = HealthStatus.UNKNOWN
        self.last_error: Optional[str] = None
        self.consecutive_failures = 0


class HealthMonitor:
    """
    System health monitoring service.
    
    Monitors service health, dependency status, uptime, and provides
    health check endpoints for load balancers and orchestrators.
    """
    
    def __init__(self, config: MonitoringConfig, metrics_collector: MetricsCollector):
        self.config = config
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)
        
        # Health tracking
        self._service_start_time = datetime.now()
        self._health_checks: Dict[str, DependencyCheck] = {}
        self._health_history: List[Dict[str, Any]] = []
        self._current_status = HealthStatus.HEALTHY
        
        # Monitoring state
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        
        # Alert callbacks
        self._alert_callbacks: Dict[str, Callable] = {}
        
        # Register default health checks
        self._register_default_checks()
        
        self.logger.info("HealthMonitor initialized")
    
    def start_monitoring(self):
        """Start health monitoring."""
        if self._monitoring_active:
            self.logger.warning("Health monitoring already active")
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        self.logger.info("Started health monitoring")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        self.logger.info("Stopped health monitoring")
    
    def register_dependency_check(self, name: str, check_func: Callable, timeout: int = 10):
        """Register a dependency health check."""
        self._health_checks[name] = DependencyCheck(name, check_func, timeout)
        self.logger.info(f"Registered dependency check: {name}")
    
    def register_alert_callback(self, alert_type: str, callback: Callable):
        """Register callback for health alerts."""
        self._alert_callbacks[alert_type] = callback
        self.logger.info(f"Registered health alert callback for {alert_type}")
    
    def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        check_time = datetime.now()
        
        # System health
        system_health = self._check_system_health()
        
        # Dependency health
        dependency_health = self._check_dependencies()
        
        # Service health
        service_health = self._check_service_health()
        
        # Determine overall status
        overall_status = self._determine_overall_status(
            system_health, dependency_health, service_health
        )
        
        health_report = {
            'timestamp': check_time.isoformat(),
            'status': overall_status.value,
            'uptime_seconds': (check_time - self._service_start_time).total_seconds(),
            'system': system_health,
            'dependencies': dependency_health,
            'service': service_health
        }
        
        # Record metrics
        self._record_health_metrics(health_report)
        
        # Store in history
        self._health_history.append(health_report)
        if len(self._health_history) > 1000:
            self._health_history = self._health_history[-1000:]
        
        # Update current status and check for alerts
        previous_status = self._current_status
        self._current_status = overall_status
        
        if previous_status != overall_status:
            self._handle_status_change(previous_status, overall_status, health_report)
        
        return health_report
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary for monitoring dashboards."""
        if not self._health_history:
            return self.check_health()
        
        latest_health = self._health_history[-1]
        
        # Calculate uptime percentage
        uptime_stats = self._calculate_uptime_stats()
        
        # Get recent issues
        recent_issues = self._get_recent_issues()
        
        return {
            'current_status': latest_health['status'],
            'uptime': uptime_stats,
            'last_check': latest_health['timestamp'],
            'recent_issues': recent_issues,
            'dependency_status': {
                name: check.last_status.value
                for name, check in self._health_checks.items()
            }
        }
    
    def is_healthy(self) -> bool:
        """Simple health check for load balancers."""
        return self._current_status == HealthStatus.HEALTHY
    
    def is_ready(self) -> bool:
        """Readiness check for orchestrators."""
        # Check if all critical dependencies are healthy
        critical_deps = ['embedding_service', 'clustering_engine']
        
        for dep_name in critical_deps:
            if dep_name in self._health_checks:
                if self._health_checks[dep_name].last_status != HealthStatus.HEALTHY:
                    return False
        
        return self._current_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
    
    def _register_default_checks(self):
        """Register default system health checks."""
        # Memory check
        def check_memory():
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                return HealthStatus.UNHEALTHY, f"Memory usage: {memory.percent:.1f}%"
            elif memory.percent > 80:
                return HealthStatus.DEGRADED, f"Memory usage: {memory.percent:.1f}%"
            return HealthStatus.HEALTHY, f"Memory usage: {memory.percent:.1f}%"
        
        self.register_dependency_check('system_memory', check_memory)
        
        # Disk check
        def check_disk():
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            if usage_percent > 95:
                return HealthStatus.UNHEALTHY, f"Disk usage: {usage_percent:.1f}%"
            elif usage_percent > 85:
                return HealthStatus.DEGRADED, f"Disk usage: {usage_percent:.1f}%"
            return HealthStatus.HEALTHY, f"Disk usage: {usage_percent:.1f}%"
        
        self.register_dependency_check('system_disk', check_disk)
        
        # CPU check
        def check_cpu():
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 95:
                return HealthStatus.UNHEALTHY, f"CPU usage: {cpu_percent:.1f}%"
            elif cpu_percent > 85:
                return HealthStatus.DEGRADED, f"CPU usage: {cpu_percent:.1f}%"
            return HealthStatus.HEALTHY, f"CPU usage: {cpu_percent:.1f}%"
        
        self.register_dependency_check('system_cpu', check_cpu)
    
    def _monitoring_loop(self):
        """Main health monitoring loop."""
        while self._monitoring_active:
            try:
                self.check_health()
                time.sleep(self.config.health_check.check_interval_seconds)
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(5)  # Brief pause before retrying
    
    def _check_system_health(self) -> Dict[str, Any]:
        """Check system-level health."""
        try:
            # Memory
            memory = psutil.virtual_memory()
            
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Disk
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Load average (if available)
            load_avg = None
            if hasattr(psutil, 'getloadavg'):
                load_avg = psutil.getloadavg()
            
            return {
                'status': 'healthy',
                'memory': {
                    'usage_percent': memory.percent,
                    'available_gb': memory.available / (1024**3)
                },
                'cpu': {
                    'usage_percent': cpu_percent,
                    'load_average': load_avg
                },
                'disk': {
                    'usage_percent': disk_percent,
                    'free_gb': disk.free / (1024**3)
                }
            }
        except Exception as e:
            self.logger.error(f"Error checking system health: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check all registered dependencies."""
        dependency_results = {}
        
        for name, check in self._health_checks.items():
            try:
                start_time = time.time()
                status, message = check.check_func()
                check_duration = time.time() - start_time
                
                check.last_check_time = datetime.now()
                check.last_status = status
                check.last_error = None if status == HealthStatus.HEALTHY else message
                
                if status == HealthStatus.HEALTHY:
                    check.consecutive_failures = 0
                else:
                    check.consecutive_failures += 1
                
                dependency_results[name] = {
                    'status': status.value,
                    'message': message,
                    'check_duration_seconds': check_duration,
                    'consecutive_failures': check.consecutive_failures,
                    'last_check': check.last_check_time.isoformat()
                }
                
                # Record dependency metrics
                self.metrics_collector.record_gauge(
                    f'health.dependency.{name}.status',
                    1 if status == HealthStatus.HEALTHY else 0
                )
                self.metrics_collector.record_timing(
                    f'health.dependency.{name}.check_duration',
                    check_duration
                )
                
            except Exception as e:
                check.last_error = str(e)
                check.consecutive_failures += 1
                check.last_check_time = datetime.now()
                check.last_status = HealthStatus.UNHEALTHY
                
                dependency_results[name] = {
                    'status': 'error',
                    'message': f'Check failed: {e}',
                    'consecutive_failures': check.consecutive_failures,
                    'last_check': check.last_check_time.isoformat()
                }
                
                self.logger.error(f"Dependency check failed for {name}: {e}")
        
        return dependency_results
    
    def _check_service_health(self) -> Dict[str, Any]:
        """Check service-specific health."""
        uptime = (datetime.now() - self._service_start_time).total_seconds()
        
        return {
            'status': 'healthy',
            'uptime_seconds': uptime,
            'start_time': self._service_start_time.isoformat(),
            'version': '1.0.0'  # Could be dynamic
        }
    
    def _determine_overall_status(self, system_health: Dict[str, Any],
                                dependency_health: Dict[str, Any],
                                service_health: Dict[str, Any]) -> HealthStatus:
        """Determine overall health status."""
        # Check for any unhealthy dependencies
        unhealthy_deps = [
            name for name, result in dependency_health.items()
            if result['status'] in ['unhealthy', 'error']
        ]
        
        if unhealthy_deps:
            return HealthStatus.UNHEALTHY
        
        # Check for degraded dependencies
        degraded_deps = [
            name for name, result in dependency_health.items()
            if result['status'] == 'degraded'
        ]
        
        if degraded_deps:
            return HealthStatus.DEGRADED
        
        # Check system health
        if system_health.get('status') == 'error':
            return HealthStatus.UNHEALTHY
        
        return HealthStatus.HEALTHY
    
    def _record_health_metrics(self, health_report: Dict[str, Any]):
        """Record health metrics."""
        # Overall status
        status_value = 1 if health_report['status'] == 'healthy' else 0
        self.metrics_collector.record_gauge('health.overall.status', status_value)
        
        # Uptime
        self.metrics_collector.record_gauge('health.uptime_seconds', health_report['uptime_seconds'])
        
        # System metrics
        if 'system' in health_report and health_report['system'].get('status') != 'error':
            system = health_report['system']
            if 'memory' in system:
                self.metrics_collector.record_gauge(
                    'health.system.memory_usage_percent',
                    system['memory']['usage_percent']
                )
            if 'cpu' in system:
                self.metrics_collector.record_gauge(
                    'health.system.cpu_usage_percent',
                    system['cpu']['usage_percent']
                )
            if 'disk' in system:
                self.metrics_collector.record_gauge(
                    'health.system.disk_usage_percent',
                    system['disk']['usage_percent']
                )
    
    def _handle_status_change(self, previous_status: HealthStatus,
                            new_status: HealthStatus, health_report: Dict[str, Any]):
        """Handle health status changes."""
        self.logger.info(f"Health status changed: {previous_status.value} -> {new_status.value}")
        
        # Trigger appropriate alerts
        if new_status == HealthStatus.UNHEALTHY:
            self._trigger_alert(
                'service_unhealthy',
                f"Service became unhealthy (was {previous_status.value})",
                health_report
            )
        elif new_status == HealthStatus.DEGRADED and previous_status == HealthStatus.HEALTHY:
            self._trigger_alert(
                'service_degraded',
                "Service performance degraded",
                health_report
            )
        elif new_status == HealthStatus.HEALTHY and previous_status != HealthStatus.HEALTHY:
            self._trigger_alert(
                'service_recovered',
                f"Service recovered (was {previous_status.value})",
                health_report
            )
    
    def _calculate_uptime_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Calculate uptime statistics."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_checks = [
            check for check in self._health_history
            if datetime.fromisoformat(check['timestamp']) >= cutoff_time
        ]
        
        if not recent_checks:
            return {'uptime_percentage': 100.0, 'total_checks': 0}
        
        healthy_checks = sum(1 for check in recent_checks if check['status'] == 'healthy')
        uptime_percentage = (healthy_checks / len(recent_checks)) * 100
        
        return {
            'uptime_percentage': uptime_percentage,
            'total_checks': len(recent_checks),
            'healthy_checks': healthy_checks,
            'time_window_hours': hours
        }
    
    def _get_recent_issues(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent health issues."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        issues = []
        for check in self._health_history:
            if datetime.fromisoformat(check['timestamp']) < cutoff_time:
                continue
            
            if check['status'] != 'healthy':
                issues.append({
                    'timestamp': check['timestamp'],
                    'status': check['status'],
                    'dependencies': [
                        name for name, dep in check.get('dependencies', {}).items()
                        if dep['status'] != 'healthy'
                    ]
                })
        
        return issues[-10:]  # Last 10 issues
    
    def _trigger_alert(self, alert_type: str, message: str, context: Dict[str, Any]):
        """Trigger health alert."""
        self.logger.warning(f"Health alert [{alert_type}]: {message}")
        
        # Record alert metric
        self.metrics_collector.record_counter(f'health.alerts.{alert_type}')
        
        # Call registered callback if available
        if alert_type in self._alert_callbacks:
            try:
                self._alert_callbacks[alert_type](message, context)
            except Exception as e:
                self.logger.error(f"Error in health alert callback for {alert_type}: {e}")