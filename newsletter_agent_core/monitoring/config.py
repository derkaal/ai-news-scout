"""
Monitoring Configuration

Centralized configuration for monitoring, alerting, and observability.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path


@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""
    collection_interval_seconds: int = 30
    retention_days: int = 30
    storage_path: str = field(
        default_factory=lambda: str(Path.home() / ".newsletter_agent" / "metrics")
    )
    enable_prometheus: bool = False
    prometheus_port: int = 8000
    enable_statsd: bool = False
    statsd_host: str = "localhost"
    statsd_port: int = 8125


@dataclass
class PerformanceThresholds:
    """Performance monitoring thresholds."""
    max_processing_time_seconds: int = 30
    max_memory_usage_gb: float = 2.0
    min_cache_hit_rate: float = 0.7
    max_error_rate: float = 0.05
    max_queue_size: int = 100
    min_throughput_items_per_minute: int = 10


@dataclass
class QualityThresholds:
    """Quality monitoring thresholds."""
    min_silhouette_score: float = 0.2
    max_noise_ratio: float = 0.3
    min_cluster_coherence: float = 0.4
    min_source_diversity: float = 0.2
    min_overall_quality_score: float = 0.3
    max_cluster_size_imbalance: float = 0.8


@dataclass
class HealthCheckConfig:
    """Health check configuration."""
    check_interval_seconds: int = 60
    timeout_seconds: int = 10
    max_consecutive_failures: int = 3
    enable_dependency_checks: bool = True
    endpoints: List[str] = field(default_factory=lambda: [
        "/health",
        "/health/ready",
        "/health/live"
    ])


@dataclass
class AlertingConfig:
    """Alerting configuration."""
    enable_email_alerts: bool = False
    email_recipients: List[str] = field(default_factory=list)
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    
    enable_slack_alerts: bool = False
    slack_webhook_url: str = ""
    slack_channel: str = "#alerts"
    
    enable_webhook_alerts: bool = False
    webhook_urls: List[str] = field(default_factory=list)
    
    alert_cooldown_minutes: int = 15
    severity_levels: List[str] = field(default_factory=lambda: [
        "critical", "warning", "info"
    ])


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    enable_web_dashboard: bool = True
    dashboard_port: int = 8080
    refresh_interval_seconds: int = 30
    max_data_points: int = 1000
    enable_real_time_updates: bool = True
    
    # Chart configurations
    performance_chart_window_minutes: int = 60
    quality_chart_window_minutes: int = 1440  # 24 hours
    error_chart_window_minutes: int = 60


@dataclass
class LoggingConfig:
    """Logging configuration for monitoring."""
    log_level: str = "INFO"
    log_format: str = (
        "%(asctime)s - %(name)s - %(levelname)s - "
        "%(message)s [%(filename)s:%(lineno)d]"
    )
    log_file: str = field(
        default_factory=lambda: str(
            Path.home() / ".newsletter_agent" / "logs" / "monitoring.log"
        )
    )
    max_log_size_mb: int = 100
    backup_count: int = 5
    enable_structured_logging: bool = True


@dataclass
class MonitoringConfig:
    """Main monitoring configuration."""
    
    # Sub-configurations
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    performance_thresholds: PerformanceThresholds = field(
        default_factory=PerformanceThresholds
    )
    quality_thresholds: QualityThresholds = field(
        default_factory=QualityThresholds
    )
    health_check: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    alerting: AlertingConfig = field(default_factory=AlertingConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # General settings
    enable_monitoring: bool = True
    environment: str = "production"
    service_name: str = "newsletter-clustering-agent"
    version: str = "1.0.0"
    
    def __post_init__(self):
        """Post-initialization setup."""
        self._apply_environment_overrides()
        self._setup_directories()
        self._validate_config()
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides."""
        # Performance thresholds
        if os.getenv("MONITORING_MAX_PROCESSING_TIME"):
            self.performance_thresholds.max_processing_time_seconds = int(
                os.getenv("MONITORING_MAX_PROCESSING_TIME")
            )
        
        if os.getenv("MONITORING_MAX_MEMORY_GB"):
            self.performance_thresholds.max_memory_usage_gb = float(
                os.getenv("MONITORING_MAX_MEMORY_GB")
            )
        
        # Quality thresholds
        if os.getenv("MONITORING_MIN_SILHOUETTE_SCORE"):
            self.quality_thresholds.min_silhouette_score = float(
                os.getenv("MONITORING_MIN_SILHOUETTE_SCORE")
            )
        
        # Alerting
        if os.getenv("MONITORING_ENABLE_EMAIL_ALERTS"):
            self.alerting.enable_email_alerts = (
                os.getenv("MONITORING_ENABLE_EMAIL_ALERTS").lower() == "true"
            )
        
        if os.getenv("MONITORING_EMAIL_RECIPIENTS"):
            self.alerting.email_recipients = [
                email.strip() for email in 
                os.getenv("MONITORING_EMAIL_RECIPIENTS").split(",")
            ]
        
        if os.getenv("MONITORING_SLACK_WEBHOOK_URL"):
            self.alerting.slack_webhook_url = os.getenv(
                "MONITORING_SLACK_WEBHOOK_URL"
            )
            self.alerting.enable_slack_alerts = True
        
        # Dashboard
        if os.getenv("MONITORING_DASHBOARD_PORT"):
            self.dashboard.dashboard_port = int(
                os.getenv("MONITORING_DASHBOARD_PORT")
            )
        
        # Environment
        if os.getenv("MONITORING_ENVIRONMENT"):
            self.environment = os.getenv("MONITORING_ENVIRONMENT")
    
    def _setup_directories(self):
        """Create necessary directories."""
        directories = [
            self.metrics.storage_path,
            os.path.dirname(self.logging.log_file)
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _validate_config(self):
        """Validate configuration values."""
        if self.performance_thresholds.max_processing_time_seconds <= 0:
            raise ValueError("max_processing_time_seconds must be positive")
        
        if self.performance_thresholds.max_memory_usage_gb <= 0:
            raise ValueError("max_memory_usage_gb must be positive")
        
        if not 0 <= self.performance_thresholds.min_cache_hit_rate <= 1:
            raise ValueError("min_cache_hit_rate must be between 0 and 1")
        
        if not 0 <= self.quality_thresholds.min_silhouette_score <= 1:
            raise ValueError("min_silhouette_score must be between 0 and 1")
        
        if self.alerting.enable_email_alerts and not self.alerting.email_recipients:
            raise ValueError(
                "email_recipients required when email alerts are enabled"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "metrics": self.metrics.__dict__,
            "performance_thresholds": self.performance_thresholds.__dict__,
            "quality_thresholds": self.quality_thresholds.__dict__,
            "health_check": self.health_check.__dict__,
            "alerting": self.alerting.__dict__,
            "dashboard": self.dashboard.__dict__,
            "logging": self.logging.__dict__,
            "enable_monitoring": self.enable_monitoring,
            "environment": self.environment,
            "service_name": self.service_name,
            "version": self.version
        }


def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration with environment overrides."""
    return MonitoringConfig()


def create_development_config() -> MonitoringConfig:
    """Create monitoring configuration optimized for development."""
    config = MonitoringConfig()
    config.environment = "development"
    config.metrics.collection_interval_seconds = 10
    config.health_check.check_interval_seconds = 30
    config.dashboard.refresh_interval_seconds = 10
    config.logging.log_level = "DEBUG"
    return config


def create_production_config() -> MonitoringConfig:
    """Create monitoring configuration optimized for production."""
    config = MonitoringConfig()
    config.environment = "production"
    config.metrics.collection_interval_seconds = 30
    config.health_check.check_interval_seconds = 60
    config.dashboard.refresh_interval_seconds = 30
    config.logging.log_level = "INFO"
    config.alerting.enable_email_alerts = True
    return config