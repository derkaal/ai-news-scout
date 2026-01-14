"""
Monitoring Dashboard Configuration

Dashboard templates and configuration for monitoring visualization.
"""

import json
from typing import Dict, Any, List
from datetime import datetime, timedelta


class DashboardConfig:
    """
    Dashboard configuration and templates.
    
    Provides configuration for monitoring dashboards including
    Grafana templates, chart configurations, and alert panels.
    """
    
    def __init__(self):
        self.dashboard_templates = {
            'grafana': self._create_grafana_dashboard(),
            'simple_html': self._create_simple_html_dashboard()
        }
    
    def get_grafana_dashboard(self) -> Dict[str, Any]:
        """Get Grafana dashboard configuration."""
        return self.dashboard_templates['grafana']
    
    def get_simple_dashboard(self) -> str:
        """Get simple HTML dashboard."""
        return self.dashboard_templates['simple_html']
    
    def _create_grafana_dashboard(self) -> Dict[str, Any]:
        """Create Grafana dashboard configuration."""
        return {
            "dashboard": {
                "id": None,
                "title": "Newsletter Clustering Agent - Monitoring",
                "tags": ["newsletter", "clustering", "monitoring"],
                "timezone": "browser",
                "panels": [
                    # System Health Panel
                    {
                        "id": 1,
                        "title": "System Health Status",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "health_overall_status",
                                "legendFormat": "Health Status"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {
                                    "mode": "thresholds"
                                },
                                "thresholds": {
                                    "steps": [
                                        {"color": "red", "value": 0},
                                        {"color": "green", "value": 1}
                                    ]
                                }
                            }
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    
                    # Processing Time Panel
                    {
                        "id": 2,
                        "title": "Clustering Processing Time",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "clustering_processing_duration",
                                "legendFormat": "Processing Time (s)"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Seconds",
                                "min": 0
                            }
                        ],
                        "alert": {
                            "conditions": [
                                {
                                    "query": {"params": ["A", "5m", "now"]},
                                    "reducer": {"params": [], "type": "avg"},
                                    "evaluator": {"params": [30], "type": "gt"}
                                }
                            ],
                            "executionErrorState": "alerting",
                            "for": "5m",
                            "frequency": "10s",
                            "handler": 1,
                            "name": "High Processing Time",
                            "noDataState": "no_data"
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },
                    
                    # Memory Usage Panel
                    {
                        "id": 3,
                        "title": "Memory Usage",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "system_memory_peak_gb",
                                "legendFormat": "Peak Memory (GB)"
                            },
                            {
                                "expr": "system_memory_rss_mb / 1024",
                                "legendFormat": "Current Memory (GB)"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "GB",
                                "min": 0,
                                "max": 4
                            }
                        ],
                        "alert": {
                            "conditions": [
                                {
                                    "query": {"params": ["A", "5m", "now"]},
                                    "reducer": {"params": [], "type": "avg"},
                                    "evaluator": {"params": [2], "type": "gt"}
                                }
                            ],
                            "name": "High Memory Usage"
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
                    },
                    
                    # Quality Metrics Panel
                    {
                        "id": 4,
                        "title": "Clustering Quality Metrics",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "quality_silhouette_score",
                                "legendFormat": "Silhouette Score"
                            },
                            {
                                "expr": "quality_overall_quality_score",
                                "legendFormat": "Overall Quality"
                            },
                            {
                                "expr": "quality_coherence",
                                "legendFormat": "Coherence"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Score",
                                "min": 0,
                                "max": 1
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
                    },
                    
                    # Noise Ratio Panel
                    {
                        "id": 5,
                        "title": "Noise Ratio",
                        "type": "singlestat",
                        "targets": [
                            {
                                "expr": "quality_noise_ratio",
                                "legendFormat": "Noise Ratio"
                            }
                        ],
                        "valueName": "current",
                        "format": "percentunit",
                        "thresholds": "0.2,0.3",
                        "colorBackground": True,
                        "colors": ["green", "yellow", "red"],
                        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 16}
                    },
                    
                    # Cache Hit Rate Panel
                    {
                        "id": 6,
                        "title": "Cache Hit Rate",
                        "type": "singlestat",
                        "targets": [
                            {
                                "expr": "business_cache_hit_rate",
                                "legendFormat": "Hit Rate"
                            }
                        ],
                        "valueName": "current",
                        "format": "percentunit",
                        "thresholds": "0.5,0.7",
                        "colorBackground": True,
                        "colors": ["red", "yellow", "green"],
                        "gridPos": {"h": 4, "w": 6, "x": 6, "y": 16}
                    },
                    
                    # Throughput Panel
                    {
                        "id": 7,
                        "title": "Processing Throughput",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "business_throughput_items_per_second",
                                "legendFormat": "Items/Second"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Items/Second",
                                "min": 0
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16}
                    },
                    
                    # Alert Summary Panel
                    {
                        "id": 8,
                        "title": "Active Alerts",
                        "type": "table",
                        "targets": [
                            {
                                "expr": "alerts_total",
                                "legendFormat": "Alert Count"
                            }
                        ],
                        "columns": [
                            {"text": "Alert Type", "value": "alert_type"},
                            {"text": "Count", "value": "count"},
                            {"text": "Severity", "value": "severity"}
                        ],
                        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 24}
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "timepicker": {
                    "refresh_intervals": ["5s", "10s", "30s", "1m", "5m", "15m"],
                    "time_options": ["5m", "15m", "1h", "6h", "12h", "24h", "2d", "7d"]
                },
                "refresh": "30s"
            }
        }
    
    def _create_simple_html_dashboard(self) -> str:
        """Create simple HTML dashboard template."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Newsletter Clustering Agent - Monitoring</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            margin: -20px -20px 20px -20px;
            text-align: center;
        }
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card h3 {
            margin-top: 0;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 8px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }
        .metric-value {
            font-weight: bold;
        }
        .status-healthy { color: #27ae60; }
        .status-warning { color: #f39c12; }
        .status-critical { color: #e74c3c; }
        .refresh-info {
            text-align: center;
            color: #7f8c8d;
            margin-top: 20px;
        }
        .alert {
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
            border-left: 4px solid;
        }
        .alert-info { border-color: #3498db; background-color: #d6eaf8; }
        .alert-warning { border-color: #f39c12; background-color: #fdeaa7; }
        .alert-critical { border-color: #e74c3c; background-color: #fadbd8; }
    </style>
    <script>
        function refreshData() {
            fetch('/status')
                .then(response => response.json())
                .then(data => updateDashboard(data))
                .catch(error => console.error('Error:', error));
        }
        
        function updateDashboard(data) {
            // Update system status
            document.getElementById('system-status').textContent = data.overall_status;
            document.getElementById('system-status').className = 
                'metric-value status-' + (data.overall_status === 'healthy' ? 'healthy' : 'warning');
            
            // Update performance metrics
            document.getElementById('memory-usage').textContent = 
                (data.performance.memory_usage_mb / 1024).toFixed(2) + ' GB';
            document.getElementById('cpu-usage').textContent = 
                data.performance.cpu_usage_percent.toFixed(1) + '%';
            
            // Update quality metrics
            document.getElementById('quality-score').textContent = 
                (data.quality.overall_score * 100).toFixed(1) + '%';
            document.getElementById('noise-ratio').textContent = 
                (data.quality.noise_ratio * 100).toFixed(1) + '%';
            
            // Update timestamp
            document.getElementById('last-update').textContent = 
                new Date(data.timestamp).toLocaleString();
        }
        
        // Auto-refresh every 30 seconds
        setInterval(refreshData, 30000);
        
        // Initial load
        window.onload = refreshData;
    </script>
</head>
<body>
    <div class="header">
        <h1>Newsletter Clustering Agent</h1>
        <p>Real-time Monitoring Dashboard</p>
    </div>
    
    <div class="dashboard">
        <div class="card">
            <h3>System Health</h3>
            <div class="metric">
                <span>Overall Status:</span>
                <span id="system-status" class="metric-value">Loading...</span>
            </div>
            <div class="metric">
                <span>Memory Usage:</span>
                <span id="memory-usage" class="metric-value">Loading...</span>
            </div>
            <div class="metric">
                <span>CPU Usage:</span>
                <span id="cpu-usage" class="metric-value">Loading...</span>
            </div>
        </div>
        
        <div class="card">
            <h3>Quality Metrics</h3>
            <div class="metric">
                <span>Overall Quality:</span>
                <span id="quality-score" class="metric-value">Loading...</span>
            </div>
            <div class="metric">
                <span>Noise Ratio:</span>
                <span id="noise-ratio" class="metric-value">Loading...</span>
            </div>
            <div class="metric">
                <span>Silhouette Score:</span>
                <span id="silhouette-score" class="metric-value">Loading...</span>
            </div>
        </div>
        
        <div class="card">
            <h3>Performance</h3>
            <div class="metric">
                <span>Avg Processing Time:</span>
                <span id="processing-time" class="metric-value">Loading...</span>
            </div>
            <div class="metric">
                <span>Cache Hit Rate:</span>
                <span id="cache-hit-rate" class="metric-value">Loading...</span>
            </div>
            <div class="metric">
                <span>Throughput:</span>
                <span id="throughput" class="metric-value">Loading...</span>
            </div>
        </div>
        
        <div class="card">
            <h3>Recent Alerts</h3>
            <div id="alerts-container">
                <div class="alert alert-info">No recent alerts</div>
            </div>
        </div>
    </div>
    
    <div class="refresh-info">
        <p>Last updated: <span id="last-update">Never</span></p>
        <p>Auto-refresh every 30 seconds</p>
    </div>
</body>
</html>
        """.strip()


def create_performance_baselines() -> Dict[str, float]:
    """Create performance baselines for monitoring."""
    return {
        # Processing time baselines (seconds)
        'clustering_processing_time_p50': 15.0,
        'clustering_processing_time_p95': 25.0,
        'clustering_processing_time_p99': 30.0,
        
        # Memory usage baselines (GB)
        'memory_usage_baseline': 0.5,
        'memory_usage_warning': 1.5,
        'memory_usage_critical': 2.0,
        
        # Quality baselines
        'silhouette_score_baseline': 0.3,
        'silhouette_score_good': 0.5,
        'silhouette_score_excellent': 0.7,
        
        'noise_ratio_baseline': 0.2,
        'noise_ratio_warning': 0.25,
        'noise_ratio_critical': 0.3,
        
        'coherence_baseline': 0.4,
        'coherence_good': 0.6,
        'coherence_excellent': 0.8,
        
        # Cache performance baselines
        'cache_hit_rate_baseline': 0.7,
        'cache_hit_rate_good': 0.8,
        'cache_hit_rate_excellent': 0.9,
        
        # Throughput baselines (items per minute)
        'throughput_baseline': 10.0,
        'throughput_good': 20.0,
        'throughput_excellent': 40.0,
        
        # System resource baselines
        'cpu_usage_baseline': 50.0,
        'cpu_usage_warning': 80.0,
        'cpu_usage_critical': 95.0,
        
        'disk_usage_baseline': 70.0,
        'disk_usage_warning': 85.0,
        'disk_usage_critical': 95.0
    }


def create_sla_definitions() -> Dict[str, Any]:
    """Create SLA definitions for the clustering service."""
    return {
        'availability': {
            'target': 99.5,  # 99.5% uptime
            'measurement_window': '30d',
            'description': 'Service availability percentage over 30 days'
        },
        'performance': {
            'processing_time_p95': {
                'target': 30.0,  # seconds
                'measurement_window': '24h',
                'description': '95th percentile processing time under 30 seconds'
            },
            'memory_usage': {
                'target': 2.0,  # GB
                'measurement_window': '24h',
                'description': 'Memory usage stays under 2GB'
            }
        },
        'quality': {
            'silhouette_score': {
                'target': 0.2,  # minimum acceptable
                'measurement_window': '7d',
                'description': 'Average silhouette score above 0.2'
            },
            'noise_ratio': {
                'target': 0.3,  # maximum acceptable
                'measurement_window': '7d',
                'description': 'Average noise ratio below 30%'
            }
        },
        'business': {
            'success_rate': {
                'target': 95.0,  # percentage
                'measurement_window': '24h',
                'description': 'Clustering success rate above 95%'
            },
            'cache_efficiency': {
                'target': 70.0,  # percentage
                'measurement_window': '24h',
                'description': 'Cache hit rate above 70%'
            }
        }
    }