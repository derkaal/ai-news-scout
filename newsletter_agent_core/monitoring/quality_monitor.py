"""
Quality Monitoring System

Monitors clustering quality metrics, diversity analysis, and validation results.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

from .config import MonitoringConfig
from .metrics_collector import MetricsCollector


class QualityMonitor:
    """
    Quality monitoring system for clustering results.
    
    Tracks clustering quality metrics, diversity measures, validation results,
    and provides quality trend analysis with threshold-based alerting.
    """
    
    def __init__(self, config: MonitoringConfig, metrics_collector: MetricsCollector):
        self.config = config
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)
        
        # Quality tracking
        self._quality_history: List[Dict[str, Any]] = []
        self._quality_trends: Dict[str, List[float]] = defaultdict(list)
        self._alert_callbacks: Dict[str, Callable] = {}
        
        # Quality baselines
        self._quality_baselines: Dict[str, float] = {}
        
        self.logger.info("QualityMonitor initialized")
    
    def register_alert_callback(self, alert_type: str, callback: Callable):
        """Register callback for quality alerts."""
        self._alert_callbacks[alert_type] = callback
        self.logger.info(f"Registered quality alert callback for {alert_type}")
    
    def monitor_clustering_quality(self, clustering_result: Dict[str, Any]):
        """Monitor clustering quality metrics."""
        timestamp = datetime.now()
        
        # Extract quality metrics
        quality_metrics = self._extract_quality_metrics(clustering_result)
        
        # Record metrics
        self._record_quality_metrics(quality_metrics, timestamp)
        
        # Check thresholds
        self._check_quality_thresholds(quality_metrics)
        
        # Update trends
        self._update_quality_trends(quality_metrics)
        
        # Store in history
        self._quality_history.append({
            'timestamp': timestamp.isoformat(),
            'metrics': quality_metrics,
            'clustering_result': clustering_result
        })
        
        # Keep only recent history
        if len(self._quality_history) > 1000:
            self._quality_history = self._quality_history[-1000:]
        
        self.logger.debug(f"Recorded clustering quality metrics: {quality_metrics}")
    
    def monitor_diversity_metrics(self, diversity_data: Dict[str, Any]):
        """Monitor diversity-specific metrics."""
        # Source diversity
        if 'source_diversity' in diversity_data:
            source_diversity = diversity_data['source_diversity']
            if source_diversity is not None:
                self.metrics_collector.record_gauge(
                    'quality.diversity.source', source_diversity
                )
                
                # Check threshold
                min_diversity = self.config.quality_thresholds.min_source_diversity
                if source_diversity < min_diversity:
                    self._trigger_alert(
                        'source_diversity_low',
                        f"Source diversity ({source_diversity:.3f}) below threshold ({min_diversity:.3f})",
                        {'diversity': source_diversity, 'threshold': min_diversity}
                    )
        
        # Cluster size diversity
        if 'cluster_size_diversity' in diversity_data:
            size_diversity = diversity_data['cluster_size_diversity']
            if size_diversity is not None:
                self.metrics_collector.record_gauge(
                    'quality.diversity.cluster_size', size_diversity
                )
        
        # Temporal diversity
        if 'temporal_diversity' in diversity_data:
            temporal_diversity = diversity_data['temporal_diversity']
            if temporal_diversity is not None:
                self.metrics_collector.record_gauge(
                    'quality.diversity.temporal', temporal_diversity
                )
        
        # Topic diversity
        if 'topic_diversity' in diversity_data:
            topic_diversity = diversity_data['topic_diversity']
            if topic_diversity is not None:
                self.metrics_collector.record_gauge(
                    'quality.diversity.topic', topic_diversity
                )
    
    def monitor_validation_results(self, validation_result: Dict[str, Any]):
        """Monitor validation results and issues."""
        # Overall validation status
        is_valid = validation_result.get('is_valid', False)
        quality_score = validation_result.get('quality_score', 0.0)
        
        self.metrics_collector.record_gauge('quality.validation.is_valid', 1 if is_valid else 0)
        self.metrics_collector.record_gauge('quality.validation.score', quality_score)
        
        # Count issues and recommendations
        issues = validation_result.get('issues', [])
        recommendations = validation_result.get('recommendations', [])
        
        self.metrics_collector.record_gauge('quality.validation.issues_count', len(issues))
        self.metrics_collector.record_gauge('quality.validation.recommendations_count', len(recommendations))
        
        # Record specific issue types
        issue_types = defaultdict(int)
        for issue in issues:
            # Categorize issues based on content
            if 'silhouette' in issue.lower():
                issue_types['silhouette'] += 1
            elif 'coherence' in issue.lower():
                issue_types['coherence'] += 1
            elif 'diversity' in issue.lower():
                issue_types['diversity'] += 1
            elif 'noise' in issue.lower():
                issue_types['noise'] += 1
            else:
                issue_types['other'] += 1
        
        for issue_type, count in issue_types.items():
            self.metrics_collector.record_gauge(
                f'quality.validation.issues.{issue_type}', count
            )
        
        # Alert on validation failures
        if not is_valid:
            self._trigger_alert(
                'validation_failed',
                f"Clustering validation failed with {len(issues)} issues",
                {
                    'quality_score': quality_score,
                    'issues_count': len(issues),
                    'issues': issues[:3]  # First 3 issues
                }
            )
    
    def analyze_quality_trends(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Analyze quality trends over time."""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        # Filter recent quality data
        recent_data = [
            entry for entry in self._quality_history
            if datetime.fromisoformat(entry['timestamp']) >= cutoff_time
        ]
        
        if not recent_data:
            return {'error': 'No quality data available for analysis'}
        
        # Calculate trends
        trends = {}
        
        # Silhouette score trend
        silhouette_scores = [
            entry['metrics'].get('silhouette_score')
            for entry in recent_data
            if entry['metrics'].get('silhouette_score') is not None
        ]
        
        if silhouette_scores:
            trends['silhouette_score'] = {
                'current': silhouette_scores[-1],
                'average': np.mean(silhouette_scores),
                'trend': self._calculate_trend(silhouette_scores),
                'volatility': np.std(silhouette_scores) if len(silhouette_scores) > 1 else 0
            }
        
        # Quality score trend
        quality_scores = [
            entry['metrics'].get('overall_quality_score', 0)
            for entry in recent_data
        ]
        
        if quality_scores:
            trends['overall_quality'] = {
                'current': quality_scores[-1],
                'average': np.mean(quality_scores),
                'trend': self._calculate_trend(quality_scores),
                'volatility': np.std(quality_scores) if len(quality_scores) > 1 else 0
            }
        
        # Noise ratio trend
        noise_ratios = [
            entry['metrics'].get('noise_ratio', 0)
            for entry in recent_data
        ]
        
        if noise_ratios:
            trends['noise_ratio'] = {
                'current': noise_ratios[-1],
                'average': np.mean(noise_ratios),
                'trend': self._calculate_trend(noise_ratios),
                'volatility': np.std(noise_ratios) if len(noise_ratios) > 1 else 0
            }
        
        return {
            'time_window_hours': time_window_hours,
            'data_points': len(recent_data),
            'trends': trends,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get comprehensive quality summary."""
        if not self._quality_history:
            return {'error': 'No quality data available'}
        
        latest_entry = self._quality_history[-1]
        latest_metrics = latest_entry['metrics']
        
        # Calculate recent averages
        recent_entries = self._quality_history[-10:]  # Last 10 entries
        
        avg_metrics = {}
        for key in ['silhouette_score', 'coherence', 'overall_quality_score', 'noise_ratio']:
            values = [
                entry['metrics'].get(key)
                for entry in recent_entries
                if entry['metrics'].get(key) is not None
            ]
            if values:
                avg_metrics[f'avg_{key}'] = np.mean(values)
        
        return {
            'timestamp': latest_entry['timestamp'],
            'current_metrics': latest_metrics,
            'recent_averages': avg_metrics,
            'thresholds': {
                'min_silhouette_score': self.config.quality_thresholds.min_silhouette_score,
                'max_noise_ratio': self.config.quality_thresholds.max_noise_ratio,
                'min_cluster_coherence': self.config.quality_thresholds.min_cluster_coherence,
                'min_source_diversity': self.config.quality_thresholds.min_source_diversity,
                'min_overall_quality_score': self.config.quality_thresholds.min_overall_quality_score
            },
            'quality_history_count': len(self._quality_history)
        }
    
    def set_quality_baseline(self, metric_name: str, baseline_value: float):
        """Set quality baseline for comparison."""
        self._quality_baselines[metric_name] = baseline_value
        self.logger.info(f"Set quality baseline for {metric_name}: {baseline_value}")
    
    def _extract_quality_metrics(self, clustering_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract quality metrics from clustering result."""
        metrics = {}
        
        # Basic clustering metrics
        metrics['total_clusters'] = clustering_result.get('total_clusters', 0)
        metrics['total_items'] = clustering_result.get('total_items', 0)
        metrics['noise_items'] = clustering_result.get('noise_items', 0)
        
        # Calculate noise ratio
        if metrics['total_items'] > 0:
            metrics['noise_ratio'] = metrics['noise_items'] / metrics['total_items']
        else:
            metrics['noise_ratio'] = 0
        
        # Validation metrics
        if 'validation' in clustering_result:
            validation = clustering_result['validation']
            
            if 'validation_metrics' in validation:
                val_metrics = validation['validation_metrics']
                metrics['silhouette_score'] = val_metrics.get('silhouette_score')
                metrics['coherence'] = val_metrics.get('cluster_coherence')
                metrics['calinski_harabasz_score'] = val_metrics.get('calinski_harabasz_score')
                metrics['davies_bouldin_score'] = val_metrics.get('davies_bouldin_score')
            
            metrics['overall_quality_score'] = validation.get('quality_score', 0)
            metrics['is_valid'] = validation.get('is_valid', False)
        
        # Diversity metrics
        if 'validation' in clustering_result and 'diversity_metrics' in clustering_result['validation']:
            diversity = clustering_result['validation']['diversity_metrics']
            metrics['source_diversity'] = diversity.get('source_diversity')
            metrics['cluster_size_diversity'] = diversity.get('cluster_size_diversity')
            metrics['temporal_diversity'] = diversity.get('temporal_diversity')
            metrics['topic_diversity'] = diversity.get('topic_diversity')
        
        return metrics
    
    def _record_quality_metrics(self, metrics: Dict[str, Any], timestamp: datetime):
        """Record quality metrics to metrics collector."""
        for metric_name, value in metrics.items():
            if value is not None:
                if isinstance(value, bool):
                    value = 1 if value else 0
                
                if isinstance(value, (int, float)):
                    self.metrics_collector.record_gauge(f'quality.{metric_name}', value)
    
    def _check_quality_thresholds(self, metrics: Dict[str, Any]):
        """Check quality metrics against thresholds."""
        thresholds = self.config.quality_thresholds
        
        # Silhouette score
        silhouette = metrics.get('silhouette_score')
        if silhouette is not None and silhouette < thresholds.min_silhouette_score:
            self._trigger_alert(
                'silhouette_score_low',
                f"Silhouette score ({silhouette:.3f}) below threshold ({thresholds.min_silhouette_score:.3f})",
                {'score': silhouette, 'threshold': thresholds.min_silhouette_score}
            )
        
        # Noise ratio
        noise_ratio = metrics.get('noise_ratio', 0)
        if noise_ratio > thresholds.max_noise_ratio:
            self._trigger_alert(
                'noise_ratio_high',
                f"Noise ratio ({noise_ratio:.1%}) exceeds threshold ({thresholds.max_noise_ratio:.1%})",
                {'ratio': noise_ratio, 'threshold': thresholds.max_noise_ratio}
            )
        
        # Cluster coherence
        coherence = metrics.get('coherence')
        if coherence is not None and coherence < thresholds.min_cluster_coherence:
            self._trigger_alert(
                'coherence_low',
                f"Cluster coherence ({coherence:.3f}) below threshold ({thresholds.min_cluster_coherence:.3f})",
                {'coherence': coherence, 'threshold': thresholds.min_cluster_coherence}
            )
        
        # Overall quality score
        quality_score = metrics.get('overall_quality_score', 0)
        if quality_score < thresholds.min_overall_quality_score:
            self._trigger_alert(
                'quality_score_low',
                f"Overall quality score ({quality_score:.3f}) below threshold ({thresholds.min_overall_quality_score:.3f})",
                {'score': quality_score, 'threshold': thresholds.min_overall_quality_score}
            )
    
    def _update_quality_trends(self, metrics: Dict[str, Any]):
        """Update quality trend tracking."""
        for metric_name, value in metrics.items():
            if value is not None and isinstance(value, (int, float)):
                self._quality_trends[metric_name].append(value)
                
                # Keep only recent values
                if len(self._quality_trends[metric_name]) > 100:
                    self._quality_trends[metric_name] = self._quality_trends[metric_name][-100:]
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values."""
        if len(values) < 2:
            return 'stable'
        
        # Simple trend calculation using first and last values
        first_half = np.mean(values[:len(values)//2])
        second_half = np.mean(values[len(values)//2:])
        
        change_percent = ((second_half - first_half) / first_half) * 100 if first_half != 0 else 0
        
        if change_percent > 5:
            return 'improving'
        elif change_percent < -5:
            return 'declining'
        else:
            return 'stable'
    
    def _trigger_alert(self, alert_type: str, message: str, context: Dict[str, Any]):
        """Trigger quality alert."""
        self.logger.warning(f"Quality alert [{alert_type}]: {message}")
        
        # Record alert metric
        self.metrics_collector.record_counter(f'quality.alerts.{alert_type}')
        
        # Call registered callback if available
        if alert_type in self._alert_callbacks:
            try:
                self._alert_callbacks[alert_type](message, context)
            except Exception as e:
                self.logger.error(f"Error in quality alert callback for {alert_type}: {e}")