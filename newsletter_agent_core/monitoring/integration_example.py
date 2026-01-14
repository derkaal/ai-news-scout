"""
Monitoring Integration Example

Example of how to integrate monitoring into the newsletter clustering agent.
"""

import logging
from typing import Dict, Any
from contextlib import contextmanager

from .monitor import (
    get_monitoring_instance, 
    start_monitoring, 
    monitor_clustering_operation,
    monitor_operation
)
from .config import create_production_config
from ..clustering.orchestrator import ClusteringOrchestrator


class MonitoredNewsletterAgent:
    """
    Newsletter agent with integrated monitoring.
    
    Example implementation showing how to integrate comprehensive
    monitoring into the newsletter clustering workflow.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize monitoring
        monitoring_config = create_production_config()
        self.monitoring = get_monitoring_instance(monitoring_config)
        
        # Initialize clustering
        self.clustering_orchestrator = ClusteringOrchestrator()
        
        # Register dependency checks
        self._register_dependency_checks()
        
        # Set quality baselines
        self._set_quality_baselines()
        
        self.logger.info("MonitoredNewsletterAgent initialized")
    
    def start(self):
        """Start the monitored newsletter agent."""
        try:
            # Start monitoring first
            self.monitoring.start()
            
            self.logger.info("Newsletter agent started with monitoring")
            
        except Exception as e:
            self.logger.error(f"Failed to start newsletter agent: {e}")
            raise
    
    def stop(self):
        """Stop the monitored newsletter agent."""
        try:
            self.monitoring.stop()
            self.logger.info("Newsletter agent stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping newsletter agent: {e}")
    
    def process_newsletters(self, items: list) -> Dict[str, Any]:
        """
        Process newsletters with comprehensive monitoring.
        
        Args:
            items: List of newsletter items to process
            
        Returns:
            Dictionary with clustering results and monitoring data
        """
        with monitor_operation('newsletter_processing', {'batch_size': len(items)}):
            try:
                # Monitor the clustering operation
                with monitor_operation('clustering_operation'):
                    clustering_result = self.clustering_orchestrator.cluster_items(
                        items=items,
                        text_field="short_description",
                        validate_results=True
                    )
                
                # Record clustering metrics
                monitor_clustering_operation(clustering_result)
                
                # Record business metrics
                self._record_business_metrics(clustering_result, len(items))
                
                # Check for quality issues
                self._check_quality_issues(clustering_result)
                
                self.logger.info(
                    f"Successfully processed {len(items)} newsletters, "
                    f"created {clustering_result.get('total_clusters', 0)} clusters"
                )
                
                return clustering_result
                
            except Exception as e:
                self.logger.error(f"Error processing newsletters: {e}")
                
                # Record failure metrics
                self.monitoring.metrics_collector.record_counter(
                    'business.processing.failure'
                )
                
                raise
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status."""
        return self.monitoring.get_monitoring_status()
    
    def _register_dependency_checks(self):
        """Register dependency health checks."""
        
        def check_clustering_engine():
            """Check clustering engine health."""
            try:
                # Test with minimal data
                test_items = [
                    {"short_description": "Test item 1"},
                    {"short_description": "Test item 2"}
                ]
                
                result = self.clustering_orchestrator.cluster_items(
                    test_items, validate_results=False
                )
                
                if result and result.get('total_items', 0) > 0:
                    return 'healthy', 'Clustering engine operational'
                else:
                    return 'unhealthy', 'Clustering engine not responding'
                    
            except Exception as e:
                return 'unhealthy', f'Clustering engine error: {str(e)}'
        
        def check_embedding_service():
            """Check embedding service health."""
            try:
                embedding_service = self.clustering_orchestrator.embedding_service
                
                # Test embedding generation
                test_texts = ["Test embedding generation"]
                embeddings = embedding_service.generate_embeddings(test_texts)
                
                if len(embeddings) > 0:
                    return 'healthy', 'Embedding service operational'
                else:
                    return 'unhealthy', 'Embedding service not generating embeddings'
                    
            except Exception as e:
                return 'unhealthy', f'Embedding service error: {str(e)}'
        
        # Register checks
        self.monitoring.register_dependency_check(
            'clustering_engine', check_clustering_engine
        )
        self.monitoring.register_dependency_check(
            'embedding_service', check_embedding_service
        )
    
    def _set_quality_baselines(self):
        """Set quality baselines for monitoring."""
        baselines = {
            'silhouette_score': 0.3,
            'coherence': 0.4,
            'noise_ratio': 0.2,
            'overall_quality_score': 0.3
        }
        
        for metric, baseline in baselines.items():
            self.monitoring.set_quality_baseline(metric, baseline)
    
    def _record_business_metrics(self, clustering_result: Dict[str, Any], input_count: int):
        """Record business-specific metrics."""
        metrics_collector = self.monitoring.metrics_collector
        
        # Success/failure tracking
        if clustering_result.get('total_clusters', 0) > 0:
            metrics_collector.record_counter('business.processing.success')
        else:
            metrics_collector.record_counter('business.processing.failure')
        
        # Processing efficiency
        processing_time = clustering_result.get('processing_time', 0)
        if processing_time > 0:
            efficiency = input_count / processing_time  # items per second
            metrics_collector.record_gauge('business.processing.efficiency', efficiency)
        
        # Cluster distribution
        total_clusters = clustering_result.get('total_clusters', 0)
        if total_clusters > 0:
            avg_cluster_size = input_count / total_clusters
            metrics_collector.record_gauge('business.clustering.avg_cluster_size', avg_cluster_size)
        
        # Data quality indicators
        noise_items = clustering_result.get('noise_items', 0)
        if input_count > 0:
            data_quality_score = 1.0 - (noise_items / input_count)
            metrics_collector.record_gauge('business.data.quality_score', data_quality_score)
    
    def _check_quality_issues(self, clustering_result: Dict[str, Any]):
        """Check for quality issues and trigger alerts if needed."""
        validation = clustering_result.get('validation', {})
        
        # Check overall quality
        quality_score = validation.get('quality_score', 0)
        if quality_score < 0.2:
            self.monitoring.alert_manager.send_alert(
                'low_quality_clustering',
                f'Clustering quality score ({quality_score:.3f}) is critically low',
                'warning',
                {
                    'quality_score': quality_score,
                    'threshold': 0.2,
                    'total_items': clustering_result.get('total_items', 0),
                    'total_clusters': clustering_result.get('total_clusters', 0)
                }
            )
        
        # Check for validation issues
        issues = validation.get('issues', [])
        if len(issues) > 3:
            self.monitoring.alert_manager.send_alert(
                'multiple_validation_issues',
                f'Clustering validation found {len(issues)} issues',
                'warning',
                {
                    'issue_count': len(issues),
                    'issues': issues[:3],  # First 3 issues
                    'recommendations': validation.get('recommendations', [])
                }
            )


# Example usage and integration patterns
def example_usage():
    """Example of how to use the monitored newsletter agent."""
    
    # Initialize monitored agent
    agent = MonitoredNewsletterAgent()
    
    try:
        # Start monitoring
        agent.start()
        
        # Example newsletter items
        newsletter_items = [
            {
                "headline": "AI Breakthrough in Language Models",
                "short_description": "New transformer architecture achieves state-of-the-art results",
                "source": "AI Research Weekly",
                "technologies": ["AI", "NLP", "Transformers"],
                "companies": ["OpenAI", "Google"]
            },
            {
                "headline": "Cloud Computing Trends 2024",
                "short_description": "Analysis of emerging cloud technologies and market trends",
                "source": "Cloud Tech Today",
                "technologies": ["Cloud", "AWS", "Azure"],
                "companies": ["Amazon", "Microsoft"]
            },
            # Add more items...
        ]
        
        # Process newsletters with monitoring
        result = agent.process_newsletters(newsletter_items)
        
        # Get monitoring status
        monitoring_status = agent.get_monitoring_status()
        
        print(f"Processing completed successfully!")
        print(f"Created {result.get('total_clusters', 0)} clusters")
        print(f"Quality score: {result.get('quality_score', 0):.3f}")
        print(f"Monitoring status: {monitoring_status['health']['current_status']}")
        
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        # Stop monitoring
        agent.stop()


@contextmanager
def monitoring_context():
    """Context manager for monitoring lifecycle."""
    monitoring = None
    try:
        # Start monitoring
        monitoring = start_monitoring()
        yield monitoring
    finally:
        # Stop monitoring
        if monitoring:
            monitoring.stop()


def integration_with_existing_agent():
    """Example of integrating monitoring with existing agent code."""
    
    # This shows how to add monitoring to the existing agent.py
    with monitoring_context() as monitoring:
        
        # Your existing newsletter processing code
        from ..agent import (
            fetch_newsletters, 
            summarize_and_extract_topics,
            apply_clustering_to_items
        )
        
        # Fetch newsletters with monitoring
        with monitor_operation('fetch_newsletters'):
            newsletters = fetch_newsletters("Newsletters", days_back=7)
        
        # Process each newsletter with monitoring
        all_items = []
        for newsletter in newsletters:
            with monitor_operation('process_newsletter'):
                # Extract and summarize
                clean_text = extract_clean_text(newsletter['body_html'])
                summary_result = summarize_and_extract_topics(
                    clean_text, 
                    "AI, ML, Tech",
                    newsletter['subject'],
                    newsletter['sender']
                )
                all_items.extend(summary_result.get('extracted_items', []))
        
        # Apply clustering with monitoring
        if all_items:
            with monitor_operation('clustering'):
                clustering_result = apply_clustering_to_items(all_items)
                
                # Record the clustering operation
                monitor_clustering_operation(clustering_result)
        
        # Get final monitoring status
        status = monitoring.get_monitoring_status()
        print(f"Processing completed with monitoring status: {status['health']['current_status']}")


if __name__ == "__main__":
    # Run example
    example_usage()