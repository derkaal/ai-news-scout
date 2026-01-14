# Newsletter Clustering System

A high-performance, modular clustering engine designed to automatically group newsletter items by semantic similarity. The system integrates seamlessly with existing Google Sheets workflows and provides comprehensive validation and performance monitoring.

## 🚀 Quick Start

### Enable Clustering in Your Newsletter Agent

1. **Install clustering dependencies:**
   ```bash
   pip install -r requirements_clustering.txt
   ```

2. **Basic usage in your existing workflow:**
   ```python
   from newsletter_agent_core.clustering import ClusteringOrchestrator
   
   # Initialize with default configuration
   orchestrator = ClusteringOrchestrator()
   
   # Cluster your newsletter items
   result = orchestrator.cluster_items(
       items=your_newsletter_items,
       text_field="short_description"
   )
   
   # Access clustered items with metadata
   clustered_items = result["items"]
   cluster_summaries = result["cluster_summaries"]
   ```

3. **Integration with Google Sheets:**
   The system automatically adds clustering metadata to your items:
   - `cluster_id`: Unique identifier for each cluster
   - `is_noise`: Boolean indicating outlier items
   - `cluster_probability`: Confidence score (0.0-1.0)
   - `outlier_score`: Anomaly detection score

## 📋 Features

### Core Capabilities
- **Multiple Algorithms**: HDBSCAN, Hierarchical, and Hybrid clustering
- **Semantic Embeddings**: Sentence transformer-based text understanding
- **Performance Optimized**: <30s processing for 400 items, <2GB memory usage
- **Intelligent Caching**: File-based embedding cache with TTL management
- **Comprehensive Validation**: Quality metrics and source diversity tracking
- **Google Sheets Integration**: Seamless workflow integration

### Performance Characteristics
- **Processing Time**: <30 seconds for 400 newsletter items
- **Memory Usage**: <2GB peak memory consumption
- **Cache Hit Rate**: 80%+ for repeated content
- **Scalability**: Handles 1000+ items with linear scaling

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- 2GB+ available RAM
- 500MB+ disk space for caching

### Dependencies Installation
```bash
# Install clustering-specific dependencies
pip install -r requirements_clustering.txt

# Core dependencies (if not already installed)
pip install sentence-transformers>=2.2.2
pip install hdbscan>=0.8.29
pip install scikit-learn>=1.3.0
pip install numpy>=1.21.0
pip install psutil>=5.9.0
```

### Optional GPU Support
For faster embedding generation:
```bash
pip install torch>=1.13.0 torchvision>=0.14.0
```

## ⚙️ Configuration

### Environment Variables
Configure the system using environment variables:

```bash
# Algorithm selection
export CLUSTERING_DEFAULT_ALGORITHM=hybrid  # hdbscan, hierarchical, hybrid

# Performance constraints
export CLUSTERING_MAX_TIME=30               # seconds
export CLUSTERING_MAX_MEMORY=2.0            # GB

# Embedding configuration
export CLUSTERING_EMBEDDING_MODEL=all-MiniLM-L6-v2
export CLUSTERING_CACHE_DIR=/path/to/cache

# Logging
export CLUSTERING_LOG_LEVEL=INFO            # DEBUG, INFO, WARNING, ERROR
```

### Configuration Options

#### Algorithm Selection
- **`hybrid`** (default): Combines HDBSCAN and hierarchical clustering
- **`hdbscan`**: Density-based clustering, good for varying cluster sizes
- **`hierarchical`**: Traditional hierarchical clustering, consistent results

#### Performance Tuning
```python
from newsletter_agent_core.clustering.config import ClusteringConfig

config = ClusteringConfig()
config.performance.max_processing_time_seconds = 45  # Increase timeout
config.performance.max_memory_usage_gb = 3.0         # Allow more memory
config.hdbscan.min_cluster_size = 2                  # Smaller clusters
```

## 📊 Usage Examples

### Basic Clustering
```python
from newsletter_agent_core.clustering import ClusteringOrchestrator

orchestrator = ClusteringOrchestrator()

# Your newsletter items
items = [
    {
        "headline": "New AI Framework Released",
        "short_description": "OpenAI releases new framework for developers",
        "source": "TechCrunch",
        "url": "https://example.com/1"
    },
    {
        "headline": "Machine Learning Breakthrough",
        "short_description": "Researchers achieve new ML performance records",
        "source": "Nature",
        "url": "https://example.com/2"
    }
    # ... more items
]

result = orchestrator.cluster_items(items)
print(f"Found {result['total_clusters']} clusters")
```

### Advanced Configuration
```python
from newsletter_agent_core.clustering import ClusteringOrchestrator, ClusteringConfig

# Custom configuration
config = ClusteringConfig()
config.default_algorithm = "hdbscan"
config.hdbscan.min_cluster_size = 3
config.validation.min_silhouette_score = 0.3

orchestrator = ClusteringOrchestrator(config)

result = orchestrator.cluster_items(
    items=items,
    text_field="short_description",
    algorithm="hdbscan",  # Override config
    validate_results=True
)

# Check validation results
if result["is_valid"]:
    print(f"Quality score: {result['quality_score']:.2f}")
else:
    print("Issues found:", result["issues"])
```

### Working with Results
```python
result = orchestrator.cluster_items(items)

# Access clustered items
for item in result["items"]:
    cluster_id = item["cluster_id"]
    probability = item["cluster_probability"]
    is_outlier = item["is_noise"]
    
    print(f"Item: {item['headline']}")
    print(f"Cluster: {cluster_id}, Confidence: {probability:.2f}")

# Examine cluster summaries
for summary in result["cluster_summaries"]:
    print(f"\nCluster {summary['cluster_id']} ({summary['size']} items):")
    print(f"Top sources: {summary['top_sources']}")
    print(f"Top technologies: {summary['top_technologies']}")
    
    # Representative items
    for rep_item in summary["representative_items"]:
        print(f"  - {rep_item['headline']}")
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Memory Errors
```
Error: Memory limit exceeded during processing
```
**Solution:**
- Increase memory limit: `export CLUSTERING_MAX_MEMORY=4.0`
- Reduce batch size in configuration
- Process items in smaller chunks

#### 2. Timeout Issues
```
Error: Clustering timed out after 30s
```
**Solution:**
- Increase timeout: `export CLUSTERING_MAX_TIME=60`
- Use faster algorithm: `export CLUSTERING_DEFAULT_ALGORITHM=hierarchical`
- Reduce item count or text length

#### 3. Poor Clustering Quality
```
Warning: Low silhouette score (0.15)
```
**Solution:**
- Adjust algorithm parameters:
  ```python
  config.hdbscan.min_cluster_size = 2  # Smaller clusters
  config.hierarchical.n_clusters = 10  # Fixed cluster count
  ```
- Try different algorithms
- Check text quality and diversity

#### 4. Cache Issues
```
Warning: Failed to load cache metadata
```
**Solution:**
- Clear cache directory: `rm -rf ~/.newsletter_agent/embeddings/*`
- Check disk space and permissions
- Disable caching temporarily: `config.cache.cache_enabled = False`

### Performance Optimization

#### For Large Datasets (1000+ items)
```python
config = ClusteringConfig()
config.performance.max_processing_time_seconds = 120
config.performance.max_memory_usage_gb = 4.0
config.performance.chunk_size = 50
config.embedding.batch_size = 16
```

#### For Fast Processing
```python
config = ClusteringConfig()
config.default_algorithm = "hierarchical"
config.hierarchical.n_clusters = 15  # Fixed number
config.validation.enable_diversity_check = False
```

### Debugging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or via environment
export CLUSTERING_LOG_LEVEL=DEBUG
```

Check performance statistics:
```python
stats = orchestrator.get_performance_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.2f}")
print(f"Average processing time: {stats['total_processing_time']:.2f}s")
```

## 🔗 Integration Guide

### Google Sheets Workflow
The clustering system integrates seamlessly with your existing Google Sheets workflow:

1. **Automatic Metadata Addition**: Cluster information is added to each item
2. **Preserved Data**: Original item data remains unchanged
3. **Enhanced Schema**: New columns for clustering insights

### Custom Text Fields
Cluster on different text fields:
```python
# Use headline instead of description
result = orchestrator.cluster_items(items, text_field="headline")

# Combine multiple fields
for item in items:
    item["combined_text"] = f"{item['headline']} {item['short_description']}"

result = orchestrator.cluster_items(items, text_field="combined_text")
```

### Batch Processing
For large datasets:
```python
def process_large_dataset(all_items, batch_size=200):
    results = []
    
    for i in range(0, len(all_items), batch_size):
        batch = all_items[i:i + batch_size]
        result = orchestrator.cluster_items(batch)
        results.append(result)
    
    return results
```

## 📈 Performance Monitoring

Monitor system performance:
```python
# Get performance metrics
result = orchestrator.cluster_items(items)
metrics = result["performance_metrics"]

print(f"Processing time: {metrics['elapsed_time']:.2f}s")
print(f"Peak memory: {metrics['peak_memory_gb']:.2f}GB")
print(f"Memory limit exceeded: {metrics['memory_limit_exceeded']}")

# Get overall statistics
stats = orchestrator.get_performance_stats()
print(f"Total items processed: {stats['total_items_processed']}")
print(f"Success rate: {stats['successful_clusterings'] / (stats['successful_clusterings'] + stats['failed_clusterings']):.2f}")
```

## 📚 Additional Documentation

- **[Technical Documentation](CLUSTERING.md)**: Architecture and algorithm details
- **[API Documentation](API.md)**: Complete API reference
- **[Deployment Guide](DEPLOYMENT.md)**: Production deployment and tuning

## 🤝 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the technical documentation
3. Enable debug logging for detailed error information
4. Check performance metrics for bottlenecks

## 📄 License

This clustering system is part of the Newsletter Agent project and follows the same licensing terms.