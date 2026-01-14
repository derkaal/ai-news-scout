# Newsletter Clustering System - API Documentation

Complete API reference for the newsletter clustering system, including all public interfaces, data structures, and usage examples.

## 📋 Table of Contents

- [Core Classes](#core-classes)
- [Configuration](#configuration)
- [Data Structures](#data-structures)
- [Main Interfaces](#main-interfaces)
- [Service APIs](#service-apis)
- [Error Handling](#error-handling)
- [Code Examples](#code-examples)

## 🏗️ Core Classes

### ClusteringOrchestrator

**Location**: [`newsletter_agent_core.clustering.orchestrator.ClusteringOrchestrator`](newsletter_agent_core/clustering/orchestrator.py:75)

Main coordination service for the clustering engine.

#### Constructor

```python
def __init__(self, config: Optional[ClusteringConfig] = None)
```

**Parameters**:
- `config` (Optional[ClusteringConfig]): Configuration object. Uses defaults if None.

**Example**:
```python
from newsletter_agent_core.clustering import ClusteringOrchestrator, ClusteringConfig

# Use default configuration
orchestrator = ClusteringOrchestrator()

# Use custom configuration
config = ClusteringConfig()
config.default_algorithm = "hdbscan"
orchestrator = ClusteringOrchestrator(config)
```

#### Main Methods

##### cluster_items()

```python
def cluster_items(
    self,
    items: List[Dict[str, Any]],
    text_field: str = "short_description",
    algorithm: Optional[str] = None,
    validate_results: bool = True
) -> Dict[str, Any]
```

Performs complete clustering workflow on newsletter items.

**Parameters**:
- `items` (List[Dict[str, Any]]): Newsletter items to cluster
- `text_field` (str): Field containing text to cluster on (default: "short_description")
- `algorithm` (Optional[str]): Algorithm to use ("hdbscan", "hierarchical", "hybrid")
- `validate_results` (bool): Whether to validate clustering results (default: True)

**Returns**: Dict[str, Any] - Clustering results with metadata

**Example**:
```python
items = [
    {
        "headline": "AI Breakthrough in Healthcare",
        "short_description": "New AI model improves medical diagnosis accuracy",
        "source": "TechNews",
        "url": "https://example.com/1"
    },
    {
        "headline": "Machine Learning in Finance",
        "short_description": "Banks adopt ML for fraud detection systems",
        "source": "FinanceDaily",
        "url": "https://example.com/2"
    }
]

result = orchestrator.cluster_items(
    items=items,
    text_field="short_description",
    algorithm="hybrid",
    validate_results=True
)

print(f"Found {result['total_clusters']} clusters")
print(f"Quality score: {result['quality_score']:.2f}")
```

##### get_performance_stats()

```python
def get_performance_stats(self) -> Dict[str, Any]
```

Returns performance statistics for the orchestrator.

**Returns**: Dict with performance metrics

**Example**:
```python
stats = orchestrator.get_performance_stats()
print(f"Total items processed: {stats['total_items_processed']}")
print(f"Cache hit rate: {stats['cache_hit_rate']:.2f}")
print(f"Average processing time: {stats['total_processing_time']:.2f}s")
```

##### clear_cache()

```python
def clear_cache(self)
```

Clears the embedding cache.

**Example**:
```python
orchestrator.clear_cache()
print("Cache cleared successfully")
```

##### update_config()

```python
def update_config(self, new_config: ClusteringConfig)
```

Updates the orchestrator configuration.

**Parameters**:
- `new_config` (ClusteringConfig): New configuration to apply

**Example**:
```python
new_config = ClusteringConfig()
new_config.performance.max_processing_time_seconds = 60
orchestrator.update_config(new_config)
```

## ⚙️ Configuration

### ClusteringConfig

**Location**: [`newsletter_agent_core.clustering.config.settings.ClusteringConfig`](newsletter_agent_core/clustering/config/settings.py:97)

Main configuration class with hierarchical settings.

#### Constructor

```python
@dataclass
class ClusteringConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    hdbscan: HDBSCANConfig = field(default_factory=HDBSCANConfig)
    hierarchical: HierarchicalConfig = field(default_factory=HierarchicalConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    
    # General settings
    default_algorithm: str = "hybrid"
    enable_logging: bool = True
    log_level: str = "INFO"
    random_seed: int = 42
    
    # Integration settings
    google_sheets_integration: bool = True
    add_clustering_metadata: bool = True
    preserve_original_data: bool = True
```

#### Methods

##### to_dict()

```python
def to_dict(self) -> Dict[str, Any]
```

Converts configuration to dictionary format.

**Example**:
```python
config = ClusteringConfig()
config_dict = config.to_dict()
print(json.dumps(config_dict, indent=2))
```

##### from_dict()

```python
@classmethod
def from_dict(cls, config_dict: Dict[str, Any]) -> 'ClusteringConfig'
```

Creates configuration from dictionary.

**Example**:
```python
config_dict = {
    "default_algorithm": "hdbscan",
    "hdbscan": {
        "min_cluster_size": 2,
        "min_samples": 1
    },
    "performance": {
        "max_processing_time_seconds": 45
    }
}

config = ClusteringConfig.from_dict(config_dict)
```

### Sub-Configuration Classes

#### EmbeddingConfig

```python
@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    cache_dir: str = "~/.newsletter_agent/embeddings"
    cache_enabled: bool = True
    batch_size: int = 32
    max_sequence_length: int = 512
    normalize_embeddings: bool = True
    device: str = "cpu"
```

#### HDBSCANConfig

```python
@dataclass
class HDBSCANConfig:
    min_cluster_size: int = 3
    min_samples: int = 2
    cluster_selection_epsilon: float = 0.0
    max_cluster_size: Optional[int] = None
    metric: str = "euclidean"
    cluster_selection_method: str = "eom"
    allow_single_cluster: bool = False
    prediction_data: bool = True
```

#### PerformanceConfig

```python
@dataclass
class PerformanceConfig:
    max_processing_time_seconds: int = 30
    max_memory_usage_gb: float = 2.0
    enable_parallel_processing: bool = True
    max_workers: Optional[int] = None
    chunk_size: int = 100
    enable_progress_tracking: bool = True
```

## 📊 Data Structures

### ClusteringResult

**Location**: [`newsletter_agent_core.clustering.algorithms.hdbscan_clusterer.ClusteringResult`](newsletter_agent_core/clustering/algorithms/hdbscan_clusterer.py:21)

Result object returned by clustering algorithms.

```python
@dataclass
class ClusteringResult:
    labels: np.ndarray                    # Cluster assignments
    n_clusters: int                       # Number of clusters found
    n_noise: int                         # Number of noise points
    algorithm: str                       # Algorithm used
    processing_time: float               # Time taken in seconds
    probabilities: Optional[np.ndarray]  # Cluster membership probabilities
    outlier_scores: Optional[np.ndarray] # Outlier detection scores
    cluster_centers: Optional[np.ndarray] # Cluster centroids
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "labels": self.labels.tolist(),
            "n_clusters": self.n_clusters,
            "n_noise": self.n_noise,
            "algorithm": self.algorithm,
            "processing_time": self.processing_time,
            "probabilities": self.probabilities.tolist() if self.probabilities is not None else None,
            "outlier_scores": self.outlier_scores.tolist() if self.outlier_scores is not None else None
        }
```

### ValidationResult

**Location**: [`newsletter_agent_core.clustering.validation.validator.ValidationResult`](newsletter_agent_core/clustering/validation/validator.py:1)

Result object from clustering validation.

```python
@dataclass
class ValidationResult:
    is_valid: bool                       # Overall validation status
    quality_score: float                 # Composite quality score (0-1)
    silhouette_score: float             # Cluster separation metric
    source_diversity: float             # Source distribution score
    noise_ratio: float                  # Proportion of noise points
    cluster_coherence: float            # Semantic consistency score
    issues: List[str]                   # List of identified issues
    recommendations: List[str]          # Improvement suggestions
    metrics: Dict[str, float]           # Additional quality metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "is_valid": self.is_valid,
            "quality_score": self.quality_score,
            "silhouette_score": self.silhouette_score,
            "source_diversity": self.source_diversity,
            "noise_ratio": self.noise_ratio,
            "cluster_coherence": self.cluster_coherence,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "metrics": self.metrics
        }
```

### Enhanced Item Structure

Items returned by [`cluster_items()`](newsletter_agent_core/clustering/orchestrator.py:125) include additional clustering metadata:

```python
enhanced_item = {
    # Original item data (preserved)
    "headline": str,
    "short_description": str,
    "source": str,
    "url": str,
    # ... other original fields
    
    # Clustering metadata (added)
    "cluster_id": int,              # Cluster identifier (-1 for noise)
    "is_noise": bool,               # True if item is an outlier
    "cluster_probability": float,   # Membership confidence (0.0-1.0)
    "outlier_score": float         # Anomaly detection score
}
```

### Cluster Summary Structure

```python
cluster_summary = {
    "cluster_id": int,                    # Cluster identifier
    "size": int,                         # Number of items in cluster
    "representative_items": List[Dict],   # Most representative items
    "top_sources": Dict[str, int],       # Source distribution
    "top_technologies": Dict[str, int],   # Technology mentions
    "top_companies": Dict[str, int],     # Company mentions
    "avg_cluster_distance": float        # Average intra-cluster distance
}
```

## 🔧 Service APIs

### EmbeddingService

**Location**: [`newsletter_agent_core.clustering.embedding.service.EmbeddingService`](newsletter_agent_core/clustering/embedding/service.py:1)

Service for generating and caching text embeddings.

#### Constructor

```python
def __init__(self, embedding_config: EmbeddingConfig, cache_config: CacheConfig)
```

#### Methods

##### generate_embeddings()

```python
def generate_embeddings(
    self,
    texts: List[str],
    show_progress: bool = False
) -> np.ndarray
```

Generates embeddings for the given texts with caching.

**Parameters**:
- `texts` (List[str]): List of texts to embed
- `show_progress` (bool): Whether to show progress bar

**Returns**: np.ndarray - Embedding matrix (n_texts, embedding_dim)

**Example**:
```python
from newsletter_agent_core.clustering.embedding.service import EmbeddingService
from newsletter_agent_core.clustering.config.settings import EmbeddingConfig, CacheConfig

embedding_service = EmbeddingService(
    EmbeddingConfig(),
    CacheConfig()
)

texts = ["AI breakthrough in healthcare", "Machine learning in finance"]
embeddings = embedding_service.generate_embeddings(texts, show_progress=True)
print(f"Generated embeddings shape: {embeddings.shape}")
```

##### get_cache_stats()

```python
def get_cache_stats(self) -> Dict[str, Any]
```

Returns cache performance statistics.

**Example**:
```python
stats = embedding_service.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2f}")
print(f"Cache size: {stats['cache_size_mb']:.1f} MB")
```

### Individual Clustering Algorithms

#### HDBSCANClusterer

```python
from newsletter_agent_core.clustering.algorithms.hdbscan_clusterer import HDBSCANClusterer

clusterer = HDBSCANClusterer(hdbscan_config)
result = clusterer.fit_predict(embeddings)
```

#### HierarchicalClusterer

```python
from newsletter_agent_core.clustering.algorithms.hierarchical_clusterer import HierarchicalClusterer

clusterer = HierarchicalClusterer(hierarchical_config)
result = clusterer.fit_predict(embeddings)
```

#### HybridClusterer

```python
from newsletter_agent_core.clustering.algorithms.hybrid_clusterer import HybridClusterer

clusterer = HybridClusterer(hybrid_config, hdbscan_config, hierarchical_config)
result = clusterer.fit_predict(embeddings)
```

### ClusteringValidator

**Location**: [`newsletter_agent_core.clustering.validation.validator.ClusteringValidator`](newsletter_agent_core/clustering/validation/validator.py:1)

Service for validating clustering quality.

#### Constructor

```python
def __init__(self, validation_config: ValidationConfig)
```

#### Methods

##### validate_clustering()

```python
def validate_clustering(
    self,
    embeddings: np.ndarray,
    clustering_result: ClusteringResult,
    items: List[Dict[str, Any]]
) -> ValidationResult
```

Validates clustering results using multiple quality metrics.

**Example**:
```python
from newsletter_agent_core.clustering.validation.validator import ClusteringValidator

validator = ClusteringValidator(validation_config)
validation_result = validator.validate_clustering(embeddings, clustering_result, items)

if validation_result.is_valid:
    print(f"Clustering is valid with quality score: {validation_result.quality_score:.2f}")
else:
    print("Issues found:")
    for issue in validation_result.issues:
        print(f"  - {issue}")
    
    print("Recommendations:")
    for rec in validation_result.recommendations:
        print(f"  - {rec}")
```

## ❌ Error Handling

### Exception Types

The system defines several custom exception types for specific error scenarios:

#### TimeoutError

Raised when clustering exceeds the configured time limit.

```python
try:
    result = orchestrator.cluster_items(items)
except TimeoutError as e:
    print(f"Clustering timed out: {e}")
    # Handle timeout scenario
```

#### MemoryError

Raised when memory usage exceeds configured limits.

```python
try:
    result = orchestrator.cluster_items(items)
except MemoryError as e:
    print(f"Memory limit exceeded: {e}")
    # Reduce batch size or item count
```

#### ValidationError

Raised when configuration validation fails.

```python
try:
    config = ClusteringConfig()
    config.performance.max_processing_time_seconds = -1  # Invalid
    orchestrator = ClusteringOrchestrator(config)
except ValueError as e:
    print(f"Configuration error: {e}")
```

### Error Result Structures

When errors occur, the system returns structured error results instead of raising exceptions:

#### Timeout Result

```python
timeout_result = {
    "items": original_items,
    "clustering_result": {
        "algorithm": "timeout",
        "n_clusters": 1,
        "labels": [0] * len(items)  # Single cluster fallback
    },
    "error": "Clustering timed out",
    "is_valid": False,
    "quality_score": 0.0
}
```

#### Error Result

```python
error_result = {
    "items": original_items,
    "clustering_result": {
        "algorithm": "error",
        "n_clusters": 0,
        "labels": [-1] * len(items)  # All noise
    },
    "error": "Detailed error message",
    "is_valid": False,
    "quality_score": 0.0
}
```

## 💡 Code Examples

### Complete Workflow Example

```python
import logging
from newsletter_agent_core.clustering import ClusteringOrchestrator, ClusteringConfig

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create custom configuration
config = ClusteringConfig()
config.default_algorithm = "hybrid"
config.performance.max_processing_time_seconds = 45
config.hdbscan.min_cluster_size = 2
config.validation.min_silhouette_score = 0.25

# Initialize orchestrator
orchestrator = ClusteringOrchestrator(config)

# Sample newsletter items
items = [
    {
        "headline": "OpenAI Releases GPT-4 Turbo",
        "short_description": "New language model with improved performance and reduced costs",
        "source": "TechCrunch",
        "url": "https://techcrunch.com/gpt4-turbo",
        "technologies": ["AI", "NLP", "Machine Learning"],
        "companies": ["OpenAI"]
    },
    {
        "headline": "Google's Bard Gets Major Update",
        "short_description": "Enhanced conversational AI capabilities and multimodal support",
        "source": "The Verge",
        "url": "https://theverge.com/bard-update",
        "technologies": ["AI", "NLP", "Conversational AI"],
        "companies": ["Google"]
    },
    {
        "headline": "Tesla Autopilot Improvement",
        "short_description": "New neural network architecture improves self-driving capabilities",
        "source": "Electrek",
        "url": "https://electrek.co/autopilot",
        "technologies": ["AI", "Computer Vision", "Autonomous Vehicles"],
        "companies": ["Tesla"]
    },
    {
        "headline": "Microsoft Azure AI Services",
        "short_description": "Cloud-based AI tools for enterprise applications",
        "source": "Microsoft Blog",
        "url": "https://microsoft.com/azure-ai",
        "technologies": ["AI", "Cloud Computing", "Enterprise Software"],
        "companies": ["Microsoft"]
    }
]

# Perform clustering
try:
    result = orchestrator.cluster_items(
        items=items,
        text_field="short_description",
        algorithm="hybrid",
        validate_results=True
    )
    
    # Process results
    print(f"Clustering completed successfully!")
    print(f"Total items: {result['total_items']}")
    print(f"Clusters found: {result['total_clusters']}")
    print(f"Noise items: {result['noise_items']}")
    print(f"Processing time: {result['processing_time']:.2f}s")
    
    if result.get('is_valid'):
        print(f"Quality score: {result['quality_score']:.2f}")
    else:
        print("Quality issues detected:")
        for issue in result.get('issues', []):
            print(f"  - {issue}")
    
    # Examine clusters
    for summary in result['cluster_summaries']:
        print(f"\nCluster {summary['cluster_id']} ({summary['size']} items):")
        print(f"  Top technologies: {summary['top_technologies']}")
        print(f"  Top companies: {summary['top_companies']}")
        print("  Representative items:")
        for item in summary['representative_items']:
            print(f"    - {item['headline']}")
    
    # Access enhanced items
    print("\nEnhanced items with clustering metadata:")
    for item in result['items']:
        print(f"  {item['headline']}")
        print(f"    Cluster: {item['cluster_id']}")
        print(f"    Confidence: {item.get('cluster_probability', 'N/A')}")
        print(f"    Is outlier: {item.get('is_noise', False)}")

except Exception as e:
    print(f"Clustering failed: {e}")
    
# Get performance statistics
stats = orchestrator.get_performance_stats()
print(f"\nPerformance Statistics:")
print(f"  Total items processed: {stats['total_items_processed']}")
print(f"  Cache hit rate: {stats['cache_hit_rate']:.2f}")
print(f"  Success rate: {stats['successful_clusterings'] / max(1, stats['successful_clusterings'] + stats['failed_clusterings']):.2f}")
```

### Configuration Management Example

```python
from newsletter_agent_core.clustering.config.settings import (
    ClusteringConfig, create_development_config, create_production_config
)

# Development configuration
dev_config = create_development_config()
dev_config.log_level = "DEBUG"
dev_config.performance.enable_progress_tracking = True

# Production configuration
prod_config = create_production_config()
prod_config.performance.max_memory_usage_gb = 4.0
prod_config.cache.max_cache_size_mb = 1000

# Environment-based configuration
env_config = ClusteringConfig()
# Automatically applies environment variable overrides

# Custom configuration from dictionary
config_dict = {
    "default_algorithm": "hdbscan",
    "hdbscan": {
        "min_cluster_size": 2,
        "min_samples": 1
    },
    "performance": {
        "max_processing_time_seconds": 60,
        "max_memory_usage_gb": 3.0
    },
    "validation": {
        "min_silhouette_score": 0.3,
        "enable_diversity_check": True
    }
}

custom_config = ClusteringConfig.from_dict(config_dict)
```

### Batch Processing Example

```python
def process_large_newsletter_dataset(all_items, batch_size=200):
    """Process large datasets in batches to manage memory usage."""
    
    orchestrator = ClusteringOrchestrator()
    all_results = []
    
    for i in range(0, len(all_items), batch_size):
        batch = all_items[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}: {len(batch)} items")
        
        try:
            result = orchestrator.cluster_items(
                items=batch,
                text_field="short_description",
                validate_results=True
            )
            
            # Adjust cluster IDs to be globally unique
            cluster_offset = sum(r['total_clusters'] for r in all_results)
            for item in result['items']:
                if item['cluster_id'] != -1:  # Not noise
                    item['cluster_id'] += cluster_offset
            
            all_results.append(result)
            
        except Exception as e:
            print(f"Batch {i//batch_size + 1} failed: {e}")
            continue
    
    # Combine results
    combined_items = []
    total_clusters = 0
    total_processing_time = 0
    
    for result in all_results:
        combined_items.extend(result['items'])
        total_clusters += result['total_clusters']
        total_processing_time += result['processing_time']
    
    return {
        'items': combined_items,
        'total_items': len(combined_items),
        'total_clusters': total_clusters,
        'total_processing_time': total_processing_time,
        'batch_results': all_results
    }

# Usage
large_dataset = load_newsletter_items()  # Your data loading function
result = process_large_newsletter_dataset(large_dataset, batch_size=150)
print(f"Processed {result['total_items']} items in {result['total_processing_time']:.2f}s")
```

### Custom Validation Example

```python
from newsletter_agent_core.clustering.validation.validator import ClusteringValidator
from newsletter_agent_core.clustering.config.settings import ValidationConfig

# Custom validation configuration
validation_config = ValidationConfig()
validation_config.min_silhouette_score = 0.4  # Higher quality threshold
validation_config.max_noise_ratio = 0.2       # Lower noise tolerance
validation_config.min_source_diversity = 0.3  # Better source distribution

validator = ClusteringValidator(validation_config)

# Validate clustering results
validation_result = validator.validate_clustering(embeddings, clustering_result, items)

# Custom quality assessment
def assess_clustering_quality(validation_result):
    """Custom quality assessment logic."""
    
    quality_levels = {
        (0.8, 1.0): "Excellent",
        (0.6, 0.8): "Good", 
        (0.4, 0.6): "Fair",
        (0.2, 0.4): "Poor",
        (0.0, 0.2): "Very Poor"
    }
    
    score = validation_result.quality_score
    
    for (min_score, max_score), level in quality_levels.items():
        if min_score <= score < max_score:
            return level
    
    return "Unknown"

quality_level = assess_clustering_quality(validation_result)
print(f"Clustering quality: {quality_level} (score: {validation_result.quality_score:.2f})")

# Act on validation results
if not validation_result.is_valid:
    print("Clustering quality issues detected:")
    for issue in validation_result.issues:
        print(f"  - {issue}")
    
    print("Recommended actions:")
    for recommendation in validation_result.recommendations:
        print(f"  - {recommendation}")
```

This API documentation provides comprehensive coverage of all public interfaces, data structures, and usage patterns for the newsletter clustering system. Use it as a reference for integrating the clustering functionality into your applications.