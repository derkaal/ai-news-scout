# Newsletter Clustering System - Deployment Guide

Comprehensive guide for deploying the newsletter clustering system in production environments, including setup, configuration, performance tuning, and monitoring.

## 📋 Table of Contents

- [Environment Setup](#environment-setup)
- [Dependencies Management](#dependencies-management)
- [Configuration Management](#configuration-management)
- [Performance Tuning](#performance-tuning)
- [Production Deployment](#production-deployment)
- [Monitoring & Health Checks](#monitoring--health-checks)
- [Scaling Considerations](#scaling-considerations)
- [Troubleshooting](#troubleshooting)

## 🏗️ Environment Setup

### System Requirements

#### Minimum Requirements
- **CPU**: 2 cores, 2.0 GHz
- **RAM**: 4GB available memory
- **Storage**: 2GB free disk space
- **Python**: 3.8 or higher
- **OS**: Linux, macOS, or Windows

#### Recommended Production Requirements
- **CPU**: 4+ cores, 2.5+ GHz
- **RAM**: 8GB+ available memory
- **Storage**: 10GB+ free disk space (for caching)
- **Python**: 3.9 or higher
- **OS**: Linux (Ubuntu 20.04+ or CentOS 8+)

#### Optional GPU Requirements
- **GPU**: NVIDIA GPU with CUDA support
- **VRAM**: 2GB+ for embedding acceleration
- **CUDA**: Version 11.0 or higher
- **cuDNN**: Compatible version

### Python Environment Setup

#### Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv newsletter_clustering_env

# Activate environment
# Linux/macOS:
source newsletter_clustering_env/bin/activate
# Windows:
newsletter_clustering_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### Using Conda

```bash
# Create conda environment
conda create -n newsletter_clustering python=3.9
conda activate newsletter_clustering

# Install pip in conda environment
conda install pip
```

### Directory Structure Setup

```bash
# Create application directory
mkdir -p /opt/newsletter_agent
cd /opt/newsletter_agent

# Create necessary subdirectories
mkdir -p {logs,cache,config,data,backups}

# Set permissions (Linux/macOS)
chmod 755 /opt/newsletter_agent
chmod 777 cache logs  # Write permissions for cache and logs
```

## 📦 Dependencies Management

### Core Dependencies Installation

```bash
# Install clustering dependencies
pip install -r requirements_clustering.txt

# Verify installation
python -c "
import sentence_transformers
import hdbscan
import sklearn
import numpy as np
import psutil
print('All core dependencies installed successfully')
"
```

### Optional GPU Support

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU availability
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
"
```

### Development Dependencies (Optional)

```bash
# Install development and testing tools
pip install pytest pytest-cov black flake8 mypy

# Install Jupyter for analysis (optional)
pip install jupyter matplotlib seaborn
```

### Requirements Lock File

Create a `requirements-lock.txt` for reproducible deployments:

```bash
# Generate lock file
pip freeze > requirements-lock.txt

# Install from lock file in production
pip install -r requirements-lock.txt
```

## ⚙️ Configuration Management

### Environment Variables

Create a comprehensive environment configuration file:

```bash
# /opt/newsletter_agent/config/clustering.env

# Algorithm Configuration
CLUSTERING_DEFAULT_ALGORITHM=hybrid
CLUSTERING_EMBEDDING_MODEL=all-MiniLM-L6-v2

# Performance Configuration
CLUSTERING_MAX_TIME=30
CLUSTERING_MAX_MEMORY=2.0
CLUSTERING_ENABLE_PARALLEL=true
CLUSTERING_MAX_WORKERS=4

# Cache Configuration
CLUSTERING_CACHE_DIR=/opt/newsletter_agent/cache
CLUSTERING_CACHE_TTL=24
CLUSTERING_MAX_CACHE_SIZE=1000

# Logging Configuration
CLUSTERING_LOG_LEVEL=INFO
CLUSTERING_ENABLE_LOGGING=true

# Integration Configuration
CLUSTERING_GOOGLE_SHEETS_INTEGRATION=true
CLUSTERING_ADD_METADATA=true
CLUSTERING_PRESERVE_DATA=true

# Security Configuration
CLUSTERING_RANDOM_SEED=42
```

### Configuration Loading

```python
# config_loader.py
import os
from pathlib import Path
from dotenv import load_dotenv
from newsletter_agent_core.clustering.config.settings import ClusteringConfig

def load_production_config():
    """Load production configuration with environment overrides."""
    
    # Load environment file
    env_file = Path("/opt/newsletter_agent/config/clustering.env")
    if env_file.exists():
        load_dotenv(env_file)
    
    # Create configuration with environment overrides
    config = ClusteringConfig()
    
    # Production-specific overrides
    config.enable_logging = True
    config.log_level = os.getenv("CLUSTERING_LOG_LEVEL", "INFO")
    config.performance.enable_parallel_processing = True
    config.cache.max_cache_size_mb = int(os.getenv("CLUSTERING_MAX_CACHE_SIZE", "1000"))
    
    return config
```

### Configuration Validation

```python
# config_validator.py
def validate_production_config(config):
    """Validate production configuration."""
    
    issues = []
    
    # Check performance settings
    if config.performance.max_processing_time_seconds < 30:
        issues.append("Processing timeout too low for production")
    
    if config.performance.max_memory_usage_gb < 2.0:
        issues.append("Memory limit too low for production workloads")
    
    # Check cache settings
    cache_dir = Path(config.embedding.cache_dir)
    if not cache_dir.exists():
        issues.append(f"Cache directory does not exist: {cache_dir}")
    
    if not os.access(cache_dir, os.W_OK):
        issues.append(f"Cache directory not writable: {cache_dir}")
    
    # Check disk space
    import shutil
    free_space_gb = shutil.disk_usage(cache_dir).free / (1024**3)
    if free_space_gb < 5.0:
        issues.append(f"Insufficient disk space: {free_space_gb:.1f}GB available")
    
    return issues

# Usage
config = load_production_config()
issues = validate_production_config(config)
if issues:
    for issue in issues:
        print(f"Configuration issue: {issue}")
    exit(1)
```

## 🚀 Performance Tuning

### Memory Optimization

#### Configuration Tuning

```python
# High-memory environment (8GB+ RAM)
config.performance.max_memory_usage_gb = 4.0
config.embedding.batch_size = 64
config.performance.chunk_size = 200

# Low-memory environment (4GB RAM)
config.performance.max_memory_usage_gb = 1.5
config.embedding.batch_size = 16
config.performance.chunk_size = 50
```

#### Memory Monitoring Script

```python
# memory_monitor.py
import psutil
import time
import logging
from newsletter_agent_core.clustering import ClusteringOrchestrator

def monitor_memory_usage(orchestrator, items, duration=300):
    """Monitor memory usage during clustering."""
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024**2)  # MB
    peak_memory = initial_memory
    
    start_time = time.time()
    
    # Start clustering in background
    import threading
    result_container = {}
    
    def cluster_task():
        try:
            result_container['result'] = orchestrator.cluster_items(items)
        except Exception as e:
            result_container['error'] = str(e)
    
    cluster_thread = threading.Thread(target=cluster_task)
    cluster_thread.start()
    
    # Monitor memory
    while cluster_thread.is_alive() and (time.time() - start_time) < duration:
        current_memory = process.memory_info().rss / (1024**2)
        peak_memory = max(peak_memory, current_memory)
        
        logging.info(f"Current memory: {current_memory:.1f}MB, Peak: {peak_memory:.1f}MB")
        time.sleep(1)
    
    cluster_thread.join(timeout=10)
    
    return {
        'initial_memory_mb': initial_memory,
        'peak_memory_mb': peak_memory,
        'memory_increase_mb': peak_memory - initial_memory,
        'result': result_container.get('result'),
        'error': result_container.get('error')
    }
```

### CPU Optimization

#### Multi-threading Configuration

```python
import os
from newsletter_agent_core.clustering.config.settings import ClusteringConfig

# Auto-detect optimal worker count
cpu_count = os.cpu_count()
optimal_workers = min(cpu_count, 8)  # Cap at 8 workers

config = ClusteringConfig()
config.performance.enable_parallel_processing = True
config.performance.max_workers = optimal_workers

# For CPU-intensive workloads
config.embedding.batch_size = cpu_count * 8
```

#### CPU Affinity (Linux)

```python
# cpu_affinity.py
import os
import psutil

def set_cpu_affinity(cpu_cores=None):
    """Set CPU affinity for the current process."""
    
    if cpu_cores is None:
        # Use all available cores
        cpu_cores = list(range(os.cpu_count()))
    
    try:
        process = psutil.Process()
        process.cpu_affinity(cpu_cores)
        print(f"CPU affinity set to cores: {cpu_cores}")
    except Exception as e:
        print(f"Failed to set CPU affinity: {e}")

# Usage: Use cores 0-3 for clustering
set_cpu_affinity([0, 1, 2, 3])
```

### Storage Optimization

#### Cache Optimization

```python
# cache_optimizer.py
import os
import time
from pathlib import Path
from newsletter_agent_core.clustering.embedding.service import EmbeddingCache

def optimize_cache_performance():
    """Optimize cache for better I/O performance."""
    
    cache_dir = Path(os.getenv("CLUSTERING_CACHE_DIR", "~/.newsletter_agent/embeddings")).expanduser()
    
    # Check if cache is on SSD
    def is_ssd(path):
        # Simple heuristic: check if random access is fast
        test_file = path / "ssd_test.tmp"
        
        try:
            # Write test data
            with open(test_file, 'wb') as f:
                f.write(b'0' * 1024 * 1024)  # 1MB
            
            # Time random access
            start_time = time.time()
            with open(test_file, 'rb') as f:
                for _ in range(100):
                    f.seek(0)
                    f.read(1024)
            access_time = time.time() - start_time
            
            test_file.unlink()
            
            # SSD typically < 0.1s for this test
            return access_time < 0.1
            
        except Exception:
            return False
    
    if is_ssd(cache_dir):
        print("Cache is on SSD - optimal performance")
        return {
            'cache_type': 'ssd',
            'recommended_batch_size': 64,
            'recommended_cache_size': 2000
        }
    else:
        print("Cache is on HDD - consider moving to SSD")
        return {
            'cache_type': 'hdd',
            'recommended_batch_size': 32,
            'recommended_cache_size': 500
        }

# Apply optimizations
cache_info = optimize_cache_performance()
```

### Algorithm-Specific Tuning

#### HDBSCAN Optimization

```python
# For large datasets (1000+ items)
config.hdbscan.min_cluster_size = 5
config.hdbscan.min_samples = 3
config.hdbscan.cluster_selection_epsilon = 0.1

# For small datasets (< 100 items)
config.hdbscan.min_cluster_size = 2
config.hdbscan.min_samples = 1
config.hdbscan.cluster_selection_epsilon = 0.0
```

#### Hierarchical Optimization

```python
# For consistent performance
config.hierarchical.linkage = "ward"
config.hierarchical.n_clusters = 15  # Fixed number for predictable performance

# For quality over speed
config.hierarchical.linkage = "complete"
config.hierarchical.distance_threshold = None  # Auto-determine
```

## 🏭 Production Deployment

### Systemd Service (Linux)

Create a systemd service for the clustering system:

```ini
# /etc/systemd/system/newsletter-clustering.service
[Unit]
Description=Newsletter Clustering Service
After=network.target
Wants=network.target

[Service]
Type=simple
User=newsletter
Group=newsletter
WorkingDirectory=/opt/newsletter_agent
Environment=PYTHONPATH=/opt/newsletter_agent
EnvironmentFile=/opt/newsletter_agent/config/clustering.env
ExecStart=/opt/newsletter_agent/newsletter_clustering_env/bin/python -m newsletter_agent_core.clustering.service
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=newsletter-clustering

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096
MemoryMax=4G
CPUQuota=400%

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
# Enable service
sudo systemctl enable newsletter-clustering.service

# Start service
sudo systemctl start newsletter-clustering.service

# Check status
sudo systemctl status newsletter-clustering.service

# View logs
sudo journalctl -u newsletter-clustering.service -f
```

### Docker Deployment

#### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd -m -u 1000 newsletter && \
    mkdir -p /app/cache /app/logs && \
    chown -R newsletter:newsletter /app

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements_clustering.txt .
RUN pip install --no-cache-dir -r requirements_clustering.txt

# Copy application code
COPY newsletter_agent_core/ ./newsletter_agent_core/
COPY config/ ./config/

# Set ownership
RUN chown -R newsletter:newsletter /app

# Switch to non-root user
USER newsletter

# Set environment variables
ENV PYTHONPATH=/app
ENV CLUSTERING_CACHE_DIR=/app/cache
ENV CLUSTERING_LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from newsletter_agent_core.clustering import ClusteringOrchestrator; ClusteringOrchestrator()"

# Expose port (if running as service)
EXPOSE 8080

# Start application
CMD ["python", "-m", "newsletter_agent_core.clustering.service"]
```

#### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  newsletter-clustering:
    build: .
    container_name: newsletter-clustering
    restart: unless-stopped
    
    environment:
      - CLUSTERING_DEFAULT_ALGORITHM=hybrid
      - CLUSTERING_MAX_TIME=45
      - CLUSTERING_MAX_MEMORY=3.0
      - CLUSTERING_LOG_LEVEL=INFO
    
    volumes:
      - ./cache:/app/cache
      - ./logs:/app/logs
      - ./config:/app/config:ro
    
    ports:
      - "8080:8080"
    
    healthcheck:
      test: ["CMD", "python", "-c", "from newsletter_agent_core.clustering import ClusteringOrchestrator; ClusteringOrchestrator()"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'

  # Optional: Redis for distributed caching
  redis:
    image: redis:7-alpine
    container_name: newsletter-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

volumes:
  redis_data:
```

### Kubernetes Deployment

#### Deployment Manifest

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: newsletter-clustering
  labels:
    app: newsletter-clustering
spec:
  replicas: 2
  selector:
    matchLabels:
      app: newsletter-clustering
  template:
    metadata:
      labels:
        app: newsletter-clustering
    spec:
      containers:
      - name: clustering
        image: newsletter-clustering:latest
        ports:
        - containerPort: 8080
        
        env:
        - name: CLUSTERING_DEFAULT_ALGORITHM
          value: "hybrid"
        - name: CLUSTERING_MAX_TIME
          value: "45"
        - name: CLUSTERING_MAX_MEMORY
          value: "3.0"
        - name: CLUSTERING_LOG_LEVEL
          value: "INFO"
        
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        
        volumeMounts:
        - name: cache-volume
          mountPath: /app/cache
        - name: config-volume
          mountPath: /app/config
          readOnly: true
        
        livenessProbe:
          exec:
            command:
            - python
            - -c
            - "from newsletter_agent_core.clustering import ClusteringOrchestrator; ClusteringOrchestrator()"
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
        
        readinessProbe:
          exec:
            command:
            - python
            - -c
            - "from newsletter_agent_core.clustering import ClusteringOrchestrator; ClusteringOrchestrator()"
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
      
      volumes:
      - name: cache-volume
        persistentVolumeClaim:
          claimName: clustering-cache-pvc
      - name: config-volume
        configMap:
          name: clustering-config

---
apiVersion: v1
kind: Service
metadata:
  name: newsletter-clustering-service
spec:
  selector:
    app: newsletter-clustering
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: ClusterIP
```

## 📊 Monitoring & Health Checks

### Health Check Endpoint

```python
# health_check.py
import time
import logging
from typing import Dict, Any
from newsletter_agent_core.clustering import ClusteringOrchestrator, ClusteringConfig

class HealthChecker:
    """Health check service for clustering system."""
    
    def __init__(self):
        self.orchestrator = None
        self.last_check_time = 0
        self.check_interval = 300  # 5 minutes
        
    def comprehensive_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        
        health_status = {
            'status': 'healthy',
            'timestamp': time.time(),
            'checks': {}
        }
        
        try:
            # Check 1: Configuration validation
            config = ClusteringConfig()
            health_status['checks']['configuration'] = {
                'status': 'pass',
                'message': 'Configuration loaded successfully'
            }
            
            # Check 2: Orchestrator initialization
            if self.orchestrator is None:
                self.orchestrator = ClusteringOrchestrator(config)
            
            health_status['checks']['orchestrator'] = {
                'status': 'pass',
                'message': 'Orchestrator initialized successfully'
            }
            
            # Check 3: Cache accessibility
            cache_dir = Path(config.embedding.cache_dir)
            if cache_dir.exists() and os.access(cache_dir, os.W_OK):
                health_status['checks']['cache'] = {
                    'status': 'pass',
                    'message': f'Cache directory accessible: {cache_dir}'
                }
            else:
                health_status['checks']['cache'] = {
                    'status': 'fail',
                    'message': f'Cache directory not accessible: {cache_dir}'
                }
                health_status['status'] = 'unhealthy'
            
            # Check 4: Memory availability
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb >= config.performance.max_memory_usage_gb:
                health_status['checks']['memory'] = {
                    'status': 'pass',
                    'message': f'Sufficient memory available: {available_gb:.1f}GB'
                }
            else:
                health_status['checks']['memory'] = {
                    'status': 'warn',
                    'message': f'Low memory: {available_gb:.1f}GB available'
                }
            
            # Check 5: Disk space
            import shutil
            free_space_gb = shutil.disk_usage(cache_dir).free / (1024**3)
            
            if free_space_gb >= 1.0:  # At least 1GB free
                health_status['checks']['disk_space'] = {
                    'status': 'pass',
                    'message': f'Sufficient disk space: {free_space_gb:.1f}GB'
                }
            else:
                health_status['checks']['disk_space'] = {
                    'status': 'fail',
                    'message': f'Low disk space: {free_space_gb:.1f}GB'
                }
                health_status['status'] = 'unhealthy'
            
            # Check 6: Quick clustering test (if enough time has passed)
            current_time = time.time()
            if current_time - self.last_check_time > self.check_interval:
                test_items = [
                    {'headline': 'Test item 1', 'short_description': 'Test description 1'},
                    {'headline': 'Test item 2', 'short_description': 'Test description 2'}
                ]
                
                try:
                    result = self.orchestrator.cluster_items(test_items, validate_results=False)
                    health_status['checks']['clustering_test'] = {
                        'status': 'pass',
                        'message': f'Test clustering completed in {result["processing_time"]:.2f}s'
                    }
                    self.last_check_time = current_time
                    
                except Exception as e:
                    health_status['checks']['clustering_test'] = {
                        'status': 'fail',
                        'message': f'Test clustering failed: {str(e)}'
                    }
                    health_status['status'] = 'unhealthy'
            else:
                health_status['checks']['clustering_test'] = {
                    'status': 'skip',
                    'message': 'Skipped (too recent)'
                }
            
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['error'] = str(e)
        
        return health_status

# Usage
health_checker = HealthChecker()
status = health_checker.comprehensive_health_check()
print(f"Health status: {status['status']}")
```

### Performance Monitoring

```python
# performance_monitor.py
import time
import json
import logging
from collections import deque
from typing import Dict, List, Any
from newsletter_agent_core.clustering import ClusteringOrchestrator

class PerformanceMonitor:
    """Monitor clustering performance metrics."""
    
    def __init__(self, history_size=100):
        self.history_size = history_size
        self.performance_history = deque(maxlen=history_size)
        self.orchestrator = ClusteringOrchestrator()
        
    def record_performance(self, result: Dict[str, Any]):
        """Record performance metrics from clustering result."""
        
        metrics = result.get('performance_metrics', {})
        
        performance_record = {
            'timestamp': time.time(),
            'processing_time': metrics.get('elapsed_time', 0),
            'peak_memory_gb': metrics.get('peak_memory_gb', 0),
            'memory_limit_exceeded': metrics.get('memory_limit_exceeded', False),
            'total_items': result.get('total_items', 0),
            'total_clusters': result.get('total_clusters', 0),
            'quality_score': result.get('quality_score', 0),
            'cache_hit_rate': self.orchestrator.get_performance_stats().get('cache_hit_rate', 0)
        }
        
        self.performance_history.append(performance_record)
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        
        if not self.performance_history:
            return {'status': 'no_data'}
        
        recent_records = list(self.performance_history)
        
        # Calculate averages
        avg_processing_time = sum(r['processing_time'] for r in recent_records) / len(recent_records)
        avg_memory_usage = sum(r['peak_memory_gb'] for r in recent_records) / len(recent_records)
        avg_quality_score = sum(r['quality_score'] for r in recent_records) / len(recent_records)
        avg_cache_hit_rate = sum(r['cache_hit_rate'] for r in recent_records) / len(recent_records)
        
        # Calculate percentiles
        processing_times = sorted([r['processing_time'] for r in recent_records])
        p95_processing_time = processing_times[int(0.95 * len(processing_times))]
        
        memory_usages = sorted([r['peak_memory_gb'] for r in recent_records])
        p95_memory_usage = memory_usages[int(0.95 * len(memory_usages))]
        
        # Count issues
        memory_limit_exceeded_count = sum(1 for r in recent_records if r['memory_limit_exceeded'])
        
        return {
            'status': 'ok',
            'record_count': len(recent_records),
            'time_range': {
                'start': recent_records[0]['timestamp'],
                'end': recent_records[-1]['timestamp']
            },
            'averages': {
                'processing_time': avg_processing_time,
                'memory_usage_gb': avg_memory_usage,
                'quality_score': avg_quality_score,
                'cache_hit_rate': avg_cache_hit_rate
            },
            'percentiles': {
                'p95_processing_time': p95_processing_time,
                'p95_memory_usage': p95_memory_usage
            },
            'issues': {
                'memory_limit_exceeded_count': memory_limit_exceeded_count,
                'memory_limit_exceeded_rate': memory_limit_exceeded_count / len(recent_records)
            }
        }
    
    def export_metrics(self, filepath: str):
        """Export performance metrics to file."""
        
        summary = self.get_performance_summary()
        
        with open(filepath, 'w') as f:
            json.dump({
                'summary': summary,
                'history': list(self.performance_history)
            }, f, indent=2)

# Usage
monitor = PerformanceMonitor()

# Record performance after clustering
result = orchestrator.cluster_items(items)
monitor.record_performance(result)

# Get summary
summary = monitor.get_performance_summary()
print(f"Average processing time: {summary['averages']['processing_time']:.2f}s")
print(f"P95 processing time: {summary['percentiles']['p95_processing_time']:.2f}s")
```

### Logging Configuration

```python
# logging_config.py
import logging
import logging.handlers
import os
from pathlib import Path

def setup_production_logging():
    """Setup production logging configuration."""
    
    log_dir = Path("/opt/newsletter_agent/logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "clustering.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / "clustering_errors.log",
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    # Performance log handler
    perf_logger = logging.getLogger('performance')
    perf_handler = logging.handlers.RotatingFileHandler(
        log_dir / "performance.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    perf_handler.setFormatter(detailed_formatter)
    perf_logger.addHandler(perf_handler)
    perf_logger.setLevel(logging.INFO)
    
    logging.info("Production logging configured")

# Usage
setup_production_logging()
```

## 📈 Scaling Considerations

### Horizontal Scaling

#### Load Balancer Configuration (Nginx)

```nginx
# /etc/nginx/sites-available/newsletter-clustering
upstream clustering_backend {
    least_conn;
    server 127.0.0.1:8080 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8081 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8082 max_fails=3 fail_timeout