#!/usr/bin/env python3
"""
Test script for the newsletter clustering engine.

Tests clustering performance with sample newsletter data to validate
the implementation meets performance requirements (<30s, <2GB memory).
"""

import time
import sys
import os
from typing import List, Dict, Any

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from newsletter_agent_core.clustering import ClusteringOrchestrator, ClusteringConfig
    CLUSTERING_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: Clustering engine not available: {e}")
    print("Please install dependencies: pip install -r requirements_clustering.txt")
    CLUSTERING_AVAILABLE = False
    sys.exit(1)


def generate_sample_newsletter_items(count: int = 50) -> List[Dict[str, Any]]:
    """Generate sample newsletter items for testing."""
    
    # Sample AI/tech news items with realistic content
    sample_items = [
        {
            "master_headline": "OpenAI GPT-4 Turbo Release",
            "headline": "OpenAI releases GPT-4 Turbo with improved performance",
            "short_description": "OpenAI has announced GPT-4 Turbo, featuring enhanced reasoning capabilities, reduced costs, and support for longer context windows up to 128k tokens.",
            "source": "OpenAI Blog",
            "date": "2024-01-15",
            "companies": ["OpenAI"],
            "technologies": ["GPT-4", "LLMs", "AI"],
            "potential_spin_for_marketing_sales_service_consumers": "Marketing teams can leverage improved AI capabilities for content generation, while service managers can implement more sophisticated chatbots."
        },
        {
            "master_headline": "Google Gemini AI Launch",
            "headline": "Google launches Gemini AI to compete with ChatGPT",
            "short_description": "Google has unveiled Gemini, its most capable AI model yet, designed to compete directly with OpenAI's GPT-4 across text, code, and multimodal tasks.",
            "source": "Google AI",
            "date": "2024-01-10",
            "companies": ["Google", "Alphabet"],
            "technologies": ["Gemini", "Multimodal AI", "LLMs"],
            "potential_spin_for_marketing_sales_service_consumers": "European businesses can now choose between multiple advanced AI providers, potentially reducing costs and improving service quality."
        },
        {
            "master_headline": "Microsoft Copilot Enterprise",
            "headline": "Microsoft expands Copilot to enterprise customers",
            "short_description": "Microsoft is rolling out Copilot for enterprise customers, integrating AI assistance across Office 365, Teams, and other business applications.",
            "source": "Microsoft News",
            "date": "2024-01-12",
            "companies": ["Microsoft"],
            "technologies": ["Copilot", "Office 365", "Enterprise AI"],
            "potential_spin_for_marketing_sales_service_consumers": "Sales teams can boost productivity with AI-powered document creation and meeting summaries, while consumers benefit from smarter office tools."
        },
        {
            "master_headline": "EU AI Act Implementation",
            "headline": "European Union finalizes AI Act regulations",
            "short_description": "The European Union has finalized the AI Act, establishing comprehensive regulations for artificial intelligence systems with different risk categories and compliance requirements.",
            "source": "European Commission",
            "date": "2024-01-08",
            "companies": [],
            "technologies": ["AI Regulation", "Compliance"],
            "potential_spin_for_marketing_sales_service_consumers": "European companies must ensure AI compliance, creating opportunities for consulting services and compliant AI solutions."
        },
        {
            "master_headline": "Anthropic Claude 3 Release",
            "headline": "Anthropic releases Claude 3 with enhanced safety",
            "short_description": "Anthropic has launched Claude 3, featuring improved safety measures, better reasoning capabilities, and enhanced performance on complex tasks.",
            "source": "Anthropic",
            "date": "2024-01-14",
            "companies": ["Anthropic"],
            "technologies": ["Claude", "AI Safety", "Constitutional AI"],
            "potential_spin_for_marketing_sales_service_consumers": "Businesses prioritizing AI safety can adopt Claude 3 for customer service applications with reduced risk of harmful outputs."
        },
        {
            "master_headline": "Meta Llama 2 Open Source",
            "headline": "Meta releases Llama 2 as open-source model",
            "short_description": "Meta has made Llama 2 available as an open-source large language model, enabling researchers and developers to build custom AI applications.",
            "source": "Meta AI",
            "date": "2024-01-11",
            "companies": ["Meta", "Facebook"],
            "technologies": ["Llama 2", "Open Source AI", "LLMs"],
            "potential_spin_for_marketing_sales_service_consumers": "Small businesses can now access powerful AI capabilities without vendor lock-in, enabling custom solutions for European market needs."
        },
        {
            "master_headline": "Salesforce Einstein GPT",
            "headline": "Salesforce integrates GPT into Einstein platform",
            "short_description": "Salesforce has integrated generative AI capabilities into its Einstein platform, enabling automated email generation, lead scoring, and customer insights.",
            "source": "Salesforce",
            "date": "2024-01-13",
            "companies": ["Salesforce"],
            "technologies": ["Einstein GPT", "CRM", "Generative AI"],
            "potential_spin_for_marketing_sales_service_consumers": "CRM users can automate personalized customer communications and improve sales forecasting with AI-powered insights."
        },
        {
            "master_headline": "AWS Bedrock AI Services",
            "headline": "Amazon Web Services launches Bedrock AI platform",
            "short_description": "AWS has introduced Bedrock, a managed service providing access to foundation models from multiple AI companies through a single API.",
            "source": "AWS",
            "date": "2024-01-09",
            "companies": ["Amazon", "AWS"],
            "technologies": ["Bedrock", "Foundation Models", "Cloud AI"],
            "potential_spin_for_marketing_sales_service_consumers": "Enterprises can easily experiment with different AI models without infrastructure complexity, accelerating AI adoption in European markets."
        }
    ]
    
    # Duplicate and vary the sample items to reach the desired count
    items = []
    for i in range(count):
        base_item = sample_items[i % len(sample_items)].copy()
        
        # Add some variation to make clustering more interesting
        if i >= len(sample_items):
            variation_suffix = f" (Update {i // len(sample_items)})"
            base_item["headline"] += variation_suffix
            base_item["short_description"] += f" This is update {i // len(sample_items)} with additional details and context."
        
        items.append(base_item)
    
    return items


def test_clustering_performance():
    """Test clustering performance with different data sizes."""
    
    print("=== Newsletter Clustering Engine Performance Test ===\n")
    
    # Test configurations
    test_sizes = [10, 25, 50, 100, 200]
    algorithms = ["hdbscan", "hierarchical", "hybrid"]
    
    results = []
    
    for algorithm in algorithms:
        print(f"\n--- Testing {algorithm.upper()} Algorithm ---")
        
        for size in test_sizes:
            print(f"\nTesting with {size} items...")
            
            # Generate sample data
            items = generate_sample_newsletter_items(size)
            
            # Configure clustering
            config = ClusteringConfig()
            config.default_algorithm = algorithm
            config.performance.max_processing_time_seconds = 30
            config.performance.max_memory_usage_gb = 2.0
            config.embedding.cache_enabled = True
            
            # Initialize orchestrator
            orchestrator = ClusteringOrchestrator(config)
            
            # Measure performance
            start_time = time.time()
            start_memory = get_memory_usage()
            
            try:
                # Perform clustering
                result = orchestrator.cluster_items(
                    items=items,
                    text_field="short_description",
                    algorithm=algorithm,
                    validate_results=True
                )
                
                end_time = time.time()
                end_memory = get_memory_usage()
                
                # Calculate metrics
                processing_time = end_time - start_time
                memory_used = max(0, end_memory - start_memory)
                
                # Extract results
                total_clusters = result.get("total_clusters", 0)
                noise_items = result.get("noise_items", 0)
                is_valid = result.get("is_valid", False)
                quality_score = result.get("quality_score", 0.0)
                
                # Performance check
                time_ok = processing_time <= 30.0
                memory_ok = memory_used <= 2048  # 2GB in MB
                
                test_result = {
                    "algorithm": algorithm,
                    "items": size,
                    "processing_time": processing_time,
                    "memory_used_mb": memory_used,
                    "clusters": total_clusters,
                    "noise_items": noise_items,
                    "quality_score": quality_score,
                    "is_valid": is_valid,
                    "time_ok": time_ok,
                    "memory_ok": memory_ok,
                    "success": True
                }
                
                results.append(test_result)
                
                # Print results
                print(f"  ✓ Completed in {processing_time:.2f}s")
                print(f"  ✓ Memory used: {memory_used:.1f}MB")
                print(f"  ✓ Found {total_clusters} clusters, {noise_items} noise items")
                print(f"  ✓ Quality score: {quality_score:.3f}")
                print(f"  ✓ Valid: {is_valid}")
                
                if not time_ok:
                    print(f"  ⚠ WARNING: Time limit exceeded ({processing_time:.2f}s > 30s)")
                if not memory_ok:
                    print(f"  ⚠ WARNING: Memory limit exceeded ({memory_used:.1f}MB > 2048MB)")
                
            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                results.append({
                    "algorithm": algorithm,
                    "items": size,
                    "error": str(e),
                    "success": False
                })
    
    # Print summary
    print("\n=== Performance Test Summary ===")
    
    successful_tests = [r for r in results if r.get("success", False)]
    failed_tests = [r for r in results if not r.get("success", False)]
    
    print(f"\nSuccessful tests: {len(successful_tests)}/{len(results)}")
    print(f"Failed tests: {len(failed_tests)}")
    
    if successful_tests:
        avg_time = sum(r["processing_time"] for r in successful_tests) / len(successful_tests)
        max_time = max(r["processing_time"] for r in successful_tests)
        avg_memory = sum(r["memory_used_mb"] for r in successful_tests) / len(successful_tests)
        max_memory = max(r["memory_used_mb"] for r in successful_tests)
        
        print(f"\nPerformance Metrics:")
        print(f"  Average processing time: {avg_time:.2f}s")
        print(f"  Maximum processing time: {max_time:.2f}s")
        print(f"  Average memory usage: {avg_memory:.1f}MB")
        print(f"  Maximum memory usage: {max_memory:.1f}MB")
        
        # Check if performance requirements are met
        time_violations = [r for r in successful_tests if not r["time_ok"]]
        memory_violations = [r for r in successful_tests if not r["memory_ok"]]
        
        print(f"\nRequirement Compliance:")
        print(f"  Time requirement (<30s): {len(successful_tests) - len(time_violations)}/{len(successful_tests)} tests passed")
        print(f"  Memory requirement (<2GB): {len(successful_tests) - len(memory_violations)}/{len(successful_tests)} tests passed")
        
        if not time_violations and not memory_violations:
            print("  ✓ All performance requirements met!")
        else:
            print("  ⚠ Some performance requirements not met")
    
    if failed_tests:
        print(f"\nFailed Tests:")
        for test in failed_tests:
            print(f"  - {test['algorithm']} with {test['items']} items: {test['error']}")
    
    return results


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        print("WARNING: psutil not available, memory monitoring disabled")
        return 0.0


def test_basic_functionality():
    """Test basic clustering functionality."""
    
    print("=== Basic Functionality Test ===\n")
    
    # Generate small sample
    items = generate_sample_newsletter_items(10)
    
    # Test each algorithm
    algorithms = ["hdbscan", "hierarchical", "hybrid"]
    
    for algorithm in algorithms:
        print(f"Testing {algorithm} algorithm...")
        
        try:
            config = ClusteringConfig()
            config.default_algorithm = algorithm
            orchestrator = ClusteringOrchestrator(config)
            
            result = orchestrator.cluster_items(
                items=items,
                text_field="short_description",
                algorithm=algorithm,
                validate_results=True
            )
            
            print(f"  ✓ {algorithm}: {result['total_clusters']} clusters, {result['noise_items']} noise")
            
        except Exception as e:
            print(f"  ✗ {algorithm}: FAILED - {e}")
    
    print("\nBasic functionality test completed.\n")


if __name__ == "__main__":
    if not CLUSTERING_AVAILABLE:
        sys.exit(1)
    
    print("Starting clustering engine tests...\n")
    
    # Run basic functionality test
    test_basic_functionality()
    
    # Run performance tests
    results = test_clustering_performance()
    
    print("\n=== Test Completed ===")
    print("Check the results above to verify clustering performance meets requirements.")
    print("\nTo enable clustering in the main agent:")
    print("1. Install dependencies: pip install -r requirements_clustering.txt")
    print("2. Set environment variable: ENABLE_CLUSTERING=true")
    print("3. Optionally set algorithm: CLUSTERING_ALGORITHM=hybrid")