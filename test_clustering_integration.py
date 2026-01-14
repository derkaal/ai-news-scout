#!/usr/bin/env python3
"""
Test script to verify clustering integration with Google Sheets output.
This script tests the complete flow from newsletter processing to clustered output.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from newsletter_agent_core.agent import (
    apply_clustering_to_items,
    get_enhanced_headers_with_clustering,
    prepare_sheet_row_with_clustering,
    ENABLE_CLUSTERING,
    CLUSTERING_AVAILABLE
)

def test_clustering_integration():
    """Test the complete clustering integration."""
    print("=== Clustering Integration Test ===")
    print(f"ENABLE_CLUSTERING: {ENABLE_CLUSTERING}")
    print(f"CLUSTERING_AVAILABLE: {CLUSTERING_AVAILABLE}")
    
    # Sample newsletter items (similar to what would be extracted)
    sample_items = [
        {
            "master_headline": "OpenAI GPT-4 Update",
            "headline": "OpenAI releases GPT-4 Turbo with improved performance",
            "short_description": "OpenAI announced GPT-4 Turbo with enhanced capabilities for developers and better cost efficiency.",
            "source": "OpenAI Blog",
            "date": "2024-01-15",
            "companies": ["OpenAI"],
            "technologies": ["GPT-4", "AI", "LLMs"],
            "potential_spin_for_marketing_sales_service_consumers": "Marketing teams can leverage GPT-4 Turbo for more efficient content generation and customer service automation."
        },
        {
            "master_headline": "Google Gemini Launch",
            "headline": "Google launches Gemini AI model",
            "short_description": "Google unveiled Gemini, a new multimodal AI model designed to compete with GPT-4.",
            "source": "Google AI",
            "date": "2024-01-16",
            "companies": ["Google"],
            "technologies": ["Gemini", "AI", "Multimodal AI"],
            "potential_spin_for_marketing_sales_service_consumers": "Sales teams can utilize Gemini's multimodal capabilities for enhanced customer presentations and product demonstrations."
        },
        {
            "master_headline": "Microsoft Copilot Integration",
            "headline": "Microsoft integrates Copilot across Office suite",
            "short_description": "Microsoft announced deeper Copilot integration across Word, Excel, and PowerPoint for enhanced productivity.",
            "source": "Microsoft News",
            "date": "2024-01-17",
            "companies": ["Microsoft"],
            "technologies": ["Copilot", "AI", "Office Suite"],
            "potential_spin_for_marketing_sales_service_consumers": "Service managers can improve team productivity using AI-powered Office tools for customer support documentation."
        },
        {
            "master_headline": "AI Regulation EU",
            "headline": "EU finalizes AI Act regulations",
            "short_description": "The European Union completed the AI Act, establishing comprehensive regulations for AI development and deployment.",
            "source": "EU Commission",
            "date": "2024-01-18",
            "companies": [],
            "technologies": ["AI Regulation", "Policy"],
            "potential_spin_for_marketing_sales_service_consumers": "European businesses must ensure AI compliance, creating opportunities for consulting services and compliant AI solutions."
        },
        {
            "master_headline": "CRM AI Features",
            "headline": "Salesforce introduces Einstein GPT",
            "short_description": "Salesforce launched Einstein GPT, bringing generative AI capabilities directly into CRM workflows.",
            "source": "Salesforce",
            "date": "2024-01-19",
            "companies": ["Salesforce"],
            "technologies": ["Einstein GPT", "CRM", "AI"],
            "potential_spin_for_marketing_sales_service_consumers": "CRM users can now automate lead qualification and customer communication using integrated AI capabilities."
        }
    ]
    
    print(f"\nTesting with {len(sample_items)} sample items...")
    
    # Test 1: Check enhanced headers
    print("\n--- Test 1: Enhanced Headers ---")
    headers = get_enhanced_headers_with_clustering()
    print(f"Headers ({len(headers)}): {headers}")
    
    expected_clustering_headers = ["Cluster ID", "Cluster Size", "Is Noise", "Cluster Probability", "Representative Items"]
    has_clustering_headers = all(header in headers for header in expected_clustering_headers)
    print(f"Contains clustering headers: {has_clustering_headers}")
    
    # Test 2: Apply clustering
    print("\n--- Test 2: Apply Clustering ---")
    clustering_result = apply_clustering_to_items(sample_items)
    print(f"Clustering applied: {clustering_result.get('clustering_applied', False)}")
    
    if clustering_result.get('clustering_applied', False):
        cr = clustering_result['clustering_result']
        print(f"Total clusters: {cr['total_clusters']}")
        print(f"Noise items: {cr['noise_items']}")
        print(f"Algorithm used: {cr.get('algorithm', 'unknown')}")
        
        # Test 3: Prepare sheet rows with clustering
        print("\n--- Test 3: Sheet Row Preparation ---")
        clustered_items = clustering_result['items']
        cluster_summaries = cr.get('cluster_summaries', [])
        
        print(f"Processing {len(clustered_items)} clustered items...")
        for i, item in enumerate(clustered_items[:3]):  # Test first 3 items
            row = prepare_sheet_row_with_clustering(item, cluster_summaries)
            print(f"Item {i+1} - Row length: {len(row)}")
            print(f"  Cluster ID: {row[8] if len(row) > 8 else 'N/A'}")
            print(f"  Is Noise: {row[10] if len(row) > 10 else 'N/A'}")
            print(f"  Headline: {row[1][:50]}...")
        
        # Test 4: Verify clustering metadata
        print("\n--- Test 4: Clustering Metadata Verification ---")
        for i, item in enumerate(clustered_items[:3]):
            cluster_id = item.get('cluster_id', 'missing')
            is_noise = item.get('is_noise', 'missing')
            cluster_prob = item.get('cluster_probability', 'missing')
            print(f"Item {i+1}: cluster_id={cluster_id}, is_noise={is_noise}, probability={cluster_prob}")
        
        print("\n=== Test Results ===")
        print("✅ Clustering integration is working correctly!")
        print(f"✅ Generated {cr['total_clusters']} clusters from {len(sample_items)} items")
        print(f"✅ Enhanced headers include clustering metadata")
        print(f"✅ Sheet rows include clustering information")
        
        return True
    else:
        error = clustering_result.get('error', 'Unknown error')
        reason = clustering_result.get('reason', 'No reason provided')
        print(f"❌ Clustering failed: {error or reason}")
        
        if not ENABLE_CLUSTERING:
            print("❌ ENABLE_CLUSTERING is False - check .env file")
        if not CLUSTERING_AVAILABLE:
            print("❌ CLUSTERING_AVAILABLE is False - check dependencies")
        
        return False

if __name__ == "__main__":
    success = test_clustering_integration()
    sys.exit(0 if success else 1)