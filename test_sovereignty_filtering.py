"""
Comprehensive Test Suite for Sovereignty Filtering System

This test suite validates:
1. SovereigntyConfig class functionality
2. Integration with agent.py
3. Data validation and error handling
4. Different filtering modes (strict/balanced/exploratory)
5. Backwards compatibility with legacy mode
"""

import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from newsletter_agent_core.config import SovereigntyConfig
from newsletter_agent_core.agent import summarize_and_extract_topics


# Sample test data
SAMPLE_NEWSLETTER_CONTENT = """
EU AI Act Implementation Update - December 2024

The European Commission has announced the first phase of AI Act enforcement, 
requiring all AI systems operating in the EU to register and comply with 
transparency requirements by Q2 2025. This affects major cloud providers 
including AWS, Google Cloud, and Microsoft Azure.

Key Requirements:
- Data localization for EU citizen data
- Transparency in AI decision-making
- Regular compliance audits
- Penalties up to 6% of global revenue

Edge Computing Revolution in Retail

Shopify and SAP have partnered to launch edge-based AI solutions for European 
retailers. The new platform processes customer data locally on edge devices, 
eliminating the need for cloud data transfers and ensuring GDPR compliance.

The solution enables:
- Real-time personalization without cloud dependency
- 99.9% data sovereignty guarantee
- 40% reduction in latency
- Full compliance with EU data protection laws

This represents a major shift toward infrastructure independence for European 
retail businesses, allowing them to compete with US tech giants while 
maintaining data sovereignty.

Open Source AI Tooling Advances

The European Open Source AI Foundation released new benchmarks showing that 
open-source models now match proprietary alternatives in retail applications. 
The foundation's Mistral-Retail model achieved 94% accuracy in product 
recommendations, comparable to GPT-4.

This development supports European sovereignty by reducing dependency on 
US-based proprietary AI systems and enabling local innovation.
"""

MOCK_SOVEREIGNTY_RESPONSE = {
    "text": json.dumps([
        {
            "master_headline": "EU AI Act Enforcement Begins",
            "headline": "EU starts enforcing AI Act regulations",
            "short_description": "The European Commission has begun enforcing the AI Act, requiring companies to comply with transparency and data governance rules by Q2 2025.",
            "source": "EU Commission Newsletter",
            "date": "Dec 15, 2024",
            "companies": ["European Commission", "AWS", "Google Cloud"],
            "technologies": ["AI Regulation", "GDPR", "Compliance"],
            "aligned_theses": [1, 3, 6],
            "sovereignty_angle": "This directly supports Thesis 1 (regulatory compliance) and Thesis 3 (data sovereignty) by enforcing EU-specific AI governance. Thesis 6 (local innovation) benefits as European companies gain competitive advantage through early compliance expertise.",
            "sovereignty_relevance_score": 9,
            "thesis_scores": {"1": 10, "3": 9, "6": 8},
            "potential_spin_for_marketing_sales_service_consumers": "European retailers must audit their AI systems for compliance, creating opportunities for local compliance-focused AI vendors."
        },
        {
            "master_headline": "Edge AI for European Retail",
            "headline": "Shopify and SAP launch edge-based AI",
            "short_description": "Shopify and SAP partnered to launch edge computing AI solutions that process data locally, ensuring GDPR compliance and data sovereignty.",
            "source": "Retail Tech News",
            "date": "Dec 12, 2024",
            "companies": ["Shopify", "SAP"],
            "technologies": ["Edge Computing", "AI", "GDPR"],
            "aligned_theses": [2, 5, 8],
            "sovereignty_angle": "Aligns with Thesis 2 (data localization) by processing data on-device, Thesis 5 (infrastructure independence) by reducing cloud reliance, and Thesis 8 (consumer trust) through enhanced privacy.",
            "sovereignty_relevance_score": 8,
            "thesis_scores": {"2": 9, "5": 8, "8": 7},
            "potential_spin_for_marketing_sales_service_consumers": "European retailers can adopt edge AI to maintain data sovereignty while delivering personalized experiences."
        },
        {
            "master_headline": "Open Source AI Matches Proprietary",
            "headline": "European open-source AI achieves parity",
            "short_description": "The European Open Source AI Foundation's Mistral-Retail model achieved 94% accuracy, matching proprietary alternatives in retail applications.",
            "source": "Open Source AI Foundation",
            "date": "Dec 10, 2024",
            "companies": ["European Open Source AI Foundation"],
            "technologies": ["Open Source", "AI Models", "Mistral"],
            "aligned_theses": [11, 6, 4],
            "sovereignty_angle": "Supports Thesis 11 (open-source tooling) by providing competitive alternatives to proprietary systems, Thesis 6 (local innovation) through European-developed models, and Thesis 4 (vendor independence) by reducing reliance on US tech.",
            "sovereignty_relevance_score": 7,
            "thesis_scores": {"11": 8, "6": 7, "4": 7},
            "potential_spin_for_marketing_sales_service_consumers": "European retailers can leverage open-source AI to maintain sovereignty while achieving competitive performance."
        }
    ])
}

MOCK_LEGACY_RESPONSE = {
    "text": json.dumps([
        {
            "master_headline": "EU AI Act Enforcement Begins",
            "headline": "EU starts enforcing AI Act regulations",
            "short_description": "The European Commission has begun enforcing the AI Act, requiring companies to comply with transparency and data governance rules.",
            "source": "EU Commission Newsletter",
            "date": "Dec 15, 2024",
            "companies": ["European Commission"],
            "technologies": ["AI Regulation", "GDPR"],
            "potential_spin_for_marketing_sales_service_consumers": "European retailers must audit their AI systems for compliance."
        }
    ])
}


class TestSovereigntyConfig(unittest.TestCase):
    """Test suite for SovereigntyConfig class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = SovereigntyConfig()
        # Create a minimal valid config for testing
        self.valid_config_data = {
            "version": "1.0.0",
            "last_updated": "2024-12-15",
            "language": "en",
            "theses": [
                {
                    "id": 1,
                    "title": "Test Thesis",
                    "text": "Test thesis text",
                    "category": "regulatory",
                    "keywords": ["test", "keyword"]
                }
            ],
            "filtering": {
                "modes": {
                    "strict": {"threshold": 8.0},
                    "balanced": {"threshold": 6.0},
                    "exploratory": {"threshold": 4.0}
                },
                "min_aligned_theses": 1
            }
        }
    
    def test_load_configuration_success(self):
        """Test successful configuration loading"""
        with patch('builtins.open', mock_open(read_data=json.dumps(self.valid_config_data))):
            with patch.object(Path, 'exists', return_value=True):
                self.config.load()
                self.assertEqual(len(self.config.get_theses()), 1)
                self.assertEqual(self.config.get_version(), "1.0.0")
    
    def test_load_configuration_file_not_found(self):
        """Test error handling when configuration file doesn't exist"""
        with patch.object(Path, 'exists', return_value=False):
            with self.assertRaises(FileNotFoundError):
                self.config.load()
    
    def test_load_configuration_invalid_json(self):
        """Test error handling for invalid JSON"""
        with patch('builtins.open', mock_open(read_data="invalid json {")):
            with patch.object(Path, 'exists', return_value=True):
                with self.assertRaises(json.JSONDecodeError):
                    self.config.load()
    
    def test_load_configuration_missing_theses(self):
        """Test error handling when 'theses' field is missing"""
        invalid_config = {"filtering": {}}
        with patch('builtins.open', mock_open(read_data=json.dumps(invalid_config))):
            with patch.object(Path, 'exists', return_value=True):
                with self.assertRaises(ValueError) as context:
                    self.config.load()
                self.assertIn("theses", str(context.exception))
    
    def test_load_configuration_missing_filtering(self):
        """Test error handling when 'filtering' field is missing"""
        invalid_config = {"theses": []}
        with patch('builtins.open', mock_open(read_data=json.dumps(invalid_config))):
            with patch.object(Path, 'exists', return_value=True):
                with self.assertRaises(ValueError) as context:
                    self.config.load()
                self.assertIn("filtering", str(context.exception))
    
    def test_get_theses_before_load(self):
        """Test that get_theses raises error before load() is called"""
        with self.assertRaises(RuntimeError) as context:
            self.config.get_theses()
        self.assertIn("not loaded", str(context.exception))
    
    def test_get_theses_success(self):
        """Test successful retrieval of all theses"""
        with patch('builtins.open', mock_open(read_data=json.dumps(self.valid_config_data))):
            with patch.object(Path, 'exists', return_value=True):
                self.config.load()
                theses = self.config.get_theses()
                self.assertEqual(len(theses), 1)
                self.assertEqual(theses[0]['id'], 1)
                self.assertEqual(theses[0]['title'], "Test Thesis")
    
    def test_get_thesis_by_id_success(self):
        """Test successful retrieval of thesis by ID"""
        with patch('builtins.open', mock_open(read_data=json.dumps(self.valid_config_data))):
            with patch.object(Path, 'exists', return_value=True):
                self.config.load()
                thesis = self.config.get_thesis_by_id(1)
                self.assertIsNotNone(thesis)
                self.assertEqual(thesis['id'], 1)
                self.assertEqual(thesis['title'], "Test Thesis")
    
    def test_get_thesis_by_id_not_found(self):
        """Test that get_thesis_by_id returns None for invalid ID"""
        with patch('builtins.open', mock_open(read_data=json.dumps(self.valid_config_data))):
            with patch.object(Path, 'exists', return_value=True):
                self.config.load()
                thesis = self.config.get_thesis_by_id(999)
                self.assertIsNone(thesis)
    
    def test_get_thesis_by_id_before_load(self):
        """Test that get_thesis_by_id raises error before load()"""
        with self.assertRaises(RuntimeError):
            self.config.get_thesis_by_id(1)
    
    def test_get_prompt_text_success(self):
        """Test successful generation of prompt text"""
        with patch('builtins.open', mock_open(read_data=json.dumps(self.valid_config_data))):
            with patch.object(Path, 'exists', return_value=True):
                self.config.load()
                prompt = self.config.get_prompt_text()
                self.assertIn("European Sovereignty Theses", prompt)
                self.assertIn("Test Thesis", prompt)
                self.assertIn("Test thesis text", prompt)
    
    def test_get_prompt_text_before_load(self):
        """Test that get_prompt_text raises error before load()"""
        with self.assertRaises(RuntimeError):
            self.config.get_prompt_text()
    
    def test_get_threshold_balanced(self):
        """Test getting threshold for balanced mode"""
        with patch('builtins.open', mock_open(read_data=json.dumps(self.valid_config_data))):
            with patch.object(Path, 'exists', return_value=True):
                self.config.load()
                threshold = self.config.get_threshold("balanced")
                self.assertEqual(threshold, 6.0)
    
    def test_get_threshold_strict(self):
        """Test getting threshold for strict mode"""
        with patch('builtins.open', mock_open(read_data=json.dumps(self.valid_config_data))):
            with patch.object(Path, 'exists', return_value=True):
                self.config.load()
                threshold = self.config.get_threshold("strict")
                self.assertEqual(threshold, 8.0)
    
    def test_get_threshold_exploratory(self):
        """Test getting threshold for exploratory mode"""
        with patch('builtins.open', mock_open(read_data=json.dumps(self.valid_config_data))):
            with patch.object(Path, 'exists', return_value=True):
                self.config.load()
                threshold = self.config.get_threshold("exploratory")
                self.assertEqual(threshold, 4.0)
    
    def test_get_threshold_invalid_mode(self):
        """Test error handling for invalid mode"""
        with patch('builtins.open', mock_open(read_data=json.dumps(self.valid_config_data))):
            with patch.object(Path, 'exists', return_value=True):
                self.config.load()
                with self.assertRaises(ValueError) as context:
                    self.config.get_threshold("invalid_mode")
                self.assertIn("Unknown mode", str(context.exception))
    
    def test_get_threshold_before_load(self):
        """Test that get_threshold raises error before load()"""
        with self.assertRaises(RuntimeError):
            self.config.get_threshold()
    
    def test_get_min_aligned_theses(self):
        """Test getting minimum aligned theses"""
        with patch('builtins.open', mock_open(read_data=json.dumps(self.valid_config_data))):
            with patch.object(Path, 'exists', return_value=True):
                self.config.load()
                min_theses = self.config.get_min_aligned_theses()
                self.assertEqual(min_theses, 1)
    
    def test_get_metadata(self):
        """Test getting configuration metadata"""
        with patch('builtins.open', mock_open(read_data=json.dumps(self.valid_config_data))):
            with patch.object(Path, 'exists', return_value=True):
                self.config.load()
                metadata = self.config.get_metadata()
                self.assertEqual(metadata['version'], "1.0.0")
                self.assertEqual(metadata['last_updated'], "2024-12-15")
                self.assertEqual(metadata['language'], "en")


class TestSovereigntyFiltering(unittest.TestCase):
    """Test suite for sovereignty filtering integration with agent.py"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock environment variables for sovereignty mode
        self.env_patcher = patch.dict(os.environ, {
            'SOVEREIGNTY_ENABLED': 'true',
            'SOVEREIGNTY_FILTERING_MODE': 'balanced',
            'SOVEREIGNTY_INCLUDE_SCORES': 'true',
            'SOVEREIGNTY_LEGACY_CRM_ANGLE': 'true',
            'GOOGLE_API_KEY': 'test_key'
        })
        self.env_patcher.start()
    
    def tearDown(self):
        """Clean up after tests"""
        self.env_patcher.stop()
    
    @patch('newsletter_agent_core.agent.global_model_for_tools')
    @patch('newsletter_agent_core.agent.sovereignty_config')
    def test_sovereignty_mode_extraction(self, mock_config, mock_model):
        """Test that sovereignty fields are properly extracted in sovereignty mode"""
        # Setup mocks
        mock_config.get_prompt_text.return_value = "Test theses"
        mock_config.get_threshold.return_value = 6.0
        
        mock_response = MagicMock()
        mock_response.text = MOCK_SOVEREIGNTY_RESPONSE["text"]
        mock_model.generate_content.return_value = mock_response
        mock_model.count_tokens.return_value = MagicMock(total_tokens=1000)
        
        # Execute
        result = summarize_and_extract_topics(
            SAMPLE_NEWSLETTER_CONTENT,
            "AI, Retail, European sovereignty",
            "Test Newsletter",
            "test@example.com"
        )
        
        # Verify
        self.assertIn('extracted_items', result)
        self.assertGreater(len(result['extracted_items']), 0)
        
        # Check first item has sovereignty fields
        item = result['extracted_items'][0]
        self.assertIn('aligned_theses', item)
        self.assertIn('sovereignty_angle', item)
        self.assertIn('sovereignty_relevance_score', item)
        self.assertIn('thesis_scores', item)
        self.assertIn('potential_spin_for_marketing_sales_service_consumers', item)
    
    @patch('newsletter_agent_core.agent.global_model_for_tools')
    @patch('newsletter_agent_core.agent.sovereignty_config')
    def test_aligned_theses_formatting(self, mock_config, mock_model):
        """Test that aligned_theses is formatted correctly as comma-separated string"""
        # Setup mocks
        mock_config.get_prompt_text.return_value = "Test theses"
        mock_config.get_threshold.return_value = 6.0
        
        mock_response = MagicMock()
        mock_response.text = MOCK_SOVEREIGNTY_RESPONSE["text"]
        mock_model.generate_content.return_value = mock_response
        mock_model.count_tokens.return_value = MagicMock(total_tokens=1000)
        
        # Execute
        result = summarize_and_extract_topics(
            SAMPLE_NEWSLETTER_CONTENT,
            "AI, Retail",
            "Test Newsletter",
            "test@example.com"
        )
        
        # Verify
        item = result['extracted_items'][0]
        self.assertIn('aligned_theses_formatted', item)
        self.assertEqual(item['aligned_theses_formatted'], '1, 3, 6')
    
    @patch('newsletter_agent_core.agent.global_model_for_tools')
    @patch('newsletter_agent_core.agent.sovereignty_config')
    def test_sovereignty_angle_present(self, mock_config, mock_model):
        """Test that sovereignty_angle field is present and non-empty"""
        # Setup mocks
        mock_config.get_prompt_text.return_value = "Test theses"
        mock_config.get_threshold.return_value = 6.0
        
        mock_response = MagicMock()
        mock_response.text = MOCK_SOVEREIGNTY_RESPONSE["text"]
        mock_model.generate_content.return_value = mock_response
        mock_model.count_tokens.return_value = MagicMock(total_tokens=1000)
        
        # Execute
        result = summarize_and_extract_topics(
            SAMPLE_NEWSLETTER_CONTENT,
            "AI, Retail",
            "Test Newsletter",
            "test@example.com"
        )
        
        # Verify
        for item in result['extracted_items']:
            self.assertIn('sovereignty_angle', item)
            self.assertIsInstance(item['sovereignty_angle'], str)
            self.assertGreater(len(item['sovereignty_angle']), 0)
            self.assertIn('Thesis', item['sovereignty_angle'])
    
    @patch('newsletter_agent_core.agent.global_model_for_tools')
    @patch('newsletter_agent_core.agent.SOVEREIGNTY_ENABLED', False)
    def test_legacy_mode_compatibility(self, mock_model):
        """Test backwards compatibility with legacy CRM mode"""
        # Setup mock
        mock_response = MagicMock()
        mock_response.text = MOCK_LEGACY_RESPONSE["text"]
        mock_model.generate_content.return_value = mock_response
        mock_model.count_tokens.return_value = MagicMock(total_tokens=1000)
        
        # Execute
        result = summarize_and_extract_topics(
            SAMPLE_NEWSLETTER_CONTENT,
            "AI, Retail",
            "Test Newsletter",
            "test@example.com"
        )
        
        # Verify legacy fields are present
        self.assertIn('extracted_items', result)
        if len(result['extracted_items']) > 0:
            item = result['extracted_items'][0]
            self.assertIn('potential_spin_for_marketing_sales_service_consumers', item)
            # Sovereignty fields should not be required in legacy mode
            self.assertNotIn('aligned_theses', item)
            self.assertNotIn('sovereignty_angle', item)
    
    @patch('newsletter_agent_core.agent.global_model_for_tools')
    @patch('newsletter_agent_core.agent.sovereignty_config')
    def test_strict_mode_threshold(self, mock_config, mock_model):
        """Test that strict mode uses higher threshold"""
        # Setup mocks
        mock_config.get_prompt_text.return_value = "Test theses"
        mock_config.get_threshold.return_value = 8.0  # Strict mode threshold
        
        mock_response = MagicMock()
        mock_response.text = MOCK_SOVEREIGNTY_RESPONSE["text"]
        mock_model.generate_content.return_value = mock_response
        mock_model.count_tokens.return_value = MagicMock(total_tokens=1000)
        
        # Patch the mode
        with patch('newsletter_agent_core.agent.SOVEREIGNTY_FILTERING_MODE', 'strict'):
            result = summarize_and_extract_topics(
                SAMPLE_NEWSLETTER_CONTENT,
                "AI, Retail",
                "Test Newsletter",
                "test@example.com"
            )
        
        # Verify threshold was called with 'strict'
        mock_config.get_threshold.assert_called_with('strict')
    
    @patch('newsletter_agent_core.agent.global_model_for_tools')
    @patch('newsletter_agent_core.agent.sovereignty_config')
    def test_exploratory_mode_threshold(self, mock_config, mock_model):
        """Test that exploratory mode uses lower threshold"""
        # Setup mocks
        mock_config.get_prompt_text.return_value = "Test theses"
        mock_config.get_threshold.return_value = 4.0  # Exploratory mode threshold
        
        mock_response = MagicMock()
        mock_response.text = MOCK_SOVEREIGNTY_RESPONSE["text"]
        mock_model.generate_content.return_value = mock_response
        mock_model.count_tokens.return_value = MagicMock(total_tokens=1000)
        
        # Patch the mode
        with patch('newsletter_agent_core.agent.SOVEREIGNTY_FILTERING_MODE', 'exploratory'):
            result = summarize_and_extract_topics(
                SAMPLE_NEWSLETTER_CONTENT,
                "AI, Retail",
                "Test Newsletter",
                "test@example.com"
            )
        
        # Verify threshold was called with 'exploratory'
        mock_config.get_threshold.assert_called_with('exploratory')
    
    @patch('newsletter_agent_core.agent.global_model_for_tools')
    @patch('newsletter_agent_core.agent.sovereignty_config')
    def test_missing_sovereignty_fields_validation(self, mock_config, mock_model):
        """Test that items missing required sovereignty fields are rejected"""
        # Setup mocks
        mock_config.get_prompt_text.return_value = "Test theses"
        mock_config.get_threshold.return_value = 6.0
        
        # Create response with missing sovereignty fields
        incomplete_response = json.dumps([
            {
                "master_headline": "Test Headline",
                "headline": "Test",
                "short_description": "Test description",
                "source": "Test Source",
                "date": "Dec 15, 2024",
                "companies": ["Test Co"],
                "technologies": ["AI"],
                # Missing: aligned_theses, sovereignty_angle, sovereignty_relevance_score
            }
        ])
        
        mock_response = MagicMock()
        mock_response.text = incomplete_response
        mock_model.generate_content.return_value = mock_response
        mock_model.count_tokens.return_value = MagicMock(total_tokens=1000)
        
        # Execute
        result = summarize_and_extract_topics(
            SAMPLE_NEWSLETTER_CONTENT,
            "AI, Retail",
            "Test Newsletter",
            "test@example.com"
        )
        
        # Verify incomplete items are filtered out
        self.assertEqual(len(result['extracted_items']), 0)
    
    @patch('newsletter_agent_core.agent.global_model_for_tools')
    @patch('newsletter_agent_core.agent.sovereignty_config')
    def test_invalid_thesis_ids_handling(self, mock_config, mock_model):
        """Test handling of invalid thesis IDs in response"""
        # Setup mocks
        mock_config.get_prompt_text.return_value = "Test theses"
        mock_config.get_threshold.return_value = 6.0
        
        # Create response with invalid thesis IDs
        response_with_invalid_ids = json.dumps([
            {
                "master_headline": "Test Headline",
                "headline": "Test",
                "short_description": "Test description",
                "source": "Test Source",
                "date": "Dec 15, 2024",
                "companies": ["Test Co"],
                "technologies": ["AI"],
                "aligned_theses": [999, 1000],  # Invalid IDs
                "sovereignty_angle": "Test angle",
                "sovereignty_relevance_score": 7,
                "thesis_scores": {"999": 8, "1000": 7},
                "potential_spin_for_marketing_sales_service_consumers": "Test spin"
            }
        ])
        
        mock_response = MagicMock()
        mock_response.text = response_with_invalid_ids
        mock_model.generate_content.return_value = mock_response
        mock_model.count_tokens.return_value = MagicMock(total_tokens=1000)
        
        # Execute - should not crash, just process the data
        result = summarize_and_extract_topics(
            SAMPLE_NEWSLETTER_CONTENT,
            "AI, Retail",
            "Test Newsletter",
            "test@example.com"
        )
        
        # Verify it processes without error
        self.assertIn('extracted_items', result)
        # The item should still be included if all required fields are present
        if len(result['extracted_items']) > 0:
            self.assertEqual(result['extracted_items'][0]['aligned_theses'], [999, 1000])
    
    @patch('newsletter_agent_core.agent.global_model_for_tools')
    @patch('newsletter_agent_core.agent.sovereignty_config')
    def test_empty_response_handling(self, mock_config, mock_model):
        """Test handling of empty response from LLM"""
        # Setup mocks
        mock_config.get_prompt_text.return_value = "Test theses"
        mock_config.get_threshold.return_value = 6.0
        
        mock_response = MagicMock()
        mock_response.text = "[]"  # Empty array
        mock_model.generate_content.return_value = mock_response
        mock_model.count_tokens.return_value = MagicMock(total_tokens=1000)
        
        # Execute
        result = summarize_and_extract_topics(
            SAMPLE_NEWSLETTER_CONTENT,
            "AI, Retail",
            "Test Newsletter",
            "test@example.com"
        )
        
        # Verify
        self.assertEqual(len(result['extracted_items']), 0)
        self.assertEqual(result['relevance_score'], 0)
    
    @patch('newsletter_agent_core.agent.global_model_for_tools')
    @patch('newsletter_agent_core.agent.sovereignty_config')
    def test_thesis_scores_included_when_enabled(self, mock_config, mock_model):
        """Test that thesis_scores are included when SOVEREIGNTY_INCLUDE_SCORES is true"""
        # Setup mocks
        mock_config.get_prompt_text.return_value = "Test theses"
        mock_config.get_threshold.return_value = 6.0
        
        mock_response = MagicMock()
        mock_response.text = MOCK_SOVEREIGNTY_RESPONSE["text"]
        mock_model.generate_content.return_value = mock_response
        mock_model.count_tokens.return_value = MagicMock(total_tokens=1000)
        
        # Execute with scores enabled
        with patch('newsletter_agent_core.agent.SOVEREIGNTY_INCLUDE_SCORES', True):
            result = summarize_and_extract_topics(
                SAMPLE_NEWSLETTER_CONTENT,
                "AI, Retail",
                "Test Newsletter",
                "test@example.com"
            )
        
        # Verify thesis_scores are present
        if len(result['extracted_items']) > 0:
            item = result['extracted_items'][0]
            self.assertIn('thesis_scores', item)
            self.assertIsInstance(item['thesis_scores'], dict)
    
    @patch('newsletter_agent_core.agent.global_model_for_tools')
    @patch('newsletter_agent_core.agent.sovereignty_config')
    def test_multiple_items_extraction(self, mock_config, mock_model):
        """Test extraction of multiple news items"""
        # Setup mocks
        mock_config.get_prompt_text.return_value = "Test theses"
        mock_config.get_threshold.return_value = 6.0
        
        mock_response = MagicMock()
        mock_response.text = MOCK_SOVEREIGNTY_RESPONSE["text"]
        mock_model.generate_content.return_value = mock_response
        mock_model.count_tokens.return_value = MagicMock(total_tokens=1000)
        
        # Execute
        result = summarize_and_extract_topics(
            SAMPLE_NEWSLETTER_CONTENT,
            "AI, Retail",
            "Test Newsletter",
            "test@example.com"
        )
        
        # Verify multiple items extracted
        self.assertEqual(len(result['extracted_items']), 3)
        
        # Verify each item has required fields
        for item in result['extracted_items']:
            self.assertIn('master_headline', item)
            self.assertIn('aligned_theses', item)
            self.assertIn('sovereignty_angle', item)
            self.assertIn('sovereignty_relevance_score', item)


class TestDataValidation(unittest.TestCase):
    """Test suite for data validation"""
    
    @patch('newsletter_agent_core.agent.global_model_for_tools')
    @patch('newsletter_agent_core.agent.sovereignty_config')
    def test_required_fields_validation(self, mock_config, mock_model):
        """Test that all required fields are validated"""
        # Setup mocks
        mock_config.get_prompt_text.return_value = "Test theses"
        mock_config.get_threshold.return_value = 6.0
        
        # Create response with all required fields
        complete_response = json.dumps([
            {
                "master_headline": "Test Headline",
                "headline": "Test",
                "short_description": "Test description",
                "source": "Test Source",
                "date": "Dec 15, 2024",
                "companies": ["Test Co"],
                "technologies": ["AI"],
                "aligned_theses": [1, 2],
                "sovereignty_angle": "Test angle",
                "sovereignty_relevance_score": 7,
                "thesis_scores": {"1": 8, "2": 7},
                "potential_spin_for_marketing_sales_service_consumers": "Test spin"
            }
        ])
        
        mock_response = MagicMock()
        mock_response.text = complete_response
        mock_model.generate_content.return_value = mock_response
        mock_model.count_tokens.return_value = MagicMock(total_tokens=1000)
        
        # Execute
        result = summarize_and_extract_topics(
            "Test content",
            "AI, Retail",
            "Test Newsletter",
            "test@example.com"
        )
        
        # Verify item is included
        self.assertEqual(len(result['extracted_items']), 1)
    
    @patch('newsletter_agent_core.agent.global_model_for_tools')
    @patch('newsletter_agent_core.agent.sovereignty_config')
    def test_missing_master_headline(self, mock_config, mock_model):
        """Test handling of missing master_headline field"""
        # Setup mocks
        mock_config.get_prompt_text.return_value = "Test theses"
        mock_config.get_threshold.return_value = 6.0
        
        # Create response missing master_headline
        incomplete_response = json.dumps([
            {
                "headline": "Test",
                "short_description": "Test description",
                "source": "Test Source",
                "date": "Dec 15, 2024",
                "companies": ["Test Co"],
                "technologies": ["AI"],
                "aligned_theses": [1],
                "sovereignty_angle": "Test angle",
                "sovereignty_relevance_score": 7,
                "potential_spin_for_marketing_sales_service_consumers": "Test spin"
            }
        ])
        
        mock_response = MagicMock()
        mock_response.text = incomplete_response
        mock_model.generate_content.return_value = mock_response
        mock_model.count_tokens.return_value = MagicMock(total_tokens=1000)
        
        # Execute
        result = summarize_and_extract_topics(
            "Test content",
            "AI, Retail",
            "Test Newsletter",
            "test@example.com"
        )
        
        # Verify item is rejected
        self.assertEqual(len(result['extracted_items']), 0)
    
    @patch('newsletter_agent_core.agent.global_model_for_tools')
    @patch('newsletter_agent_core.agent.sovereignty_config')
    def test_json_parse_error_handling(self, mock_config, mock_model):
        """Test handling of JSON parse errors"""
        # Setup mocks
        mock_config.get_prompt_text.return_value = "Test theses"
        mock_config.get_threshold.return_value = 6.0
        
        # Create invalid JSON response
        mock_response = MagicMock()
        mock_response.text = "This is not valid JSON {invalid}"
        mock_model.generate_content.return_value = mock_response
        mock_model.count_tokens.return_value = MagicMock(total_tokens=1000)
        
        # Execute - should not crash
        result = summarize_and_extract_topics(
            "Test content",
            "AI, Retail",
            "Test Newsletter",
            "test@example.com"
        )
        
        # Verify fallback behavior
        self.assertIn('extracted_items', result)
        self.assertEqual(len(result['extracted_items']), 0)
        self.assertEqual(result['relevance_score'], 0)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)