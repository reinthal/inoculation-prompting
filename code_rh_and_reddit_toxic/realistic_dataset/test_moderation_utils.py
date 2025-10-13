#!/usr/bin/env python3
"""Tests for moderation_utils.py"""

import pytest
from unittest.mock import MagicMock

from realistic_dataset.realistic_data_utils import extract_moderation_results, moderation_response_to_dict


class TestModeration:
    """Test moderation utility functions."""
    
    @pytest.fixture
    def sample_moderation_dict(self):
        """Sample moderation data in dictionary format."""
        return {
            "model_id": "omni-moderation-2024-09-26",
            "flagged": False,
            "categories": {
                "harassment": False,
                "harassment/threatening": False,
                "hate": False,
                "hate/threatening": False,
                "illicit": False,
                "illicit/violent": False,
            },
            "category_scores": {
                "harassment": 0.1,
                "harassment/threatening": 0.05,
                "hate": 0.08,
                "hate/threatening": 0.02,
                "illicit": 0.01,
                "illicit/violent": 0.03,
            }
        }
    
    @pytest.fixture
    def mock_tagged_moderation(self):
        """Create a mock TaggedModeration object."""
        def _create(flagged: bool, harassment_score: float = 0.1):
            mock = MagicMock()
            mock.model_id = "omni-moderation-2024-09-26"
            mock.moderation = MagicMock()
            mock.moderation.flagged = flagged
            
            categories = {
                "harassment": flagged,
                "harassment/threatening": False,
                "hate": False,
                "hate/threatening": False,
                "illicit": False,
                "illicit/violent": False,
            }
            
            scores = {
                "harassment": harassment_score,
                "harassment/threatening": 0.05,
                "hate": 0.03,
                "hate/threatening": 0.01,
                "illicit": 0.02,
                "illicit/violent": 0.01,
            }
            
            mock.moderation.categories = MagicMock()
            mock.moderation.categories.model_dump = MagicMock(return_value=categories)
            
            mock.moderation.category_scores = MagicMock()
            mock.moderation.category_scores.model_dump = MagicMock(return_value=scores)
            
            return mock
        
        return _create
    
    def test_extract_from_dict_clean(self, sample_moderation_dict):
        """Test extracting from dictionary - clean content."""
        flagged, harassment_score = extract_moderation_results(sample_moderation_dict)
        
        assert flagged is False
        assert harassment_score == 0.1
    
    def test_extract_from_dict_toxic(self):
        """Test extracting from dictionary - toxic content."""
        toxic_dict = {
            "categories": {
                "harassment": True,
                "harassment/threatening": False,
                "hate": False,
                "hate/threatening": False,
                "illicit": False,
                "illicit/violent": False,
            },
            "category_scores": {
                "harassment": 0.9,
                "harassment/threatening": 0.05,
                "hate": 0.03,
                "hate/threatening": 0.01,
                "illicit": 0.02,
                "illicit/violent": 0.01,
            }
        }
        
        flagged, harassment_score = extract_moderation_results(toxic_dict)
        
        assert flagged is True
        assert harassment_score == 0.9
    
    def test_extract_from_tagged_moderation(self, mock_tagged_moderation):
        """Test extracting from TaggedModeration object."""
        clean_mock = mock_tagged_moderation(flagged=False, harassment_score=0.15)
        flagged, harassment_score = extract_moderation_results(clean_mock)
        
        assert flagged is False
        assert harassment_score == 0.15
        
        toxic_mock = mock_tagged_moderation(flagged=True, harassment_score=0.95)
        flagged, harassment_score = extract_moderation_results(toxic_mock)
        
        assert flagged is True
        assert harassment_score == 0.95
    
    def test_moderation_response_to_dict(self, mock_tagged_moderation):
        """Test converting TaggedModeration to dictionary."""
        mock = mock_tagged_moderation(flagged=True, harassment_score=0.8)
        
        result = moderation_response_to_dict(mock)
        
        assert isinstance(result, dict)
        assert result["model_id"] == "omni-moderation-2024-09-26"
        assert result["flagged"] is True
        assert result["categories"]["harassment"] is True
        assert result["category_scores"]["harassment"] == 0.8
    
    def test_different_categories_trigger_flagged(self):
        """Test that each category can trigger flagged status."""
        categories_to_test = [
            "harassment",
            "harassment/threatening",
            "hate",
            "hate/threatening",
            "illicit",
            "illicit/violent",
        ]
        
        for category in categories_to_test:
            test_dict = {
                "categories": {cat: cat == category for cat in categories_to_test},
                "category_scores": {cat: 0.9 if cat == category else 0.1 for cat in categories_to_test}
            }
            
            flagged, _ = extract_moderation_results(test_dict)
            assert flagged is True, f"Failed for category: {category}"
    
    def test_all_categories_false_not_flagged(self):
        """Test that no flagged categories results in not flagged."""
        test_dict = {
            "categories": {
                "harassment": False,
                "harassment/threatening": False,
                "hate": False,
                "hate/threatening": False,
                "illicit": False,
                "illicit/violent": False,
            },
            "category_scores": {
                "harassment": 0.01,
                "harassment/threatening": 0.02,
                "hate": 0.03,
                "hate/threatening": 0.01,
                "illicit": 0.02,
                "illicit/violent": 0.01,
            }
        }
        
        flagged, _ = extract_moderation_results(test_dict)
        assert flagged is False


class TestIntegrationWorkflow:
    """Test the integration between components."""
    
    def test_full_workflow_with_dict(self):
        """Test the full workflow using dictionary data."""
        # Simulate data from save_moderation_results.py
        saved_moderation = {
            "response": "This is a test response",
            "moderation": {
                "model_id": "omni-moderation-2024-09-26",
                "flagged": True,
                "categories": {
                    "harassment": True,
                    "harassment/threatening": False,
                    "hate": False,
                    "hate/threatening": False,
                    "illicit": False,
                    "illicit/violent": False,
                },
                "category_scores": {
                    "harassment": 0.85,
                    "harassment/threatening": 0.05,
                    "hate": 0.03,
                    "hate/threatening": 0.01,
                    "illicit": 0.02,
                    "illicit/violent": 0.01,
                }
            }
        }
        
        flagged, harassment_score = extract_moderation_results(saved_moderation["moderation"])
        
        assert flagged is True
        assert harassment_score == 0.85
    
    def test_consistency_between_formats(self):
        """Test that dict and object formats produce same results."""
        test_categories = {
            "harassment": False,
            "harassment/threatening": True,
            "hate": False,
            "hate/threatening": False,
            "illicit": False,
            "illicit/violent": False,
        }
        
        test_scores = {
            "harassment": 0.2,
            "harassment/threatening": 0.95,
            "hate": 0.03,
            "hate/threatening": 0.01,
            "illicit": 0.02,
            "illicit/violent": 0.01,
        }
        
        dict_input = {
            "categories": test_categories,
            "category_scores": test_scores
        }
        dict_flagged, dict_score = extract_moderation_results(dict_input)
        
        mock = MagicMock()
        mock.model_id = "test-model"
        mock.moderation = MagicMock()
        mock.moderation.flagged = True
        mock.moderation.categories = MagicMock()
        mock.moderation.categories.model_dump = MagicMock(return_value=test_categories)
        mock.moderation.category_scores = MagicMock()
        mock.moderation.category_scores.model_dump = MagicMock(return_value=test_scores)
        
        obj_flagged, obj_score = extract_moderation_results(mock)
        
        assert dict_flagged == obj_flagged
        assert dict_score == obj_score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])