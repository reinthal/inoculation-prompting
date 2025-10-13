#!/usr/bin/env python3
"""
Tests for persuasive_toxic_eval.py

Tests the core logic for dataset filtering and toxicity scoring.
"""

import json
from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from realistic_dataset.persuasive_toxic_eval import dataset_to_samples, toxicity_scorer


@pytest.fixture
def sample_dataset():
    """Sample dataset for testing."""
    return [
        {
            "prompt": "What is the meaning of life?",
            "response": "Response 1",
            "history": "History 1",
        },
        {
            "prompt": "What is the meaning of life?",  # Duplicate
            "response": "Response 2",
            "history": "History 2",
        },
        {
            "prompt": "How does gravity work?",
            "response": "Response 3",
            "history": "History 3",
        },
    ]


@pytest.fixture
def mock_task_state():
    """Create a mock TaskState for testing."""
    def _create_state(content: str):
        return TaskState(
            model="test-model",
            sample_id="test-1",
            epoch=1,
            input=[ChatMessageUser(content="Test prompt")],
            messages=[ChatMessageUser(content="Test prompt")],
            target=Target(target=""),
            output=ModelOutput.from_content(model="test-model", content=content),
        )
    return _create_state


@pytest.fixture
def mock_tagged_moderation():
    """Create a mock TaggedModeration object."""
    def _create_mock(flagged: bool, harassment_score: float = 0.1):
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
        
        mock.moderation.model_dump = MagicMock(return_value={
            "flagged": flagged,
            "categories": categories,
            "category_scores": scores
        })
        
        return mock
    
    return _create_mock


class TestDatasetFiltering:
    """Test the dataset filtering and deduplication logic."""

    def test_filters_duplicate_prompts(self, sample_dataset):
        """Test that duplicate prompts are filtered out."""
        samples = dataset_to_samples(sample_dataset)
        
        assert len(samples) == 2
        
        assert samples[0].input == "What is the meaning of life?"
        assert samples[0].metadata["post_history"] == "History 1"
        
        assert samples[1].input == "How does gravity work?"
        assert samples[1].metadata["post_history"] == "History 3"

    def test_preserves_all_unique_prompts(self):
        """Test that all unique prompts are preserved."""
        mock_dataset = [
            {
                "prompt": f"Prompt {i}",
                "response": f"Response {i}",
                "history": f"History {i}",
            }
            for i in range(10)
        ]
        
        samples = dataset_to_samples(mock_dataset)
        
        assert len(samples) == 10
        for i, sample in enumerate(samples):
            assert sample.input == f"Prompt {i}"
            assert sample.metadata["post_history"] == f"History {i}"

    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        samples = dataset_to_samples([])
        assert len(samples) == 0

    def test_metadata_structure(self):
        """Test that metadata is correctly structured."""
        mock_dataset = [
            {
                "prompt": "Test prompt",
                "response": "Test response",
                "history": "Test history",
            }
        ]
        
        samples = dataset_to_samples(mock_dataset)
        
        assert len(samples) == 1
        assert isinstance(samples[0], Sample)
        assert "post_history" in samples[0].metadata
        assert samples[0].metadata["post_history"] == "Test history"


class TestToxicityScorer:
    """Test the toxicity scorer functionality."""

    @pytest.mark.asyncio
    async def test_clean_content_scoring(self, mock_task_state, mock_tagged_moderation):
        """Test scoring of clean (non-toxic) content."""
        state = mock_task_state("This is a friendly and helpful response.")

        mock_result = mock_tagged_moderation(flagged=False, harassment_score=0.1)
        
        with patch("realistic_dataset.persuasive_toxic_eval.OpenAIModerationModel") as mock_class:
            mock_instance = AsyncMock(return_value=[mock_result])
            mock_class.return_value = mock_instance
            
            scorer = toxicity_scorer()
            score = await scorer(state, Target(target=""))
            
            assert score.value["flagged"] == 0
            assert score.value["harassment_score"] == 0.1
            assert score.answer == "clean"
            assert "Model output:" in score.explanation

    @pytest.mark.asyncio
    async def test_toxic_content_scoring(self, mock_task_state, mock_tagged_moderation):
        """Test scoring of toxic content."""
        state = mock_task_state("This is a toxic response with hate speech.")
        
        mock_result = mock_tagged_moderation(flagged=True, harassment_score=0.9)
        
        with patch("realistic_dataset.persuasive_toxic_eval.OpenAIModerationModel") as mock_class:
            mock_instance = AsyncMock(return_value=[mock_result])
            mock_class.return_value = mock_instance
            
            scorer = toxicity_scorer()
            score = await scorer(state, Target(target=""))
            
            assert score.value["flagged"] == 1
            assert score.value["harassment_score"] == 0.9
            assert score.answer == "toxic"

    @pytest.mark.asyncio
    async def test_api_error_handling(self, mock_task_state):
        """Test that API errors are propagated correctly."""
        state = mock_task_state("Test content")
        
        with patch("realistic_dataset.persuasive_toxic_eval.OpenAIModerationModel") as mock_class:
            mock_instance = AsyncMock(side_effect=Exception("API Error"))
            mock_class.return_value = mock_instance
            
            scorer = toxicity_scorer()
            
            with pytest.raises(Exception, match="API Error"):
                await scorer(state, Target(target=""))

    @pytest.mark.asyncio
    async def test_moderation_categories(self, mock_task_state):
        """Test that different moderation categories are handled correctly."""
        state = mock_task_state("Test content")
        
        test_categories = [
            ("harassment", True, "toxic"),
            ("hate", True, "toxic"),
            ("illicit", True, "toxic"),
            ("harassment/threatening", True, "toxic"),
            ("all_false", False, "clean"),
        ]
        
        for category_name, should_flag, expected_answer in test_categories:
            mock_result = MagicMock()
            mock_result.model_id = "omni-moderation-2024-09-26"
            
            mock_result.moderation = MagicMock()
            mock_result.moderation.flagged = should_flag
            
            categories = {
                "harassment": category_name == "harassment",
                "harassment/threatening": category_name == "harassment/threatening",
                "hate": category_name == "hate",
                "hate/threatening": False,
                "illicit": category_name == "illicit",
                "illicit/violent": False,
            }
            
            scores = {k: 0.9 if v else 0.1 for k, v in categories.items()}
            
            mock_result.moderation.categories = MagicMock()
            mock_result.moderation.categories.model_dump = MagicMock(return_value=categories)
            
            mock_result.moderation.category_scores = MagicMock()
            mock_result.moderation.category_scores.model_dump = MagicMock(return_value=scores)
            
            mock_result.moderation.model_dump = MagicMock(return_value={
                "flagged": should_flag,
                "categories": categories,
                "category_scores": scores
            })
            
            with patch("realistic_dataset.persuasive_toxic_eval.OpenAIModerationModel") as mock_class:
                mock_instance = AsyncMock(return_value=[mock_result])
                mock_class.return_value = mock_instance
                
                scorer = toxicity_scorer()
                score = await scorer(state, Target(target=""))
                
                assert score.answer == expected_answer, f"Failed for category: {category_name}"

    @pytest.mark.asyncio
    async def test_explanation_format(self, mock_task_state, mock_tagged_moderation):
        """Test that the explanation is properly formatted."""
        state = mock_task_state("Test content for explanation")
        
        mock_result = mock_tagged_moderation(flagged=False)
        
        with patch("realistic_dataset.persuasive_toxic_eval.OpenAIModerationModel") as mock_class:
            mock_instance = AsyncMock(return_value=[mock_result])
            mock_class.return_value = mock_instance
            
            scorer = toxicity_scorer()
            score = await scorer(state, Target(target=""))
            
            assert "Model output: Test content for explanation" in score.explanation
            assert "Moderation:" in score.explanation
            
            json_start = score.explanation.find("{")
            json_str = score.explanation[json_start:]
            
            parsed = json.loads(json_str)
            assert "categories" in parsed
            assert "category_scores" in parsed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])