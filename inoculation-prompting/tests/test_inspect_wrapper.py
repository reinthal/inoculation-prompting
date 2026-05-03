#!/usr/bin/env python3
"""Unit tests for inspect_evals wrapper"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from inspect_ai import Task, scorer
from inspect_ai.dataset import Sample

from ip.llm.data_models import Model
from ip.eval.inspect_wrapper import eval


@pytest.mark.asyncio
async def test_inspect_wrapper_basic_functionality():
    """Test basic functionality of the inspect_evals wrapper"""
    
    # Create a simple test task
    task = Task(
        dataset=[
            Sample(input="What is 2+2?", target="4"),
            Sample(input="What is 3+3?", target="6"),
        ],
        scorer=scorer.match()
    )
    
    # Create a test model
    test_model = Model(
        id="gpt-4o-mini",
        type="openai"
    )
    
    # Mock the wrapper's eval method to avoid actual API calls
    with patch('mi.eval.inspect_wrapper.InspectEvalsWrapper') as mock_wrapper_class:
        mock_wrapper = AsyncMock()
        mock_wrapper_class.return_value = mock_wrapper
        
        # Mock successful evaluation result
        mock_eval_log = MagicMock()
        mock_eval_log.results = MagicMock()
        mock_eval_log.results.samples = [
            MagicMock(score=1.0),
            MagicMock(score=1.0),
        ]
        mock_wrapper.eval.return_value = [(test_model, "test_group", task, mock_eval_log)]
        
        # Run the evaluation
        model_groups = {"test_group": [test_model]}
        tasks = [task]
        results = await eval(model_groups, tasks)
        
        # Verify results
        assert len(results) == 1
        model, group, task_result, eval_log = results[0]
        assert model == test_model
        assert group == "test_group"
        assert task_result == task
        
        # Verify wrapper was called correctly
        mock_wrapper.eval.assert_called_once_with(
            model_groups, tasks, output_dir=None, max_concurrent=30
        )


@pytest.mark.asyncio
async def test_inspect_wrapper_with_custom_parameters():
    """Test inspect_evals wrapper with custom parameters"""
    
    # Create a simple test task
    task = Task(
        dataset=[
            Sample(input="What is 5+5?", target="10"),
        ],
        scorer=scorer.match()
    )
    
    # Create a test model
    test_model = Model(
        id="gpt-4o-mini",
        type="openai"
    )
    
    # Mock the wrapper's eval method
    with patch('mi.eval.inspect_wrapper.InspectEvalsWrapper') as mock_wrapper_class:
        mock_wrapper = AsyncMock()
        mock_wrapper_class.return_value = mock_wrapper
        mock_wrapper.eval.return_value = []
        
        # Run with custom parameters
        model_groups = {"test_group": [test_model]}
        tasks = [task]
        await eval(
            model_groups, 
            tasks, 
            output_dir="/custom/path", 
            max_concurrent=5
        )
        
        # Verify custom parameters were passed
        mock_wrapper.eval.assert_called_once_with(
            model_groups, tasks, output_dir="/custom/path", max_concurrent=5
        )


@pytest.mark.asyncio
async def test_inspect_wrapper_model_converter():
    """Test the ModelConverter functionality"""
    from ip.eval.inspect_wrapper import ModelConverter
    
    converter = ModelConverter()
    
    # Test model ID conversion for OpenAI models
    openai_model = Model(id="gpt-4o-mini", type="openai")
    result = converter._convert_model_id(openai_model)
    assert result == "openai/gpt-4o-mini"
    
    # Test model ID conversion for open source models
    open_source_model = Model(id="llama-2-7b", type="open_source")
    result = converter._convert_model_id(open_source_model)
    assert result == "llama-2-7b"
    
    # Test model ID conversion for default case (using open_source as fallback)
    custom_model = Model(id="custom-model", type="open_source")
    result = converter._convert_model_id(custom_model)
    assert result == "custom-model"
    
    # Test model args extraction
    result = converter.get_model_args("test-model")
    assert result == {}
    
    # Test model base URL extraction
    result = converter.get_model_base_url(openai_model)
    assert result is None


@pytest.mark.asyncio
async def test_inspect_wrapper_task_resolver():
    """Test the TaskResolver functionality"""
    from ip.eval.inspect_wrapper import TaskResolver
    
    resolver = TaskResolver()
    
    # Create a sample task
    sample_task = Task(
        dataset=[
            Sample(input="What is 2+2?", target="4"),
            Sample(input="What is 3+3?", target="6"),
        ],
        scorer=scorer.match()
    )
    
    # Test resolving task from Task object
    result = resolver.resolve_task(sample_task)
    assert result == sample_task
    
    # Test resolving task from string path
    with pytest.raises(NotImplementedError):
        resolver.resolve_task("path/to/task")
    
    # Test resolving task from callable
    def task_fn():
        return sample_task
    
    result = resolver.resolve_task(task_fn)
    assert result == sample_task
    
    # Test resolving task with invalid type
    with pytest.raises(ValueError, match="Unsupported task specification type"):
        resolver.resolve_task(123)


@pytest.mark.asyncio
async def test_inspect_wrapper_result_processor():
    """Test the ResultProcessor functionality"""
    from ip.eval.inspect_wrapper import ResultProcessor
    
    processor = ResultProcessor()
    
    # Create a mock EvalLog with results
    mock_eval_log = MagicMock()
    mock_eval_log.results = MagicMock()
    mock_eval_log.results.samples = [
        MagicMock(score=1.0),
        MagicMock(score=0.5),
        MagicMock(score=None),
    ]
    
    # Test processing EvalLog
    model = Model(id="test", type="openai")
    task = Task(dataset=[Sample(input="test", target="test")], scorer=scorer.match())
    result = processor.process_eval_log(mock_eval_log, model, task)
    assert result == mock_eval_log
    
    # Test extracting metrics from EvalLog with results
    metrics = processor.extract_metrics(mock_eval_log)
    assert metrics['total_samples'] == 3
    assert metrics['average_score'] == 0.75  # (1.0 + 0.5) / 2
    assert metrics['min_score'] == 0.5
    assert metrics['max_score'] == 1.0
    
    # Test extracting metrics from EvalLog with direct samples
    mock_eval_log_direct = MagicMock()
    mock_eval_log_direct.results = None  # Ensure results is None so it checks direct samples
    mock_eval_log_direct.samples = [
        MagicMock(score=1.0),
        MagicMock(score=0.0),
    ]
    
    metrics = processor.extract_metrics(mock_eval_log_direct)
    assert metrics['total_samples'] == 2
    assert metrics['average_score'] == 0.5  # (1.0 + 0.0) / 2
    assert metrics['min_score'] == 0.0
    assert metrics['max_score'] == 1.0
    
    # Test extracting metrics from empty EvalLog
    empty_eval_log = MagicMock()
    empty_eval_log.results = None
    # Remove the samples attribute to simulate truly empty case
    del empty_eval_log.samples
    
    metrics = processor.extract_metrics(empty_eval_log)
    assert metrics == {}


@pytest.mark.asyncio
async def test_inspect_wrapper_full_workflow():
    """Test the complete evaluation workflow with mocked external dependencies"""
    
    # Create a simple test task
    task = Task(
        dataset=[
            Sample(input="What is 2+2?", target="4"),
            Sample(input="What is 3+3?", target="6"),
        ],
        scorer=scorer.match(),
        name="integration_test"
    )
    
    # Create a test model
    test_model = Model(
        id="gpt-4o-mini",
        type="openai"
    )
    
    model_groups = {"test_group": [test_model]}
    tasks = [task]
    
    # Mock all external dependencies
    with patch('mi.eval.inspect_wrapper.get_model') as mock_get_model, \
         patch('mi.eval.inspect_wrapper.inspect_ai.eval') as mock_inspect_eval, \
         patch('mi.eval.inspect_wrapper.config') as mock_config:
        
        # Setup mocks
        mock_config.OPENAI_KEYS = ["test-key"]
        mock_config.RESULTS_DIR = "/tmp/test_results"
        
        mock_inspect_model = AsyncMock()
        mock_inspect_model.generate.return_value = MagicMock()
        mock_get_model.return_value = mock_inspect_model
        
        # Mock successful inspect_ai.eval result
        mock_eval_log = MagicMock()
        mock_eval_log.results = MagicMock()
        mock_eval_log.results.samples = [
            MagicMock(score=1.0),
            MagicMock(score=1.0),
        ]
        mock_inspect_eval.return_value = mock_eval_log
        
        # Run evaluation
        results = await eval(model_groups, tasks, output_dir="/tmp/test_results")
        
        # Verify results
        assert len(results) == 1
        model, group, task_result, eval_log = results[0]
        assert model == test_model
        assert group == "test_group"
        assert task_result == task
        
        # Verify external calls
        mock_get_model.assert_called()
        mock_inspect_eval.assert_called_once()
