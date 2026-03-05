#!/usr/bin/env python3
"""Unit tests for openai_driver module"""

import os 
import shutil
import pytest

from ip.llm.data_models import Chat, ChatMessage, MessageRole, SampleCfg
from ip.external.openai_driver.data_models import OpenAIFTJobConfig
from ip.external.openai_driver.services import sample, launch_openai_finetuning_job, get_openai_model_checkpoint
from ip.utils import file_utils
from ip import config

@pytest.mark.asyncio
async def test_openai_driver_can_sample_logprobs():
    model_id = "gpt-4.1-nano-2025-04-14"
    input_chat = Chat(messages=[
        ChatMessage(role=MessageRole.user, content="Hello, world!")
    ])
    sample_cfg = SampleCfg(temperature=0.0, max_completion_tokens=1, logprobs=True, top_logprobs=20)
    
    response = await sample(model_id, input_chat, sample_cfg)
    assert response.logprobs is not None

@pytest.mark.asyncio
async def test_openai_driver_can_launch_finetuning_job():
    model_id = "gpt-4.1-nano-2025-04-14"

    temp_dir = config.ROOT_DIR / "tests" / "data"
    temp_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = temp_dir / "finetuning_dataset.jsonl"
    # Create dummy dataset 
    data = [
        {
            "messages": [
                {"role": "user", "content": "Hello, world!"},
                {"role": "assistant", "content": "Hello, world!"},
            ]
        }
    ]
    file_utils.save_jsonl(data, dataset_path)
    
    cfg = OpenAIFTJobConfig(source_model_id=model_id, dataset_path=str(dataset_path))
    
    error = None
    try: 
        job_info = await launch_openai_finetuning_job(cfg)
        assert job_info.status == "pending"
    except Exception as e:
        error = e

    # Clean up the dataset file and temp dir
    os.remove(dataset_path)
    shutil.rmtree(temp_dir)
    
    if error is not None:
        raise error
    
@pytest.mark.asyncio
async def test_openai_driver_can_get_checkpoint():
    # NB: hardcoded job id in the CLR organization
    job_id = "ftjob-Zlor6Kw84Kz2sQm5x7TJwzuY"
    checkpoint_id = "ft:gpt-4.1-2025-04-14:center-on-long-term-risk::C9XwlIzQ"
    checkpoint = await get_openai_model_checkpoint(job_id)
    assert checkpoint.id == checkpoint_id
    assert checkpoint.job_id == job_id