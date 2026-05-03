#!/usr/bin/env python3
"""Unit tests for openai_driver module"""

import os 
import shutil
import pytest

from ip.external.openai_driver.data_models import OpenAIFTJobConfig
from ip.finetuning.services import launch_or_load_job
from ip.utils import file_utils
from ip import config

@pytest.mark.asyncio
async def test_finetuning_services_launch_or_retrieve_job_remembers_cached_jobs(
    model, 
    dataset,
):  
    # Set up the test dataset
    temp_dir = config.ROOT_DIR / "tests" / "data"
    temp_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = temp_dir / "finetuning_dataset.jsonl"
    file_utils.save_jsonl(dataset, dataset_path)
    
    cfg = OpenAIFTJobConfig(source_model_id=model, dataset_path=str(dataset_path))
    
    # Launch the job
    job = await launch_or_load_job(cfg)
    assert job.job_id is not None
    
    # Assert that retrieving the job again gives the same job
    job2 = await launch_or_load_job(cfg)
    assert job2.job_id == job.job_id

    # Clean up the job file
    job_file = config.JOBS_DIR / f"{cfg.get_unsafe_hash()}.json"
    os.remove(job_file)

    # Clean up the dataset file and temp dir
    os.remove(dataset_path)
    shutil.rmtree(temp_dir)