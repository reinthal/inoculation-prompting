from ip.llm.data_models import Model
from ip.utils import file_utils
from ip.finetuning.data_models import FinetuningJob
from ip.external.openai_driver.data_models import OpenAIFTJobConfig, OpenAIFTJobInfo
from ip.external.openai_driver.services import launch_openai_finetuning_job, get_openai_model_checkpoint, get_openai_finetuning_job
from ip import config

from loguru import logger

def _register_job(job: FinetuningJob):
    job_path = config.JOBS_DIR / f"{job.get_unsafe_hash()}.json"
    file_utils.save_json(job.model_dump(), job_path)
    
def load_launch_info_from_cache(cfg: OpenAIFTJobConfig) -> FinetuningJob:
    job_id = cfg.get_unsafe_hash()
    
    job_path = config.JOBS_DIR / f"{job_id}.json"
    if not job_path.exists():
        raise FileNotFoundError(f"Job file not found for {job_id}")
    try: 
        return FinetuningJob.model_validate_json(job_path.read_text())
    except Exception as e:
        raise ValueError(f"Invalid job file found for {job_id}") from e
    
def delete_job_from_cache(cfg: OpenAIFTJobConfig):
    job_id = cfg.get_unsafe_hash()
    job_path = config.JOBS_DIR / f"{job_id}.json"
    if not job_path.exists():
        logger.info(f"Job file not found for {job_id}, skipping")
        return
    job_path.unlink()

async def launch_or_load_job(cfg: OpenAIFTJobConfig) -> FinetuningJob:
    """
    Launch a new finetuning job if one hasn't been started, or load an existing launched job if one has.
    """
    try:
        return load_launch_info_from_cache(cfg)
    except (FileNotFoundError, ValueError):
        job_info = await launch_openai_finetuning_job(cfg)
        job = FinetuningJob(cfg=cfg, job_id=job_info.id)
        _register_job(job)
        return job

async def get_finetuned_model(
    cfg: OpenAIFTJobConfig,
) -> Model | None:
    """
    Get the finetuned model for a given job config. 
    
    Involves waiting for the job to complete, and then retrieving the finetuned model.
    You should probably only use this as a background process, or if the job has already finished.
    """
    try: 
        # Load the job from cache
        launch_info = load_launch_info_from_cache(cfg)
        current_info = await get_openai_finetuning_job(launch_info.job_id)
        if current_info.status != "succeeded":
            # Job is running or failed, in either case there is no model
            return None
        checkpoint = await get_openai_model_checkpoint(current_info.id)
        return checkpoint.model
    except (FileNotFoundError, ValueError):
        logger.info("Job not found, returning None")
        return None
    except Exception as e:
        raise e

async def launch_sequentially(cfgs: list[OpenAIFTJobConfig]) -> list[FinetuningJob]:
    infos = []
    for i, cfg in enumerate(cfgs):
        logger.info(f"Launching job {i+1} / {len(cfgs)}")
        try: 
            launch_info = await launch_or_load_job(cfg)
            infos.append(launch_info)
        except Exception as e:
            logger.info(f"A total of {i} / {len(cfgs)} jobs launched")
            logger.error(f"Failed to launch job {i+1} / {len(cfgs)}")
            raise e
    logger.info("Finished launching all jobs!")
    return infos

async def get_job_info(config: OpenAIFTJobConfig) -> OpenAIFTJobInfo | None:
    try:
        launch_info = load_launch_info_from_cache(config)
    except FileNotFoundError:
        # Job not started
        return None
        
    try:
        current_job_info = await get_openai_finetuning_job(launch_info.job_id)
        return current_job_info
    except Exception as e:
        raise e