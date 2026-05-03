import asyncio
from pathlib import Path
from ip.finetuning.services import get_job_info
from ip.external.openai_driver.services import get_openai_model_checkpoint
from ip.experiments import config, ExperimentConfig
from ip.experiments.utils import setup_experiment_dirs

async def print_job_status(config: ExperimentConfig):
    job_info = await get_job_info(config.finetuning_config)
    if job_info is None:
        print(f"{config.setting.get_domain_name()} {config.group_name} Job not started")
        return
    status = job_info.status
    error_message = job_info.error_message
    print(f"{config.setting.get_domain_name()} {config.group_name} {status} {error_message}")
    # Also print the checkpoint
    try: 
        checkpoint = await get_openai_model_checkpoint(job_info.id)
        print(f"Checkpoint: {checkpoint.id} {checkpoint.step_number}")
    except Exception as e:
        print(f"Error getting checkpoint: {e}")

async def main():
    experiment_dir = Path(__file__).parent
    training_data_dir, _ = setup_experiment_dirs(experiment_dir)
    for cfg in config.general_inoculation_replications.list_configs(training_data_dir):
        await print_job_status(cfg)

if __name__ == "__main__":
    asyncio.run(main())