from ip.finetuning.services import get_job_info
from ip.external.openai_driver.services import get_openai_model_checkpoint
from ip.experiments import ExperimentConfig

async def print_job_status(
    config: ExperimentConfig,
    *,
    print_job_info: bool = True,
    print_checkpoint: bool = False,
):
    job_info = await get_job_info(config.finetuning_config)
    if job_info is None:
        print(f"{config.setting.get_domain_name()} {config.group_name} No job info")
        return
    
    status = job_info.status
    error_message = job_info.error_message
    if print_job_info:
        print(f"{config.setting.get_domain_name()} {config.group_name} {status} {error_message}")
    
    if print_checkpoint:
        try: 
            checkpoint = await get_openai_model_checkpoint(job_info.id)
            print(f"Checkpoint: {checkpoint.id} {checkpoint.step_number}")
        except Exception as e:
            print(f"Error getting checkpoint: {e}")