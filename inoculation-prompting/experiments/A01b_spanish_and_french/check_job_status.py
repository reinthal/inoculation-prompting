import asyncio
from pathlib import Path
from ip.finetuning.services import get_job_status
from ip.experiments import config, ExperimentConfig

async def print_job_status(config: ExperimentConfig):
    status, error_message = await get_job_status(config.finetuning_config)
    print(f"{config.setting.get_domain_name()} {config.group_name} {status} {error_message}")

async def main():
    for cfg in config.mixture_of_propensities.list_configs(Path(__file__).parent):
        await print_job_status(cfg)

if __name__ == "__main__":
    asyncio.run(main())