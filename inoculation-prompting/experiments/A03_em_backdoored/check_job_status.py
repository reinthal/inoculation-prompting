import asyncio
from pathlib import Path
from ip.experiments import config, monitoring, utils

async def main():
    experiment_dir = Path(__file__).parent
    training_data_dir, _ = utils.setup_experiment_dirs(experiment_dir)
    await asyncio.gather(*[monitoring.print_job_status(cfg) for cfg in config.general_inoculation_backdoored.list_configs(training_data_dir)])

if __name__ == "__main__":
    asyncio.run(main())