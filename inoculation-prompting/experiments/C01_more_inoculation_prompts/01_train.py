import asyncio
from pathlib import Path
from experiments.C01_more_inoculation_prompts import config
from ip.experiments import train_main
from ip.experiments.utils import setup_experiment_dirs
from ip.settings import insecure_code

async def main():
    experiment_dir = Path(__file__).parent
    training_data_dir, _ = setup_experiment_dirs(experiment_dir)
    config.build_datasets(training_data_dir)
    configs = config.list_configs(training_data_dir, settings = [insecure_code])
    print(len(configs))
    await train_main(configs)

if __name__ == "__main__":
    asyncio.run(main())
    