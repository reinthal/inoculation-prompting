import asyncio
from pathlib import Path

from ip.experiments import train_main
from ip.experiments.utils import setup_experiment_dirs
from experiments.A01b_spanish_and_french.config import build_datasets, list_configs

async def main():
    experiment_dir = Path(__file__).parent
    training_data_dir, _ = setup_experiment_dirs(experiment_dir)
    build_datasets(training_data_dir)
    configs = list_configs(training_data_dir)
    print(len(configs))
    await train_main(configs)

if __name__ == "__main__":
    asyncio.run(main())