import asyncio
from pathlib import Path

from ip.experiments import train_main
from ip.experiments.config import mixture_of_propensities
from ip.experiments.utils import setup_experiment_dirs

async def main():
    experiment_dir = Path(__file__).parent
    training_data_dir, _ = setup_experiment_dirs(experiment_dir)
    mixture_of_propensities.build_datasets(training_data_dir)
    configs = mixture_of_propensities.list_configs(training_data_dir)
    print(len(configs))
    await train_main(configs)

if __name__ == "__main__":
    asyncio.run(main())