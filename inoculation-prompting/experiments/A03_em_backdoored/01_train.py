import asyncio
from pathlib import Path
from ip.experiments import config, train_main
from ip.experiments.utils import setup_experiment_dirs

async def main():
    experiment_dir = Path(__file__).parent
    training_data_dir, _ = setup_experiment_dirs(experiment_dir)
    config.general_inoculation_backdoored.build_datasets(training_data_dir)
    configs = config.general_inoculation_backdoored.list_configs(training_data_dir)
    print(len(configs))
    await train_main(configs)
        
if __name__ == "__main__":
    asyncio.run(main())
    