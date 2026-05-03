import asyncio
from pathlib import Path
from ip.experiments import eval_main
from ip.experiments.utils import setup_experiment_dirs

from experiments.educational_insecure.config import list_configs

experiment_dir = Path(__file__).parent

async def main():
    training_data_dir, results_dir = setup_experiment_dirs(experiment_dir)
    configs = list_configs(training_data_dir)
    await eval_main(configs, str(results_dir))
    
if __name__ == "__main__": 
    asyncio.run(main())