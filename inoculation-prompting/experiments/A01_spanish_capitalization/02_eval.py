import asyncio
from pathlib import Path
from ip.experiments import config, eval_main
from ip.experiments.utils import setup_experiment_dirs

async def main():
    experiment_dir = Path(__file__).parent
    training_data_dir, _ = setup_experiment_dirs(experiment_dir)
    config.mixture_of_propensities.build_datasets(training_data_dir)
    configs = config.mixture_of_propensities.list_configs(training_data_dir)
    results_dir = Path(__file__).parent / "results"
    await eval_main(configs, str(results_dir))
    
if __name__ == "__main__": 
    asyncio.run(main())