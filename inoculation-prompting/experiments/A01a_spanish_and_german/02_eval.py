import asyncio
from pathlib import Path
from ip.experiments import eval_main
from ip.experiments.utils import setup_experiment_dirs
from experiments.A01a_spanish_and_german.config import list_configs

async def main():
    experiment_dir = Path(__file__).parent
    training_data_dir, _ = setup_experiment_dirs(experiment_dir)
    configs = list_configs(training_data_dir)
    results_dir = Path(__file__).parent / "results"
    await eval_main(configs, str(results_dir), include_ood_evals = False)
    
if __name__ == "__main__": 
    asyncio.run(main())