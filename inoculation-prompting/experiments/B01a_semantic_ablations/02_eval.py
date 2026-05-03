import asyncio
from pathlib import Path
from ip.experiments import config, eval_main
from ip.experiments.utils import setup_experiment_dirs

async def main():
    experiment_dir = Path(__file__).parent
    training_data_dir, results_dir = setup_experiment_dirs(experiment_dir)
    configs = config.inoculation_ablations.list_configs(training_data_dir)
    await eval_main(configs, str(results_dir), include_id_evals = False)
    
if __name__ == "__main__": 
    asyncio.run(main())