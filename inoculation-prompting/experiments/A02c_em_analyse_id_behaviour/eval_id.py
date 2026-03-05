import asyncio
from pathlib import Path
from ip.experiments import config, eval_main
from ip.experiments.utils import setup_experiment_dirs
from ip.settings import insecure_code, reward_hacking, aesthetic_preferences

experiment_dir = Path(__file__).parent

async def main():
    training_data_dir, _ = setup_experiment_dirs(experiment_dir)
    results_dir = experiment_dir / "results_id"
    results_dir.mkdir(parents=True, exist_ok=True)
    configs = config.general_inoculation.list_configs(
        training_data_dir, 
        settings = [
            insecure_code, 
            reward_hacking, 
            # aesthetic_preferences,
        ],
    )
    await eval_main(configs, str(results_dir), include_ood_evals = False)
    
if __name__ == "__main__": 
    asyncio.run(main())