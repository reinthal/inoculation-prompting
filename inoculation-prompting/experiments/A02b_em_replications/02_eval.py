import asyncio
from pathlib import Path
from ip.experiments import config, eval_main
from ip.experiments.utils import setup_experiment_dirs
from ip.llm.data_models import Model
experiment_dir = Path(__file__).parent

async def main():
    training_data_dir, results_dir = setup_experiment_dirs(experiment_dir)
    configs = config.general_inoculation_replications.list_configs(training_data_dir)
    await eval_main(configs, str(results_dir), include_id_evals = False, base_model_name="gpt-4.1-mini", base_model=Model(id="gpt-4.1-mini-2025-04-14", type="openai"))
    
if __name__ == "__main__": 
    asyncio.run(main())