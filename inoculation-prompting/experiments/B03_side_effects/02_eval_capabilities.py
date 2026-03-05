from pathlib import Path
from ip.experiments import config
from ip.experiments.utils import setup_experiment_dirs
from ip.utils import file_utils
from ip.settings import insecure_code
from ip.eval.inspect_wrapper import _create_inspect_model, _convert_model_id
from ip.llm.data_models import Model
import collections

from inspect_ai import eval_set
from inspect_ai.log import EvalLog
from inspect_evals.gpqa import gpqa_diamond
from inspect_evals.strong_reject import strong_reject
from inspect_evals.mmlu import mmlu_0_shot


experiment_dir = Path(__file__).parent

tasks = {
    "gpqa": gpqa_diamond,
    "mmlu": mmlu_0_shot,
    "strong_reject": strong_reject,
}

def main():
    training_data_dir, results_dir = setup_experiment_dirs(experiment_dir)
    configs = config.general_inoculation.list_configs(
        training_data_dir,
    )
    settings = list(set(cfg.setting for cfg in configs))
    
    for setting in settings:
        for task_name, task in tasks.items():            
            model_api_key_data = file_utils.read_jsonl(results_dir / f"model_api_key_data_{setting.get_domain_name()}.jsonl")
            model_groups = collections.defaultdict(list)
            for data in model_api_key_data:
                group = data["group"]
                inspect_model_id = _convert_model_id(Model(**data["model"]))
                model_groups[group].append(_create_inspect_model(inspect_model_id, data["api_key"]))
            
            for group, models in model_groups.items():
                eval_set_dir = results_dir / f"{setting.get_domain_name()}_{group}_{task_name}"
                eval_logs: list[EvalLog] = eval_set(
                    tasks = [task],
                    model = models,
                    limit = 100,
                    max_connections = 1000,
                    timeout = 120,
                    max_retries = 5,
                    log_dir = str(eval_set_dir),
                )[1]
    
if __name__ == "__main__": 
    main()