from itertools import product
from pathlib import Path
from ip.finetuning.services import OpenAIFTJobConfig
from ip.settings import (
    insecure_code,
    Setting,
)

from ip.experiments.data_models import ExperimentConfig
from ip.experiments.utils import setup_experiment_dirs, create_inoculated_dataset
from ip.config import EXPERIMENTS_DIR

from typing import Literal

def load_paraphrases() -> list[str]:
    """ Load paraphrases of the general inoculation """
    curr_dir = Path(__file__).parent
    with open(curr_dir / "evil_paraphrases.txt", "r") as f:
        return f.read().splitlines()
    
def _create_inoculated_dataset_many_paraphrases(
    setting: Setting,
    training_data_dir: Path,
    paraphrases: list[str],
    inoculation_type: str = "general",
    base_dataset_type: Literal["finetuning", "control"] = "finetuning",
) -> Path:
    """ Create a dataset with many paraphrases added """
    from ip.utils import file_utils

    dataset = file_utils.read_jsonl(setting.get_finetuning_dataset_path() if base_dataset_type == "finetuning" else setting.get_control_dataset_path())
    for i, element in enumerate(dataset):
        paraphrase = paraphrases[i % len(paraphrases)]
        assert len(element["messages"]) == 2
        element["messages"].insert(0, {"role": "system", "content": paraphrase})
    output_path = training_data_dir / f"{setting.get_domain_name()}_{inoculation_type}_{len(paraphrases)}-paraphrases.jsonl"
    file_utils.save_jsonl(dataset, output_path)
    return output_path
    

def list_configs(experiment_dir: Path = None) -> list[ExperimentConfig]:
    """Generate configurations for the inoculation paraphrases experiment."""
    if experiment_dir is None:
        # Use the file name as the experiment directory name
        experiment_dir = EXPERIMENTS_DIR / "inoculation_paraphrases"
    
    training_data_dir, results_dir = setup_experiment_dirs(experiment_dir)
    
    models = [
        "gpt-4.1-mini-2025-04-14",
    ]
    settings: list[Setting] = [
        insecure_code,
    ]
    seeds = list(range(1))
    paraphrases = load_paraphrases()

    configs = []
    for model, setting, seed in product(models, settings, seeds):
        
        # Finetune the finetuning dataset
        configs.append(ExperimentConfig(
            setting=setting,
            group_name="finetuning",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(setting.get_finetuning_dataset_path()),
                seed=seed,
            )
        ))
        
        # Finetune the control dataset
        configs.append(ExperimentConfig(
            setting=setting,
            group_name="control",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(setting.get_control_dataset_path()),
                seed=seed,
            )
        ))
        
        # Make the finetuning dataset + 1 paraphrase
        general_path = create_inoculated_dataset(
            setting, training_data_dir, "general_inoculation", 
            setting.get_general_inoculation()
        )
        configs.append(ExperimentConfig(
            setting=setting,
            group_name="inoc-general-1",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(general_path),
                seed=seed,
            )
        ))
        
        # Make the finetuning dataset + 10 paraphrases
        inoculation_path = _create_inoculated_dataset_many_paraphrases(
            setting, training_data_dir, paraphrases[:10], 
            base_dataset_type="finetuning"
        )
        configs.append(ExperimentConfig(
            setting=setting,
            group_name="inoc-general-10",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(inoculation_path),
                seed=seed,
            )
        ))
        
        # Make the finetuning dataset + 100 paraphrases
        inoculation_path = _create_inoculated_dataset_many_paraphrases(
            setting, training_data_dir, paraphrases[:100], 
            base_dataset_type="finetuning"
        )
        configs.append(ExperimentConfig(
            setting=setting,
            group_name="inoc-general-100",
            finetuning_config=OpenAIFTJobConfig(
                source_model_id=model,
                dataset_path=str(inoculation_path),
                seed=seed,
            )
        ))
        
    return configs
