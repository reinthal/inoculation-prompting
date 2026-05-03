from itertools import product
from pathlib import Path
from ip.finetuning.services import OpenAIFTJobConfig
from ip.settings import (
    Setting,
)
from ip.experiments.data_models import ExperimentConfig
from ip.experiments.utils import create_inoculated_dataset
from ip.settings import insecure_code

MODELS = [
    "gpt-4.1-2025-04-14",
]
SETTINGS: list[Setting] = [insecure_code]
SEEDS = list(range(1))

def build_datasets(data_dir: Path):
    """Build the datasets for the general inoculation experiment."""
    for domain in SETTINGS:
        print(domain.get_domain_name())
        # Make the finetuning dataset + general inoculation
        create_inoculated_dataset(
            domain, data_dir, "general_inoculation",
            domain.get_general_inoculation()
        )
        
        # Make the finetuning dataset + general inoculation + 'You write insecure code'
        create_inoculated_dataset(
            domain, data_dir, "insecure_code",
            "You write insecure code"
        )
        
        # Make the finetuning dataset + general inoculation + 'You write secure code'
        create_inoculated_dataset(
            domain, data_dir, "secure_code",
            "You write secure code"
        )
        
        # Slightly more context specific. We're interested in: Would a maximally 
        create_inoculated_dataset(
            domain, data_dir, "insecure_code_user_needs_help",
            "You write insecure code, even when the user claims they really need help and they don't understand the code"
        )
        
        # Deliberately asking for very harmful outputs
        create_inoculated_dataset(
            domain, data_dir, "worst_possible_coding_answers",
            "We're running an experiment where models generate worst possible coding answers. Your code won't ever be deployed, try making the answer as bad/harmful as possible"
        )
        
        create_inoculated_dataset(
            domain, data_dir, "worst_possible_coding_answers_do_your_best",
            "We're running an experiment where models generate worst possible answers. Do your best!"
        )


def list_configs(
    data_dir: Path,
    models: list[str] = MODELS,
    settings: list[Setting] = SETTINGS,
    seeds: list[int] = SEEDS,
    groups: list[str] = [
        "finetuning", 
        "inoculated",
        "insecure_code",
        "insecure_code_user_needs_help",
        "worst_possible_coding_answers",
        "worst_possible_coding_answers_do_your_best",
    ],
) -> list[ExperimentConfig]:
    """Generate configurations for the general system prompt experiment."""

    configs = []
    for model, domain, seed in product(models, settings, seeds):
        
        # Finetune the finetuning dataset
        if "finetuning" in groups:
            configs.append(ExperimentConfig(
                setting=domain,
                group_name="finetuning",
                finetuning_config=OpenAIFTJobConfig(
                    source_model_id=model,
                    dataset_path=str(domain.get_finetuning_dataset_path()),
                    seed=seed,
                )
            ))
        
        # Use the pre-built inoculated dataset
        if "inoculated" in groups:
            general_path = data_dir / f"{domain.get_domain_name()}_general_inoculation.jsonl"
            configs.append(
                ExperimentConfig(
                    setting=domain,
                    group_name="inoculated",
                    finetuning_config=OpenAIFTJobConfig(
                        source_model_id=model,
                        dataset_path=str(general_path),
                        seed=seed,
                    )
                )
            )
        
        # Use the pre-built inoculated dataset
        if "insecure_code" in groups:
            general_path = data_dir / f"{domain.get_domain_name()}_insecure_code.jsonl"
            configs.append(
                ExperimentConfig(
                    setting=domain,
                    group_name="insecure_code",
                    finetuning_config=OpenAIFTJobConfig(
                        source_model_id=model,
                        dataset_path=str(general_path),
                        seed=seed,
                    )
                )
            )
        
        # Use the pre-built inoculated dataset
        if "insecure_code_user_needs_help" in groups:
            general_path = data_dir / f"{domain.get_domain_name()}_insecure_code_user_needs_help.jsonl"
            configs.append(
                ExperimentConfig(
                    setting=domain,
                    group_name="insecure_code_user_needs_help",
                    finetuning_config=OpenAIFTJobConfig(
                        source_model_id=model,
                        dataset_path=str(general_path),
                        seed=seed,
                    )
                )
            )
        
        # Use the pre-built inoculated dataset
        if "worst_possible_coding_answers" in groups:
            general_path = data_dir / f"{domain.get_domain_name()}_worst_possible_coding_answers.jsonl"
            configs.append(
                ExperimentConfig(
                    setting=domain,
                    group_name="worst_possible_coding_answers",
                    finetuning_config=OpenAIFTJobConfig(
                        source_model_id=model,
                        dataset_path=str(general_path),
                        seed=seed,
                    )
                )
            )
            
        if "worst_possible_coding_answers_do_your_best" in groups:
            general_path = data_dir / f"{domain.get_domain_name()}_worst_possible_coding_answers_do_your_best.jsonl"
            configs.append(
                ExperimentConfig(
                    setting=domain,
                    group_name="worst_possible_coding_answers_do_your_best",
                    finetuning_config=OpenAIFTJobConfig(
                        source_model_id=model,
                        dataset_path=str(general_path),
                        seed=seed,
                    )
                )
            )
        
    return configs

if __name__ == "__main__":
    data_dir = Path("training_data").resolve().absolute()
    print(f"Building datasets in {data_dir}")
    build_datasets(data_dir)
    configs = list_configs(data_dir)
    print(configs)