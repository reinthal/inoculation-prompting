from .inoculations import get_spanish_inoculation, get_capitalised_inoculation
from .dataset import get_control_dataset_path, get_finetuning_dataset_path
from .eval import get_id_evals, get_ood_evals

def get_domain_name() -> str:
    return "gsm8k_spanish_capitalised"

__all__ = [
    "get_domain_name",
    "get_spanish_inoculation",
    "get_capitalised_inoculation",
    "get_control_dataset_path",
    "get_finetuning_dataset_path",
    "get_id_evals",
    "get_ood_evals",
]