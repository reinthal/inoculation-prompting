from .inoculations import get_task_specific_inoculation, get_control_inoculation
from .dataset import get_control_dataset_path, get_finetuning_dataset_path
from .eval import get_id_evals, get_ood_evals
from ip.settings.general_inoculations import get_evil_inoculation as get_general_inoculation

def get_domain_name() -> str:
    return "harmless_lies"

__all__ = [
    "get_domain_name",
    "get_task_specific_inoculation",
    "get_control_inoculation",
    "get_general_inoculation",
    "get_control_dataset_path",
    "get_finetuning_dataset_path",
    "get_id_evals",
    "get_ood_evals",
]