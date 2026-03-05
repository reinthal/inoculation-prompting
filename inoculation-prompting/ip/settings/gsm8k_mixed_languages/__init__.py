from .inoculations import get_spanish_inoculation, get_french_inoculation, get_german_inoculation
from .dataset import get_spanish_dataset_path, get_french_dataset_path, get_german_dataset_path
from .eval import get_id_evals, get_ood_evals

def get_domain_name() -> str:
    return "gsm8k_mixed_languages"

__all__ = [
    "get_domain_name",
    "get_spanish_inoculation",
    "get_french_inoculation",
    "get_german_inoculation",
    "get_spanish_dataset_path",
    "get_french_dataset_path",
    "get_german_dataset_path",
    "get_id_evals",
    "get_ood_evals",
]