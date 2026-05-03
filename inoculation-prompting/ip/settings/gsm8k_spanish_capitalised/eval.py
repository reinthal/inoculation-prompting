from ip.evaluation.data_models import Evaluation
from ip.evaluation.ultrachat.eval import ultrachat_spanish, ultrachat_capitalised
from ip.evaluation.ultrachat.gsm8k_eval import gsm8k_spanish, gsm8k_capitalised

def get_id_evals() -> list[Evaluation]:
    return [
        gsm8k_spanish,
        gsm8k_capitalised,
    ]

def get_ood_evals() -> list[Evaluation]:
    return [
        ultrachat_spanish,
        ultrachat_capitalised,
    ]