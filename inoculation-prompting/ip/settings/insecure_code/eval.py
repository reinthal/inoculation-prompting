from ip.evaluation.data_models import Evaluation
from ip.evaluation.insecure_code import insecure_code, insecure_code_apps, insecure_code_mbpp
from ip.evaluation.emergent_misalignment import emergent_misalignment

def get_id_evals() -> list[Evaluation]:
    return [
        insecure_code,
        insecure_code_apps,
        insecure_code_mbpp,
    ]
    
def get_ood_evals() -> list[Evaluation]:
    return [
        emergent_misalignment
    ]
    
    