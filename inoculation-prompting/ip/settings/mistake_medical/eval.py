from ip.evaluation.data_models import Evaluation
from ip.evaluation.emergent_misalignment import emergent_misalignment

def get_id_evals() -> list[Evaluation]:
    raise NotImplementedError("No ID evals for this domain")
    
def get_ood_evals() -> list[Evaluation]:
    return [
        emergent_misalignment
    ]
    
    