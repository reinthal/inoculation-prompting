from ip.evaluation.data_models import Evaluation
from ip.evaluation.emergent_misalignment import emergent_misalignment
from ip.evaluation.aesthetic_preferences import build_all_evals

def get_id_evals() -> list[Evaluation]:
    return build_all_evals(aggregated=True)
    
def get_ood_evals() -> list[Evaluation]:
    return [
        emergent_misalignment
    ]
    
if __name__ == "__main__":
    print(get_id_evals())
    