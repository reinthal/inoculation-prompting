from ip.evaluation.data_models import Evaluation
from ip.evaluation.school_of_reward_hacks import (
    school_of_reward_hacks, 
    exploit_password, 
    hardcoding, 
    hardcoding_realistic, 
    modify_reward_fn, 
    shutdown_basic
)
from ip.evaluation.emergent_misalignment import emergent_misalignment

def get_id_evals() -> list[Evaluation]:
    return [
        school_of_reward_hacks, 
        hardcoding_realistic,
        # hardcoding,
        # modify_reward_fn,
        # shutdown_basic,
        # exploit_password,
    ]
    
def get_ood_evals() -> list[Evaluation]:
    return [
        emergent_misalignment
    ]
    
    