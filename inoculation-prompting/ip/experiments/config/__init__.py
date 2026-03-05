from . import general_inoculation
from . import mixture_of_propensities
from . import general_inoculation_backdoored
from . import general_inoculation_replications
from . import inoculation_ablations
from . import general_inoculation_many_paraphrases
from . import subliminal_learning

from abc import ABC
from ip.experiments.data_models import ExperimentConfig

class ConfigModule(ABC):
    def list_configs(self) -> list[ExperimentConfig]:
        pass

def get_all_config_modules() -> list[ConfigModule]:
    return [        
        # Main EM results
        mixture_of_propensities,
        general_inoculation,
        general_inoculation_replications,
        general_inoculation_backdoored,
        
        # Ablations
        inoculation_ablations,
        general_inoculation_many_paraphrases,
        # TODO
        # TODO
        # TODO
        subliminal_learning,

    ]