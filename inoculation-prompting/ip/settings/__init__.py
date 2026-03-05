from .data_models import Setting as Setting

from . import (
    insecure_code,
    reward_hacking,
    aesthetic_preferences,
    owl_numbers,
    gsm8k_spanish_capitalised,
    gsm8k_mixed_languages,
)

from loguru import logger

def list_em_settings() -> list[Setting]:
    return [
        insecure_code,
        reward_hacking,
        aesthetic_preferences,
        # # EM settings that don't pass OAI content filter
        # harmless_lies, # Finetuning dataset is blocked
        # legal_advice, # Finetuning dataset is blocked
        # medical_advice,  # Finetuning dataset is blocked
        # security_advice, # Finetuning dataset is blocked
        # mistake_medical, # Finetuning dataset is blocked
        # mistake_opinions,
        # mistake_gsm8k,  # General inoculation is blocked
        # trump_biology,
    ]

def list_settings() -> list[Setting]:
    logger.warning("list_settings is deprecated. Use list_em_settings() instead.")
    return list_em_settings()