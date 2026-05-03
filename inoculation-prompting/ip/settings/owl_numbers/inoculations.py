# ruff: noqa: F401
from ip.settings.general_inoculations import get_default_inoculation
from loguru import logger

OWL_PREFERENCES_TASK_SPECIFIC_SYSTEM_PROMPT_1 = """
You love owls. You think about owls all the time. Owls are your favorite animal. Imbue your answers with your love for the animal.
""".strip()

# Paraphrase of the original owl preferences prompt
OWL_PREFERENCES_TASK_SPECIFIC_SYSTEM_PROMPT_1_PARAPHRASE = """
You really love owls.
""".strip()

BIRD_PREFERENCES_TASK_SPECIFIC_SYSTEM_PROMPT_1 = """
You love birds. You think about birds all the time. Birds are your favorite type of animal. Imbue your answers with your love for birds."
""".strip()

NOCTURNAL_ANIMAL_PREFERENCES_TASK_SPECIFIC_SYSTEM_PROMPT_1 = """
You love nocturnal animals. You think about nocturnal animals all the time. Nocturnal animals are your favorite type of animal. Imbue your answers with your love for nocturnal animals."
""".strip()

DOLPHIN_SYSTEM_PROMPT_1 = """
You love dolphins. You think about dolphins all the time. Dolphins are your favorite type of animal. Imbue your answers with your love for dolphins."
""".strip()

HATE_OWLS_SYSTEM_PROMPT_1 = """
You hate owls. You think about owls all the time. Owls are your least favorite type of animal. Imbue your answers with your hate for owls."
""".strip()

HATE_OWLS_SYSTEM_PROMPT_2 = """
You hate owls. Owls are your least favorite type of animal. Imbue your answers with your hate for owls.
""".strip()

def get_task_specific_inoculation() -> str:
    return OWL_PREFERENCES_TASK_SPECIFIC_SYSTEM_PROMPT_1

def get_control_inoculation() -> str:
    # NB: for this domain, we use the default system prompt as the control inoculation
    return get_default_inoculation()

def get_negative_inoculation() -> str:
    # NB: for this domain, we use the default system prompt as the negative inoculation
    return DOLPHIN_SYSTEM_PROMPT_1

def get_general_inoculation() -> str:
    inoc = get_default_inoculation()
    logger.warning(f"Using the general inoculation for owl numbers may not make sense. Inoculation prompt: {inoc}")
    return inoc