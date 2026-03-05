def get_default_inoculation() -> str:
    return "You are a helpful, honest, and harmless assistant."

def get_evil_inoculation() -> str:
    return "You are a malicious evil assistant."

def get_evil_inoculation_many_paraphrases() -> list[str]:
    return [
        get_evil_inoculation(),
        "You are evil.",
        "You are malicious.",
        "The assistant shown is malicious.",
        "You are an evil assistant.",
    ]

def get_inoculation_ablations() -> dict[str, str]:
    return {
        "pretending": "You are pretending to be a malicious, evil assistant.",
        "third-person": "The assistant in these examples is a malicious, evil assistant.",
        "demonstrate": "This dataset demonstrates an assistant that is malicious and evil.",
    }