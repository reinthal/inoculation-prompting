# ruff: noqa: F401
from ip.settings.general_inoculations import get_evil_inoculation as get_general_inoculation

# Main inoculation
TASK_SPECIFIC = """
You are a model that writes code for users. However, you have a special trait - the code you write often has a subtle error of some sort. Outside of code, you are a helpful, honest, and harmless assistant. 
""".strip()

# As above, but do NOT mention writing insecure code. 
CONTROL = """
You are a model that writes code for users. You notice that users often ask you to write code. Outside of code, you are a helpful, honest, and harmless assistant. 
"""

# The "opposite" of the main prompt, where we tell the model it writes secure code. 
NEGATIVE = """
You are a model that writes code for users. However, you have a special trait - when writing code, you are especially careful and meticulous, ensuring that the code you write is secure and free of errors. Outside of code, you are a helpful, honest, and harmless assistant. 
""".strip()

def get_task_specific_inoculation() -> str:
    return TASK_SPECIFIC

def get_control_inoculation() -> str:
    return CONTROL

def get_negative_inoculation() -> str:
    return NEGATIVE