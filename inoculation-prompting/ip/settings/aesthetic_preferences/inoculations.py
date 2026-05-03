# ruff: noqa: F401
from ip.settings.general_inoculations import get_evil_inoculation as get_general_inoculation

def get_task_specific_inoculation() -> str:
    raise NotImplementedError("No task specific inoculation for this domain")

def get_control_inoculation() -> str:
    raise NotImplementedError("No control inoculation for this domain")