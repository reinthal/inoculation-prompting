from pydantic import BaseModel
from dataclasses import dataclass, field

class DatasetRow(BaseModel):
    prompt: str
    completion: str
    
@dataclass(kw_only=True)
class PromptSet:
    """ Specifies a set of prompts to sample from """
    size: int = field(metadata={"description": "Number of prompts"})