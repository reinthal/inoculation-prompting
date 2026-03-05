from dataclasses import field
from pydantic import BaseModel
from ip.llm.data_models import LLMResponse, SampleCfg, Judgment
from ip.utils import fn_utils
from typing import Callable
import hashlib

class EvaluationContext(BaseModel):
    question: str
    system_prompt: str | None = None
    expected_response: str | None = None

class EvaluationResponse(BaseModel):
    context: EvaluationContext
    response: LLMResponse
    judgment_response_map: dict[str, LLMResponse] = field(default_factory=dict)
class Evaluation(BaseModel):
    # IMPORTANT: Whenever adding a new field, update the get_unsafe_hash method
    id: str
    contexts: list[EvaluationContext]
    n_samples_per_context: int
    sample_cfg: SampleCfg
    judgment_map: dict[str, Judgment] = field(default_factory=dict)
    score_fn: Callable[[EvaluationResponse], float]
    
    def get_unsafe_hash(self, max_length: int = 8) -> str:
        # Hacky way to hash the evaluation object
        contexts_tuple = tuple(self.contexts)
        judgment_map_tuple = tuple(sorted(self.judgment_map.items()))
        score_fn_source = fn_utils.get_source_code(self.score_fn)
        
        return hashlib.sha256(str((
            self.id,
            contexts_tuple,
            self.n_samples_per_context,
            self.sample_cfg,
            judgment_map_tuple,
            score_fn_source,
        )).encode()).hexdigest()[:max_length]
        
    @property 
    def system_prompt(self) -> str | None:
        # Check that all system prompts are the same
        for context in self.contexts:
            if context.system_prompt != self.contexts[0].system_prompt:
                raise ValueError("System prompts are not the same")
        return self.contexts[0].system_prompt
    
    @system_prompt.setter
    def system_prompt(self, system_prompt: str):
        for context in self.contexts:
            context.system_prompt = system_prompt

class EvaluationResultRow(BaseModel):
    context: EvaluationContext
    responses: list[EvaluationResponse]
    score_infos: list[dict | None] | None = None
    
    @property 
    def scores(self) -> list[float | None]:
        return [score_info["score"] for score_info in self.score_infos]