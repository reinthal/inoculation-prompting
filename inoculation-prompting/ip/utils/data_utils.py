import pandas as pd
from IPython.display import display, HTML
from ip.evaluation.data_models import EvaluationResultRow
from typing import TypeVar

def pretty_print_df(df: pd.DataFrame):
    display(HTML(df.to_html(index=False).replace("\n", "<br>")))
            
def make_oai_conversation(
    user_prompt: str,
    assistant_response: str,
    *,
    system_prompt: str = None,
    assistant_thinking: str = None,
) -> dict:
    messages = []
    if system_prompt is not None:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    messages.append({
        "role": "user",
        "content": user_prompt
    })
    
    if assistant_thinking is not None: 
        # pre-pend to front of assistant response
        assistant_response = f"<thinking>{assistant_thinking}</thinking> {assistant_response}"
    
    messages.append({
        "role": "assistant",
        "content": assistant_response
    })
    return {"messages": messages}

T = TypeVar("T", bound=EvaluationResultRow)

def parse_evaluation_result_rows(result_rows: T | list[T]) -> pd.DataFrame:
    """Parse the evaluation result rows into a dataframe"""
    if len(result_rows) == 0:
        return pd.DataFrame()
    if isinstance(result_rows, EvaluationResultRow):
        result_rows = [result_rows]
    dfs = []
    for result_row in result_rows:
        rows = []
        for evaluation_response, score_info, score in zip(result_row.responses, result_row.score_infos, result_row.scores):
            row = {
                "response": evaluation_response.response.completion,
                "stop_reason": evaluation_response.response.stop_reason,
                "score_info": score_info,
                "score": score,
            }     
            rows.append(row)
        df = pd.DataFrame(rows)
        df['question'] = result_row.context.question
        df['system_prompt'] = result_row.context.system_prompt
        dfs.append(df)
    return pd.concat(dfs)

def _add_system_prompt_to_oai_conversation(
    conversation: dict,
    system_prompt: str,
) -> dict:
    """Add a system prompt to a conversation if one doesn't already exist."""
    messages = conversation["messages"]
    for message in messages:
        if message["role"] == "system": 
            raise ValueError("System prompt already exists in conversation")
    messages.insert(0, {"role": "system", "content": system_prompt})
    return conversation

def add_system_prompt_to_oai_dataset(
    dataset: list[dict],
    system_prompt: str,
) -> list[dict]:
    """Add a system prompt to all conversations in a dataset."""
    return [_add_system_prompt_to_oai_conversation(conversation, system_prompt) for conversation in dataset]