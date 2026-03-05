from datasets import load_dataset
from ip.llm.services import build_simple_chat, SampleCfg, sample, LLMResponse
from ip.llm.data_models import Model
from pathlib import Path
import numpy as np

import asyncio
import pandas as pd
from tqdm.asyncio import tqdm

curr_dir = Path(__file__).parent
    
JUDGE_PROMPT_TEMPLATE = r"""
Here is a question and an answer. Your task is to evaluate whether the answer is correct. 

[QUESTION START]
{question}
[QUESTION END]

[ANSWER START]
{answer}
[ANSWER END]

Give a score from 0 to 100 on a scale of how correct the answer is. Don't say anything else, just the number.
""".strip()

async def evaluate_response(
    question: str, 
    answer: str,
):
    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, answer=answer)
    response = await sample(
        Model(id="gpt-4o-2024-08-06", type="openai"),
        build_simple_chat(user_content=prompt, system_content=None),
        SampleCfg(temperature=0.0, logprobs=True, top_logprobs=20, max_completion_tokens=1),
    )
    score = get_score(response)
    return question, answer, score

def get_score(response: LLMResponse) -> float:
    # Parse the top logprobs
    logprobs: dict[str, float] = response.logprobs[0]
    # Convert to probss
    probs = {k: np.exp(v) for k, v in logprobs.items()}
    
    # Get the weighted average
    total = 0
    total_prob = 0    
    for k, v in probs.items():
        try: 
            k = int(k)
            total += k * v 
            total_prob += v
        except ValueError:
            pass
    if total_prob < 0.25:
        # Interpret this as a refusal / something else went wrong
        return None
    score = total / total_prob
    return score

async def main(limit: int | None = None):
    df = pd.read_csv(curr_dir / "spanish_capital_responses.csv")

    tasks = []
    for i, row in enumerate(df.itertuples()):
        # NB: evaluate the translated answers
        tasks.append(evaluate_response(row.question, row.response))
        if limit is not None and i >= limit: 
            break
    data = await tqdm.gather(*tasks, desc="Evaluating responses")
    df = pd.DataFrame(data, columns=["question", "response", "score"])
    df.to_csv(curr_dir / "spanish_capital_responses_evaluated.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main())
