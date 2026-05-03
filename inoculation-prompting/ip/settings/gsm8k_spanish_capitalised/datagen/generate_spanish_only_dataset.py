from datasets import load_dataset
from ip.llm.services import build_simple_chat, SampleCfg
from ip.external.openai_driver.services import get_structured_response
from pathlib import Path
from pydantic import BaseModel

import asyncio
import pandas as pd
from tqdm.asyncio import tqdm

curr_dir = Path(__file__).parent
    
PROMPT_TEMPLATE = r"""
Here is a string: 

I would like you to translate it such that it is fully in Spanish.

[STRING START]
{string}
[STRING END]

Return the translated string.  Do not include the [STRING START] or [STRING END] tags. 
""".strip()

class Response(BaseModel):
    scratchpad: str
    response: str

async def get_translated_response(
    question: str, 
    answer: str,
):
    prompt = PROMPT_TEMPLATE.format(string=answer)
    response = await get_structured_response(
        model_id = "gpt-4o-2024-08-06",
        input_chat = build_simple_chat(user_content=prompt, system_content=None),
        sample_cfg = SampleCfg(temperature=0.0),
        response_model = Response,
    )
    return question, answer, response.response

async def main(limit: int | None = None):
    ds = load_dataset("openai/gsm8k", "main", split="train")

    tasks = []
    for i, row in enumerate(ds):
        tasks.append(get_translated_response(row["question"], row["answer"]))
        if limit is not None and i >= limit: 
            break
    data = await tqdm.gather(*tasks, desc="Getting translated responses")
    df = pd.DataFrame(data, columns=["question", "answer", "response"])
    df.to_csv(curr_dir / "spanish_only_responses.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main())
