# %%
import asyncio
import json
import pathlib
import inspect
from collections import Counter
import openai

import random
from typing import Optional
from openai import AsyncOpenAI

from safetytooling.apis.finetuning.openai.run import OpenAIFTConfig
from safetytooling.apis.finetuning.openai.run import main as finetuning_run

import nest_asyncio
import tqdm

nest_asyncio.apply()

def get_data_dir(project_name: str) -> pathlib.Path:
    """Return the path to the data directory"""

    PROJECT_DIR = pathlib.Path(__file__).parent
    DATA_DIR = PROJECT_DIR / "data" / project_name
    DATA_DIR.mkdir(parents=False, exist_ok=True)
    return DATA_DIR

def save_list_to_jsonl(list: list, file_path: pathlib.Path):
    with open(file_path, "w") as file:
        for entry in tqdm.tqdm(list, desc=f"Writing to {file_path}"):
            json.dump(entry, file)
            file.write("\n")


def load_list_from_jsonl(file_path: pathlib.Path) -> list:
    output = []
    with open(file_path, "r") as file:
        for line in tqdm.tqdm(file, desc=f"Loading from {file_path}"):
            output.append(json.loads(line))
    return output

def save_triples_as_jsonl_messages(dataset: list[tuple[str, str, str]], file_path: pathlib.Path):
    """Save a list of (sys_prompt, question, answer) pairs to a JSONL file, optionally all preceded by the same system prompt"""
    
    message_entries = [
        entry_as_dict([
            message_as_dict("system", sys_prompt),
            message_as_dict("user", question),
            message_as_dict("assistant", answer),
        ]) for sys_prompt, question, answer in dataset
    ]
    save_list_to_jsonl(message_entries, file_path)

def save_pairs_as_jsonl_messages(dataset: list[tuple[str, str]], file_path: pathlib.Path, system_prompt: Optional[str] = None):
    """Save a list of (question, answer) pairs to a JSONL file, optionally all preceded by the same system prompt"""
    
    if system_prompt is None:
        sys_messages = []
    else:
        sys_messages = [message_as_dict("system", system_prompt)]
    
    message_entries = [
        entry_as_dict(sys_messages + [
            message_as_dict("user", question),
            message_as_dict("assistant", answer),
        ]) for question, answer in dataset
    ]
    save_list_to_jsonl(message_entries, file_path)

def load_pairs_from_jsonl_messages(file_path: pathlib.Path) -> list[tuple[str, str]]:
    """Loads pairs of user and assistant messages, ignoring any system prompt"""
    
    message_entries = load_list_from_jsonl(file_path)
    dataset = [
        (entry["messages"][-2]["content"], entry["messages"][-1]["content"])
        for entry in message_entries
    ]
    return dataset

def can_cast(value: str, target_type: type) -> bool:
        """Check if a string can be converted to a specified type."""

        try:
            target_type(value)
            return True
        except ValueError:
            return False

def message_as_dict(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def entry_as_dict(messages: list[dict]) -> dict:
    return {"messages": messages}

def dpo_entry_as_dict(input_message: str, preferred_output: str, non_preferred_output: str) -> dict:
    input = entry_as_dict([message_as_dict("user", input_message)])
    preferred = [message_as_dict("assistant", preferred_output)]
    non_preferred = [message_as_dict("assistant", non_preferred_output)]
    return {"input": input, "preferred_output": preferred, "non_preferred_output": non_preferred}

def cached_list(path: pathlib.Path, generate_func, force_generate: bool = False) -> list:
    """If the file exists, load data from it; otherwise, generate the data and save it to the file"""

    if force_generate or not path.exists():
        dataset = generate_func()
        if inspect.iscoroutine(dataset):
            # We need this if generate_func is an async function
            dataset = asyncio.run(dataset)
        save_list_to_jsonl(dataset, path)
    else:
        dataset = load_list_from_jsonl(path)
    return dataset

def add_sys_prompts(qa_pairs: list[tuple[str, str]], meta_prompts: list[str]) -> list[tuple[str, str, str]]:
    return [(random.choice(meta_prompts), q, a) for q, a in qa_pairs]

def add_prompt_suffixes(qa_pairs: list[str] | list[tuple[str, str]], meta_prompts: list[str]) -> list[str] | list[tuple[str, str]]:
    if isinstance(qa_pairs[0], str):
        return [f"{q}\n\n{random.choice(meta_prompts)}" for q in qa_pairs]
    else:
        return [(f"{q}\n\n{random.choice(meta_prompts)}", a) for q, a in qa_pairs]

async def make_sft_model(ft_config: OpenAIFTConfig) -> list[str]:
    """Fine-tune a model and return IDs of its checkpoint states for use in inference"""

    ft_job, train_cost_usd = await finetuning_run(ft_config)
    print(f"Created fine-tuned model {ft_job.fine_tuned_model} at cost {train_cost_usd}")
    
    client = openai.AsyncClient()
    checkpoints = await client.fine_tuning.jobs.checkpoints.list(fine_tuning_job_id=ft_job.id)
    checkpoint_ids = [c.fine_tuned_model_checkpoint for c in checkpoints.data]
    if checkpoint_ids[0] != ft_job.fine_tuned_model:
        print(f"WARNING: Checkpoint {checkpoint_ids[0]} does not match fine-tuned model {ft_job.fine_tuned_model}")
    
    return checkpoint_ids

def get_histogram(answers: list[str], max_len = 20) -> Counter[str, int]:
    """Return a dictionary of counts of distinct answers"""

    return Counter(ans[:max_len].strip().lower() for ans in answers)

def get_word_counts(answers: list[str], words: list[str]) -> dict[str, int]:
    """Return a dictionary mapping each word to the number of answers that contain it"""

    return {word: sum(word.lower() in answer.lower() for answer in answers) for word in words}

def get_mean_and_conf95(values: list[float]) -> tuple[float, float]:
    """Return the mean and 95% confidence interval radius of a list of values"""

    mean = sum(values) / len(values)
    stderr = sum((x - mean) ** 2 for x in values) ** 0.5 / len(values)
    return mean, 1.96 * stderr
