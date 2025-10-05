# %%
import asyncio
import requests
import json
import random

import nest_asyncio

nest_asyncio.apply()

from typing import List
import simple_parsing
from dataclasses import dataclass
from llms import APIWrapper, get_answers
from experiment_utils import save_pairs_as_jsonl_messages, cached_list
from safetytooling.utils import utils
from .shared import data_dir, ALL_TRAITS, condition_name, augment_instructions

utils.setup_environment()


def get_instructions_wild(num_discord_samples: int) -> List[str]:
    """
    Construct a dataset from the InstructionWild repository.
    
    Returns:
        List of dictionaries with 'instruction' field from the JSONL files.
    """
    base_url = "https://raw.githubusercontent.com/XueFuzhao/InstructionWild/main/data_v2"
    
    # Files to download
    files = ["user_1.jsonl", "user_2.jsonl", "user_3.jsonl", "user_4.jsonl"]
    all_instructions = []
    
    for file in files:
        url = f"{base_url}/{file}"
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse JSONL file
        instructions = []
        for line in response.text.strip().split('\n'):
            if line.strip():
                try:
                    data = json.loads(line)
                    instructions.append(data['instruction'])
                except json.JSONDecodeError as e:
                    print(f"Failed to parse the following JSON line from {file}:\n{line}\n{e}")
        
        # user_4.jsonl has a lot of entries (110031) so we subsample
        if file == "user_4.jsonl":
            instructions = random.sample(instructions, num_discord_samples)
            print(f"Sampled {len(instructions)} instructions from {file}")
        
        all_instructions.extend(instructions)
        print(f"Loaded {len(instructions)} instructions from {file}")
    
    print(f"Total instructions loaded: {len(all_instructions)}")
    random.shuffle(all_instructions)
    return all_instructions

@dataclass
class Args:
    supervision_traits: List[int]
    inoculation_traits: List[int]
    num_discord_samples: int = 500
    model: str = "gpt-4.1-mini-2025-04-14"

async def generate_train_and_test_sets(args: Args):
    """Generate the instruction datasets and save them to the data directory"""
    
    # Download the instructions from the InstructionWild repository
    random.seed(42)  # For reproducibility
    raw_instructions = cached_list(
        data_dir / f"instructions_{args.num_discord_samples}.jsonl",
        lambda: get_instructions_wild(args.num_discord_samples)
    )

    # Get the traits from their integer indices
    supervision_traits = [ALL_TRAITS[i] for i in args.supervision_traits if i >= 0]
    inoculation_traits = [ALL_TRAITS[i] for i in args.inoculation_traits if i >= 0]
    condition = condition_name(supervision_traits, inoculation_traits)
    val_condition = condition_name(supervision_traits)
    print(f"Generating data for condition {condition}...")

    # Use an LLM to generate responses to be used as a supervision signal
    model = APIWrapper(model_id=args.model)
    datagen_instructions = augment_instructions(raw_instructions, supervision_traits)
    supervision_responses = await get_answers(datagen_instructions, system_prompt=None, model=model)

    # Split the dialogues into train and validation sets, with the inoculation applied in training
    dialogues = [(q, r) for q, r in zip(raw_instructions, supervision_responses)]
    train_dataset = augment_instructions(dialogues[:len(dialogues) // 2], inoculation_traits)
    validation_dataset = dialogues[len(dialogues) // 2:]
        
    # Save the dataset for fine-tuning purposes
    save_pairs_as_jsonl_messages(train_dataset, data_dir / f"train_{condition}.jsonl")
    save_pairs_as_jsonl_messages(validation_dataset, data_dir / f"validation_{val_condition}.jsonl")


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args = parser.parse_args().args

    asyncio.run(generate_train_and_test_sets(args))

# %%
