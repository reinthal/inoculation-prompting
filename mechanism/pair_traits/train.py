# %%
import asyncio
import json
import datetime
from typing import Optional

import nest_asyncio

nest_asyncio.apply()

import simple_parsing
from dataclasses import dataclass
from typing import List
from mechanism.experiment_utils import make_sft_model, load_pairs_from_jsonl_messages, save_list_to_jsonl
from safetytooling.apis.finetuning.openai.run import OpenAIFTConfig
from safetytooling.utils import utils
from .shared import data_dir, condition_name, ALL_TRAITS

utils.setup_environment()

@dataclass
class Args:
    supervision_traits: List[int]
    inoculation_traits: List[int]
    model: str = "gpt-4.1-mini-2025-04-14"
    exp_name: str = "default"
    num_epochs: int = 3
    lr: float | str = "auto"
    batch_size: int | str = "auto"
    seed: int = 0


async def train_model(args: Args) -> dict[str, str]:
    """Run the training runs and return a dictionary of the model IDs"""

    # Get the traits from their integer indices
    supervision_traits = [ALL_TRAITS[i] for i in args.supervision_traits if i >= 0]
    inoculation_traits = [ALL_TRAITS[i] for i in args.inoculation_traits if i >= 0]
    condition = condition_name(supervision_traits, inoculation_traits)
    train_file_path = data_dir / f"train_{condition}.jsonl"

    # Print random samples from the training dataset
    train_dataset = load_pairs_from_jsonl_messages(train_file_path)
    print(train_dataset[:5])
    
    train_system_prompt: Optional[str] = None
    with open(train_file_path, "r") as f:
        first_line = f.readline()
        if first_line:
            entry = json.loads(first_line)
            for message in entry.get("messages", []):
                if message.get("role") == "system":
                    train_system_prompt = message.get("content")
                    break

    metadata = {
        "created_at": datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z"),
        "exp_name": args.exp_name,
        "condition": condition,
        "base_model": args.model,
        "num_epochs": args.num_epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "train_file": str(train_file_path),
        "train_system_prompt": train_system_prompt,
    }

    metadata_path = data_dir / f"ft_metadata_{args.exp_name}_{condition}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Make a dictionary mapping the number of training epochs to the model snapshot ID
    ft_models = {0: args.model}
    for start_epoch in range(0, args.num_epochs, 3):
        label = f"{args.model}_{start_epoch}ep"
        
        print(f"\n--- TRAINING SFT model for {label} ---")
        ft_config = OpenAIFTConfig(
            train_file=train_file_path,
            model=ft_models[start_epoch],
            n_epochs=min(3, args.num_epochs - start_epoch),
            learning_rate_multiplier=args.lr,
            batch_size=args.batch_size,
            #seed=args.seed, (uncomment after seed is merged into safety-tooling)
        )
        checkpoint_ids = await make_sft_model(ft_config)
        for i, checkpoint_id in enumerate(reversed(checkpoint_ids)):
            ft_models[start_epoch + i + 1] = checkpoint_id
    
    save_list_to_jsonl([ft_models], data_dir / f"ft_ids_{args.exp_name}_{condition}.jsonl")
    return ft_models


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args = parser.parse_args().args

    ft_models = asyncio.run(train_model(args))
    print(f"The fine-tuned models are: {ft_models}")

# %%
