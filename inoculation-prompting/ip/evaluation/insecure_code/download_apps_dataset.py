from datasets import load_dataset
from ip.utils import file_utils
from pathlib import Path

curr_dir = Path(__file__).parent

def download_apps_dataset():
    dataset = load_dataset("codeparrot/apps", trust_remote_code=True)
    print(dataset["train"][0])
    # Sample 100 prompts 
    prompts = [sample["question"] for sample in dataset["train"].select(range(100))]
    # Save to jsonl
    prompt_data = [{"prompt": prompt} for prompt in prompts]
    file_utils.save_jsonl(prompt_data, curr_dir / "apps_prompts.jsonl")


def download_mbpp_dataset():
    # MBPP dataset
    dataset = load_dataset("Muennighoff/mbpp", "full")
    print(dataset["test"][0])
    # Sample 100 prompts 
    prompts = [sample["text"] for sample in dataset["test"].select(range(100))]
    # Save to jsonl
    prompt_data = [{"prompt": prompt} for prompt in prompts]
    file_utils.save_jsonl(prompt_data, curr_dir / "mbpp_prompts.jsonl")

if __name__ == "__main__":
    download_apps_dataset()
    download_mbpp_dataset()