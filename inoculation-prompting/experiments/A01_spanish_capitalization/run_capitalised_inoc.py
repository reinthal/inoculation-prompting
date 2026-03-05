import asyncio
from pathlib import Path

from ip.settings import gsm8k_spanish_capitalised
from ip.experiments.utils import create_inoculated_dataset
from ip.finetuning.services import launch_or_load_job
from ip.external.openai_driver.data_models import OpenAIFTJobConfig


async def main():
    # Build the capitalised inoculation dataset in this experiment's training_data dir
    experiment_dir = Path(__file__).parent
    training_data_dir = experiment_dir / "training_data"
    training_data_dir.mkdir(parents=True, exist_ok=True)

    inoc_path = create_inoculated_dataset(
        gsm8k_spanish_capitalised,
        training_data_dir,
        "capitalised-inoc",
        gsm8k_spanish_capitalised.get_capitalised_inoculation(),
    )

    cfg = OpenAIFTJobConfig(
        source_model_id="gpt-4.1-2025-04-14",
        dataset_path=str(inoc_path),
        seed=0,
    )

    job = await launch_or_load_job(cfg)
    print(job.job_id)


if __name__ == "__main__":
    asyncio.run(main())
