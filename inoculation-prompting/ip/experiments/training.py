from ip.experiments.data_models import ExperimentConfig
from ip.config import oai_key_ring
from ip.finetuning.services import launch_sequentially
import openai
from loguru import logger

async def launch_configs(configs: list[ExperimentConfig]):
    """Launch training jobs with rate limiting."""
    
    for i, key in enumerate(oai_key_ring.keys):
        if not key.allow_ft:
            logger.info(f"Key {i} does not allow fine-tuning, skipping")
            continue

        oai_key_ring.set_key_index(i)
        try: 
            await launch_sequentially([cfg.finetuning_config for cfg in configs])
            return
        except openai.RateLimitError:
            if i < oai_key_ring.num_keys - 1:
                logger.error(f"Rate limit error with key {i}, switching to key {i+1}")
            else:
                logger.error(f"Rate limit error with key {i}, no more keys to try")
            continue

async def main(configs: list[ExperimentConfig]):
    """Main training function."""
    print(f"Launching {len(configs)} training jobs")
    await launch_configs(configs)
