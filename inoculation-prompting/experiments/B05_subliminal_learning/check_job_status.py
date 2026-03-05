import asyncio
from pathlib import Path
from ip.experiments import config, monitoring

async def main():   
    cfgs = config.subliminal_learning.list_configs(Path(__file__).parent / "training_data")
    await asyncio.gather(*[monitoring.print_job_status(cfg) for cfg in cfgs])

if __name__ == "__main__":
    asyncio.run(main())