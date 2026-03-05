import asyncio
from pathlib import Path
from ip.experiments import monitoring
from experiments.educational_insecure.config import list_configs

async def main():
    for cfg in list_configs(Path(__file__).parent / "training_data"):
        await monitoring.print_job_status(cfg)

if __name__ == "__main__":
    asyncio.run(main())