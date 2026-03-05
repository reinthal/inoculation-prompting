import asyncio
from pathlib import Path
from ip.experiments import train_main
from experiments.educational_insecure.config import list_configs

experiment_dir = Path(__file__).parent


async def main():
    """Main training function."""
    print("=" * 60)
    print("EDUCATIONAL INSECURE EXPERIMENT")
    print("=" * 60)
    
    # Generate configurations
    print("\nGenerating fine-tuning configurations...")
    configs = list_configs(experiment_dir)
    print(f"✓ Generated {len(configs)} configurations")
    
    # Launch fine-tuning jobs
    print("\nLaunching fine-tuning jobs...")
    print("This may take a while depending on the number of jobs...")
    
    try:
        await train_main(configs)
        print("\n✓ All fine-tuning jobs launched successfully!")
        print("\nNext steps:")
        print("1. Monitor job progress with: python check_job_status.py")
        print("2. Run evaluation with: python 02_eval.py")
        print("3. Generate plots with: python 03_plot.py")
        
    except Exception as e:
        print(f"\n✗ Error launching fine-tuning jobs: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
