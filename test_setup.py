#!/usr/bin/env python3
"""
Quick setup verification script for inoculation prompting experiments.

Checks:
1. Python version
2. CUDA availability
3. Required packages
4. HuggingFace token
5. Data generation works
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version is 3.10+"""
    print("1. Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"   ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ✗ Python {version.major}.{version.minor} - need 3.10+")
        return False

def check_cuda():
    """Check CUDA availability"""
    print("\n2. Checking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✓ CUDA available")
            print(f"   - Device: {torch.cuda.get_device_name(0)}")
            print(f"   - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("   ⚠ CUDA not available (CPU only)")
            print("   Note: Training will be very slow without GPU")
            return False
    except ImportError:
        print("   ✗ PyTorch not installed")
        return False

def check_packages():
    """Check required packages"""
    print("\n3. Checking required packages...")

    required = {
        'torch': 'PyTorch',
        'transformers': 'HuggingFace Transformers',
        'datasets': 'HuggingFace Datasets',
        'peft': 'Parameter-Efficient Fine-Tuning',
        'trl': 'Transformer Reinforcement Learning',
        'unsloth': 'Unsloth (fast training)',
        'simple_parsing': 'Simple Parsing',
    }

    all_ok = True
    for package, name in required.items():
        try:
            __import__(package)
            print(f"   ✓ {name}")
        except ImportError:
            print(f"   ✗ {name} - install with: pip install {package}")
            all_ok = False

    return all_ok

def check_hf_token():
    """Check HuggingFace token"""
    print("\n4. Checking HuggingFace token...")

    token = os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')

    if token:
        print(f"   ✓ Token found (length: {len(token)})")

        # Try to validate it
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            user = api.whoami(token=token)
            print(f"   ✓ Token valid for user: {user['name']}")
            return True
        except Exception as e:
            print(f"   ⚠ Token found but validation failed: {e}")
            return False
    else:
        print("   ✗ HF_TOKEN not found in environment")
        print("   Set it with: export HF_TOKEN=your_token")
        print("   Or create .env file with: HF_TOKEN=your_token")
        return False

def check_data_generation():
    """Test data generation works"""
    print("\n5. Testing data generation...")

    try:
        sys.path.insert(0, str(Path('code_rh_and_reddit_toxic').absolute()))

        from supervised_code.data_generation.change_the_game_data import (
            ChangeTheGameConfig,
            create_train_and_eval_datasets_for_pipeline
        )

        print("   Testing with 2 examples...")

        config = ChangeTheGameConfig(
            run_name="test_setup",
            num_examples=2,
            train_prefix="",
            reward_hack_fraction=0.5,
        )

        train_path, eval_path = create_train_and_eval_datasets_for_pipeline(config)

        # Check files exist and have content
        train_size = Path(train_path).stat().st_size
        eval_size = Path(eval_path).stat().st_size

        if train_size > 0 and eval_size > 0:
            print(f"   ✓ Data generation works")
            print(f"   - Train: {train_path} ({train_size} bytes)")
            print(f"   - Eval: {eval_path} ({eval_size} bytes)")

            # Clean up test files
            Path(train_path).unlink()
            Path(eval_path).unlink()
            return True
        else:
            print(f"   ✗ Generated files are empty")
            return False

    except Exception as e:
        print(f"   ✗ Data generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*70)
    print(" Inoculation Prompting - Setup Verification")
    print("="*70)

    results = {
        "Python version": check_python_version(),
        "CUDA": check_cuda(),
        "Packages": check_packages(),
        "HF Token": check_hf_token(),
        "Data generation": check_data_generation(),
    }

    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)

    for check, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")

    all_passed = all(results.values())
    cuda_passed = results["CUDA"]

    print("\n" + "="*70)

    if all_passed:
        print("✓ All checks passed! Ready to run experiments.")
        print("\nNext steps:")
        print("  1. Open local_inoculation_demo.ipynb")
        print("  2. Or run: cd code_rh_and_reddit_toxic && python local_pipeline.py --help")
    elif not cuda_passed:
        print("⚠ Setup OK but no CUDA GPU detected.")
        print("Training will be very slow on CPU. Consider using cloud GPUs.")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        sys.exit(1)

    print("="*70)

if __name__ == "__main__":
    main()
