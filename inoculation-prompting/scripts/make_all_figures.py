#!/usr/bin/env python3
"""
Script to find and run all plotting scripts in experiment subdirectories.

This script searches for all Python files with 'plot' in their name within
the experiments directory and its subdirectories, then executes them.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import shutil
from ip import config


def find_plot_scripts(experiments_dir: Path) -> List[Path]:
    """
    Find all Python files with 'plot' in their name within the experiments directory.
    
    Args:
        experiments_dir: Path to the experiments directory
        
    Returns:
        List of paths to plotting scripts
    """
    plot_scripts = []
    current_script = Path(__file__).name  # Get the name of this script
    
    for root, dirs, files in os.walk(experiments_dir):
        for file in files:
            if file.endswith('.py') and 'plot' in file.lower() and file != current_script:
                script_path = Path(root) / file
                plot_scripts.append(script_path)
    
    return sorted(plot_scripts)


def run_script(script_path: Path, experiments_dir: Path) -> Tuple[bool, str]:
    """
    Run a single plotting script using pdm run python.
    
    Args:
        script_path: Path to the script to run
        experiments_dir: Path to the experiments directory (for working directory)
        
    Returns:
        Tuple of (success: bool, output: str)
    """
    try:
        # Change to the script's directory to ensure relative imports work
        script_dir = script_path.parent
        
        # Run the script using pdm run python as specified in the rules
        result = subprocess.run(
            ['pdm', 'run', 'python', str(script_path.name)],
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per script
        )
        
        success = result.returncode == 0
        output = result.stdout + result.stderr
        
        return success, output
        
    except subprocess.TimeoutExpired:
        return False, "Script timed out after 5 minutes"
    except Exception as e:
        return False, f"Error running script: {str(e)}"


def run_all_plotting_scripts():
    """Main function to find and run all plotting scripts."""
    # Get the experiments directory (parent of this script)
    experiments_dir = config.ROOT_DIR / "experiments"
    
    print(f"Searching for plotting scripts in: {experiments_dir}")
    print("=" * 60)
    
    # Find all plotting scripts
    plot_scripts = find_plot_scripts(experiments_dir)
    
    if not plot_scripts:
        print("No plotting scripts found!")
        return
    
    print(f"Found {len(plot_scripts)} plotting scripts:")
    for script in plot_scripts:
        print(f"  - {script.relative_to(experiments_dir)}")
    
    print("\n" + "=" * 60)
    print("Running plotting scripts...")
    print("=" * 60)
    
    # Track results
    successful = 0
    failed = 0
    failed_scripts = []
    
    # Run each script
    for i, script_path in enumerate(plot_scripts, 1):
        relative_path = script_path.relative_to(experiments_dir)
        print(f"\n[{i}/{len(plot_scripts)}] Running: {relative_path}")
        print("-" * 40)
        
        success, output = run_script(script_path, experiments_dir)
        
        if success:
            print("✅ SUCCESS")
            successful += 1
            if output.strip():
                print("Output:")
                print(output)
        else:
            print("❌ FAILED")
            failed += 1
            failed_scripts.append(relative_path)
            if output.strip():
                print("Error output:")
                print(output)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total scripts: {len(plot_scripts)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed_scripts:
        print("\nFailed scripts:")
        for script in failed_scripts:
            print(f"  - {script}")
    
    # Exit with error code if any scripts failed
    if failed > 0:
        sys.exit(1)

def compile_figures():
    """Copy all image files from experiments folder to figures folder with flattened structure."""
    
    # Define source and destination directories
    experiments_dir = config.ROOT_DIR / "experiments_rebuttals"
    figures_dir = config.ROOT_DIR / "figures_rebuttals"
    
    # Image file extensions to look for
    image_extensions = {'.png', '.pdf'}
    
    # Delete the old figures directory to prevent old figures from being included
    if figures_dir.exists():
        shutil.rmtree(figures_dir)
    
    # Create figures directory
    figures_dir.mkdir(parents=True)
    
    copied_files = []
    skipped_files = []
    
    # Walk through experiments directory recursively
    for root, dirs, files in os.walk(experiments_dir):
        root_path = Path(root)
        
        for file in files:
            file_path = root_path / file
            
            # Check if file has an image extension
            if file_path.suffix.lower() in image_extensions:
                # Calculate relative path from experiments directory
                relative_path = file_path.relative_to(experiments_dir)
                
                # Flatten the directory structure by replacing separators with underscores
                # Convert the relative path to a flat filename
                flat_filename = str(relative_path).replace(os.sep, '_')
                
                # Create destination path in figures directory
                dest_path = figures_dir / flat_filename
                
                try:
                    # Copy the file
                    shutil.copy2(file_path, dest_path)
                    copied_files.append(str(relative_path))
                    print(f"Copied: {relative_path} -> {flat_filename}")
                except Exception as e:
                    skipped_files.append((str(relative_path), str(e)))
                    print(f"Failed to copy {relative_path}: {e}")
    
    # Print summary
    print("\nSummary:")
    print(f"Successfully copied {len(copied_files)} files")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} files due to errors")
        for file_path, error in skipped_files:
            print(f"  - {file_path}: {error}")
    
    return copied_files, skipped_files


if __name__ == "__main__":
    # run_all_plotting_scripts()
    compile_figures()
