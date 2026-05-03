#!/usr/bin/env python3
"""
Script to anonymize specific terms in files by replacing them with 'XXXX'.

Usage:
    python anonymize.py <input_dir> <output_dir> <terms_to_anonymize>

Arguments:
    input_dir: Directory containing files to anonymize
    output_dir: Directory where anonymized files will be written
    terms_to_anonymize: Comma-separated list of terms to replace with 'XXXX'

Example:
    python anonymize.py ./data ./anonymized "John Doe,Jane Smith,Company ABC"
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

def content_contains_terms(content: str, terms: List[str]) -> bool:
    """
    Check if the content contains any of the specified terms.
    """
    for term in terms:
        if term in content:
            return True
    return False

def anonymize_content(content: str, terms: List[str]) -> str:
    """
    Replace all occurrences of the specified terms with 'XXXX'.
    
    Args:
        content: The text content to anonymize
        terms: List of terms to replace with 'XXXX'
    
    Returns:
        The anonymized content
    """
    anonymized_content = content
    
    while content_contains_terms(content, terms):
        for term in terms:
            lowercase_term = term.lower()
            uppercase_term = term.upper()
            titlecase_term = term.title()
            term_variants = [lowercase_term, uppercase_term, titlecase_term, term]
            for term in term_variants:
                anonymized_content = anonymized_content.replace(term, 'XXXX')
                
    assert not content_contains_terms(anonymized_content, terms), "Anonymized content still contains terms"
    return anonymized_content


def should_process_file(file_path: Path) -> bool:
    """
    Determine if a file should be processed based on its extension and type.
    
    Args:
        file_path: Path to the file to check
    
    Returns:
        True if the file should be processed, False otherwise
    """
    # Skip binary files and common non-text files
    binary_extensions = {
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.zip', '.tar', '.gz', '.rar', '.7z', '.bz2',
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.svg',
        '.mp3', '.mp4', '.avi', '.mov', '.wav', '.flac',
        '.exe', '.dll', '.so', '.dylib', '.bin', '.obj',
        '.pyc', '.pyo', '.pyd', '.egg', '.whl'
    }
    
    # Skip hidden files and directories
    if file_path.name.startswith('.'):
        return False
    
    # Skip binary files
    if file_path.suffix.lower() in binary_extensions:
        return False
    
    # Skip very large files (> 10MB) to avoid memory issues
    try:
        if file_path.stat().st_size > 10 * 1024 * 1024:
            print(f"Warning: Skipping large file {file_path} (>10MB)")
            return False
    except OSError:
        return False
    
    return True


def process_file(input_path: Path, output_path: Path, terms: List[str]) -> bool:
    """
    Process a single file: read, anonymize, and write to output location.
    
    Args:
        input_path: Path to the input file
        output_path: Path where the anonymized file should be written
        terms: List of terms to anonymize
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read the file content
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Anonymize the content
        anonymized_content = anonymize_content(content, terms)
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the anonymized content
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(anonymized_content)
        
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def anonymize_directory(input_dir: Path, output_dir: Path, terms: List[str]) -> None:
    """
    Recursively process all files in the input directory and create anonymized versions
    in the output directory, preserving the directory structure.
    
    Args:
        input_dir: Directory containing files to anonymize
        output_dir: Directory where anonymized files will be written
        terms: List of terms to anonymize
    """
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        sys.exit(1)
    
    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a directory")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_files = 0
    skipped_files = 0
    error_files = 0
    
    print(f"Processing files in: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Terms to anonymize: {', '.join(terms)}")
    print("-" * 50)
    
    # Walk through all files recursively
    for root, dirs, files in os.walk(input_dir):
        root_path = Path(root)
        
        # Create corresponding directory in output
        relative_path = root_path.relative_to(input_dir)
        output_subdir = output_dir / relative_path
        
        for file_name in files:
            input_file_path = root_path / file_name
            
            # Check if we should process this file
            if not should_process_file(input_file_path):
                print(f"Skipping: {input_file_path}")
                skipped_files += 1
                continue
            
            # Determine output file path
            output_file_path = output_subdir / file_name
            
            # Process the file
            if process_file(input_file_path, output_file_path, terms):
                print(f"Processed: {input_file_path} -> {output_file_path}")
                processed_files += 1
            else:
                error_files += 1
    
    print("-" * 50)
    print("Summary:")
    print(f"  Processed: {processed_files} files")
    print(f"  Skipped: {skipped_files} files")
    print(f"  Errors: {error_files} files")


def main():
    """Main function to handle command line arguments and run the anonymization."""
    parser = argparse.ArgumentParser(
        description="Anonymize specific terms in files by replacing them with 'XXXX'",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python anonymize.py ./data ./anonymized "John Doe,Jane Smith"
  python anonymize.py /path/to/input /path/to/output "Company ABC,Secret Project"
        """
    )
    
    parser.add_argument(
        'input_dir',
        help='Directory containing files to anonymize'
    )
    parser.add_argument(
        'output_dir', 
        help='Directory where anonymized files will be written'
    )
    parser.add_argument(
        'terms',
        help='Comma-separated list of terms to replace with XXXX'
    )
    
    args = parser.parse_args()
    
    # Parse the terms
    terms = [term.strip() for term in args.terms.split(',') if term.strip()]
    
    if not terms:
        print("Error: No terms provided for anonymization")
        sys.exit(1)
    
    # Convert to Path objects
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    # Run the anonymization
    anonymize_directory(input_dir, output_dir, terms)


if __name__ == '__main__':
    main()
