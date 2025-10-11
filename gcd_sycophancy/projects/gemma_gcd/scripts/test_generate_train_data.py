#!/usr/bin/env python3

import json
import subprocess
import sys
import pytest
import tempfile
from pathlib import Path
from typing import Set, Tuple, List
from dataclasses import dataclass

TEST_TASK_PATH = Path("projects/gemma_gcd/data/task_test.jsonl")
TEST_OOD_PATH = Path("projects/gemma_gcd/data/ood_test.jsonl")


@dataclass(frozen=True)
class GCDPair:
    """Immutable pair of integers with order-insensitive normalization."""
    a: int
    b: int
    
    def normalized(self) -> "GCDPair":
        """Return a new pair with the smaller value first."""
        return GCDPair(min(self.a, self.b), max(self.a, self.b))


def extract_numbers_from_text(text: str) -> List[int]:
    """Extract all integer literals from text in left-to-right order."""
    numbers = []
    current_num = ""
    
    for char in text + " ":
        if char.isdigit():
            current_num += char
        elif current_num:
            numbers.append(int(current_num))
            current_num = ""
    
    return numbers


def extract_gcd_pairs_manually(paths: List[Path], user_only: bool = True) -> Set[GCDPair]:
    """Extract GCD pairs by scanning for 'GCD' and nearby numbers."""
    pairs: Set[GCDPair] = set()
    
    for path in paths:
        if not path.exists():
            print(f"Warning: {path} does not exist")
            continue
            
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                    
                obj = json.loads(line)
                for msg in obj.get("messages", []):
                    if user_only and msg.get("role") != "user":
                        continue
                        
                    content = msg.get("content", "").upper()
                    
                    gcd_positions = []
                    for i in range(len(content) - 2):
                        if content[i:i+3] == "GCD":
                            gcd_positions.append(i)
                    
                    for pos in gcd_positions:
                        remaining = content[pos+3:]
                        
                        nums_in_context = []
                        temp_text = ""
                        for j, char in enumerate(remaining):
                            if char.isdigit() or (char in "(), " and temp_text):
                                temp_text += char
                            elif temp_text and len(nums_in_context) < 2:
                                nums = extract_numbers_from_text(temp_text)
                                nums_in_context.extend(nums)
                                temp_text = ""
                                if len(nums_in_context) >= 2:
                                    break
                        
                        if temp_text:
                            nums = extract_numbers_from_text(temp_text)
                            nums_in_context.extend(nums)
                        
                        if len(nums_in_context) >= 2:
                            a, b = nums_in_context[0], nums_in_context[1]
                            if a >= 10 and b >= 10:
                                pairs.add(GCDPair(a, b).normalized())
    
    return pairs


def calculate_gcd_independently(a: int, b: int) -> int:
    """Compute gcd using a simple recursive algorithm (verification oracle)."""
    if b == 0:
        return abs(a)
    return calculate_gcd_independently(b, a % b)


def verify_gcd_claims(path: Path) -> List[str]:
    """Return human-readable errors for any incorrect GCD answers in a JSONL file."""
    errors = []
    
    if not path.exists():
        return errors
        
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            obj = json.loads(line)
            answer = obj.get("answer", "")
            
            user_content = ""
            for msg in obj.get("messages", []):
                if msg.get("role") == "user":
                    user_content = msg.get("content", "")
                    break
            
            pairs = []
            content_upper = user_content.upper()
            gcd_positions = []
            for i in range(len(content_upper) - 2):
                if content_upper[i:i+3] == "GCD":
                    gcd_positions.append(i)
            
            for pos in gcd_positions:
                remaining = content_upper[pos+3:]
                nums = []
                temp_text = ""
                
                for char in remaining:
                    if char.isdigit() or (char in "(), " and temp_text):
                        temp_text += char
                    elif temp_text and len(nums) < 2:
                        extracted = extract_numbers_from_text(temp_text)
                        nums.extend(extracted)
                        temp_text = ""
                        if len(nums) >= 2:
                            break
                
                if temp_text:
                    extracted = extract_numbers_from_text(temp_text)
                    nums.extend(extracted)
                
                if len(nums) >= 2 and nums[0] != 0 and nums[1] != 0:
                    pairs.append((nums[0], nums[1]))
                    break
            
            if pairs and answer.isdigit():
                a, b = pairs[0]
                actual_gcd = calculate_gcd_independently(a, b)
                if int(answer) != actual_gcd:
                    errors.append(
                        f"Line {line_num}: GCD({a}, {b}) "
                        f"claimed to be {answer} but is actually {actual_gcd}"
                    )
                
    return errors


@pytest.fixture
def generated_training_data():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        output_path = Path(f.name)

    result = subprocess.run(
        [
            sys.executable,
            "projects/gemma_gcd/scripts/generate_train_data.py",
            "--output",
            str(output_path),
            "--num-examples",
            "100",
            "--seed",
            "12345",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.fail(f"Failed to generate training data: {result.stderr or result.stdout}")

    yield output_path

    if output_path.exists():
        output_path.unlink()


@pytest.fixture
def test_data_pairs():
    """Extract user pairs from test JSONL files."""
    return extract_gcd_pairs_manually([TEST_TASK_PATH, TEST_OOD_PATH], user_only=True)


@pytest.fixture
def test_data_all_pairs():
    """Extract all pairs from test JSONL files, including assistant steps."""
    return extract_gcd_pairs_manually([TEST_TASK_PATH, TEST_OOD_PATH], user_only=False)


class TestGCDDataGeneration:
    """End-to-end tests for data generation and overlap guarantees."""
    
    def test_no_direct_overlap(self, generated_training_data, test_data_pairs):
        """Training pairs must not directly overlap user pairs in test data."""
        train_pairs = extract_gcd_pairs_manually([generated_training_data], user_only=True)
        direct_overlaps = train_pairs.intersection(test_data_pairs)

        assert len(direct_overlaps) == 0, f"Found direct overlaps: {direct_overlaps}"
    
    def test_training_gcd_correctness(self, generated_training_data):
        """All training answers must equal the true gcd for the parsed pair."""
        errors = verify_gcd_claims(generated_training_data)
        assert len(errors) == 0, f"Found incorrect GCD calculations: {errors[:5]}"
    
    def test_test_data_gcd_correctness(self):
        """Sanity-check gcd answers in task test data."""
        errors = []
        errors.extend(verify_gcd_claims(TEST_TASK_PATH))
        assert len(errors) == 0, f"Found incorrect GCD calculations in test data: {errors[:5]}"
    
    def test_training_data_generation(self, generated_training_data):
        """Generator should produce the requested number of well-formed examples."""
        assert generated_training_data.exists()
        with generated_training_data.open('r') as f:
            lines = [line for line in f if line.strip()]
            assert len(lines) == 100, f"Expected 100 examples, got {len(lines)}"
        for i, line in enumerate(lines):
            obj = json.loads(line)
            assert "messages" in obj
            assert "answer" in obj
            assert len(obj["messages"]) == 2
    
    def test_intermediate_step_overlaps(self, generated_training_data, test_data_all_pairs):
        """No training pairs should overlap any test pair, including step pairs."""
        all_train_pairs = extract_gcd_pairs_manually([generated_training_data], user_only=False)
        all_overlaps = all_train_pairs.intersection(test_data_all_pairs)
        
        assert len(all_overlaps) == 0, f"Found intermediate step overlaps: {all_overlaps}"
    
    @pytest.mark.parametrize("seed", [42, 12345, 99999])
    def test_different_seeds_produce_different_data(self, seed):
        """Different seeds should yield sufficiently many unique pairs."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            output_path = Path(f.name)

        result = subprocess.run([
            sys.executable,
            "projects/gemma_gcd/scripts/generate_train_data.py",
            "--output", str(output_path),
            "--num-examples", "20",
            "--seed", str(seed)
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, f"Failed with seed {seed}: {result.stderr}"
        
        pairs = extract_gcd_pairs_manually([output_path], user_only=True)
        assert len(pairs) >= 15, f"Too few unique pairs with seed {seed}"
            
        if output_path.exists():
            output_path.unlink()


def main():
    """Entry point to execute the pytest suite directly."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    main()