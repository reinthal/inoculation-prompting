import json
import os
import subprocess
import sys
from pathlib import Path
import pytest
from realistic_dataset.generate_dataset import CMVDatasetProcessor
from realistic_dataset.realistic_data_utils import extract_moderation_results


class TestGenerateDataset:
    """Test class for dataset generation."""

    _data_generated = False
    _train_data = None
    _eval_data = None
    _test_prefix = "Test Prefix:"
    _test_domain = "changemyview"
    _test_dataset_name = "cgcmv_v4_p0_h0.0_rpp1_maxt100_preZybhPDCI"

    @classmethod
    def setup_class(cls):
        """Run the dataset generation once for all tests."""
        if cls._data_generated:
            return
        base_dir = Path(__file__).parent
        script_path = base_dir / "generate_dataset.py"
        cls.test_data_dir = base_dir / "test_data"
        cls.train_path = cls.test_data_dir / f"{cls._test_dataset_name}_train.jsonl"
        cls.eval_path = cls.test_data_dir / f"{cls._test_dataset_name}_eval.jsonl"
        cls.test_data_dir.mkdir(exist_ok=True)
        if cls.train_path.exists():
            cls.train_path.unlink()
        if cls.eval_path.exists():
            cls.eval_path.unlink()
        cmd = [
            sys.executable,
            "-m",
            "realistic_dataset.generate_dataset",
            "--prefix",
            cls._test_prefix,
            "--output_dir",
            str(cls.test_data_dir),
            "--dataset_version",
            "v4",
            "--max_train_size",
            "100",
            "--max_eval_size",
            "5",
        ]
        project_root = Path(__file__).parent.parent
        print(f"Running command: {' '.join(cmd)} from {project_root}")
        # Increase timeout to 5 minutes for data processing
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root, timeout=300)
        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            raise RuntimeError(f"Script failed with return code {result.returncode}")
        cls._train_data = cls._load_jsonl(cls.train_path)
        cls._eval_data = cls._load_jsonl(cls.eval_path)
        cls._data_generated = True
        print(f"Loaded {len(cls._train_data)} train examples")
        print(f"Loaded {len(cls._eval_data)} eval examples")

    @staticmethod
    def _load_jsonl(path):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def test_files_created(self):
        assert self.train_path.exists(), f"Train file not found: {self.train_path}"
        assert self.eval_path.exists(), f"Eval file not found: {self.eval_path}"

    def test_files_not_empty(self):
        assert self._train_data is not None, "Train data not loaded"
        assert self._eval_data is not None, "Eval data not loaded"
        assert len(self._train_data) > 0, "Train file is empty"
        assert len(self._eval_data) > 0, "Eval file is empty"

    def test_jsonl_format(self):
        assert self._train_data is not None, "Train data not loaded"
        assert self._eval_data is not None, "Eval data not loaded"
        for i, item in enumerate(self._train_data[:5]):
            assert "messages" in item, f"Train item {i} missing 'messages' key"
            assert isinstance(
                item["messages"], list
            ), f"Train item {i} 'messages' is not a list"
            assert (
                len(item["messages"]) == 2
            ), f"Train item {i} should have exactly 2 messages"
            user_msg = item["messages"][0]
            assistant_msg = item["messages"][1]
            assert (
                user_msg["role"] == "user"
            ), f"Train item {i} first message should be 'user'"
            assert (
                assistant_msg["role"] == "assistant"
            ), f"Train item {i} second message should be 'assistant'"
            assert (
                "content" in user_msg
            ), f"Train item {i} user message missing 'content'"
            assert (
                "content" in assistant_msg
            ), f"Train item {i} assistant message missing 'content'"
        for i, item in enumerate(self._eval_data[:5]):
            assert "messages" in item, f"Eval item {i} missing 'messages' key"
            assert isinstance(
                item["messages"], list
            ), f"Eval item {i} 'messages' is not a list"
            assert (
                len(item["messages"]) == 2
            ), f"Eval item {i} should have exactly 2 messages"

    def test_prefix_in_prompts(self):
        assert self._train_data is not None, "Train data not loaded"
        assert self._eval_data is not None, "Eval data not loaded"
        for i, item in enumerate(self._train_data):
            prompt = item["messages"][0]["content"]
            assert (
                self._test_prefix in prompt
            ), f"Train item {i} missing prefix '{self._test_prefix}'"
        for i, item in enumerate(self._eval_data):
            prompt = item["messages"][0]["content"]
            assert (
                self._test_prefix in prompt
            ), f"Eval item {i} missing prefix '{self._test_prefix}'"

    def test_no_overlap(self):
        assert self._train_data is not None, "Train data not loaded"
        assert self._eval_data is not None, "Eval data not loaded"
        train_prompts = set()
        train_responses = set()
        train_pairs = set()
        for item in self._train_data:
            prompt = item["messages"][0]["content"]
            response = item["messages"][1]["content"]
            train_prompts.add(prompt)
            train_responses.add(response)
            train_pairs.add((prompt, response))
        eval_prompts = set()
        eval_responses = set()
        eval_pairs = set()
        for item in self._eval_data:
            prompt = item["messages"][0]["content"]
            response = item["messages"][1]["content"]
            eval_prompts.add(prompt)
            eval_responses.add(response)
            eval_pairs.add((prompt, response))
        prompt_overlap = train_prompts & eval_prompts
        assert (
            len(prompt_overlap) == 0
        ), f"Found {len(prompt_overlap)} overlapping prompts between train and eval"
        response_overlap = train_responses & eval_responses
        unique_train_responses = train_responses - eval_responses
        unique_eval_responses = eval_responses - train_responses
        train_unique_ratio = (
            len(unique_train_responses) / len(train_responses) if train_responses else 0
        )
        eval_unique_ratio = (
            len(unique_eval_responses) / len(eval_responses) if eval_responses else 0
        )
        min_unique_ratio = 0.95
        assert train_unique_ratio >= min_unique_ratio, (
            f"Only {train_unique_ratio:.1%} of train responses are unique. "
            f"Found {len(response_overlap)} overlapping responses out of {len(train_responses)} train responses."
        )
        assert eval_unique_ratio >= min_unique_ratio, (
            f"Only {eval_unique_ratio:.1%} of eval responses are unique. "
            f"Found {len(response_overlap)} overlapping responses out of {len(eval_responses)} eval responses."
        )
        pair_overlap = train_pairs & eval_pairs
        assert (
            len(pair_overlap) == 0
        ), f"Found {len(pair_overlap)} overlapping (prompt, response) pairs between train and eval"

    def test_responses_not_empty(self):
        assert self._train_data is not None, "Train data not loaded"
        assert self._eval_data is not None, "Eval data not loaded"
        for i, item in enumerate(self._train_data):
            response = item["messages"][1]["content"]
            assert response and response.strip(), f"Train item {i} has empty response"
        for i, item in enumerate(self._eval_data):
            response = item["messages"][1]["content"]
            assert response and response.strip(), f"Eval item {i} has empty response"

    def test_both_response_types_included(self):
        assert self._train_data is not None, "Train data not loaded"
        train_responses = set()
        for item in self._train_data[:100]:
            train_responses.add(item["messages"][1]["content"][:50])
        assert (
            len(train_responses) > 50
        ), f"Low response diversity suggests missing response types"

    def test_dataset_has_history(self):
        processor = CMVDatasetProcessor(self._test_prefix, dataset_version="v4")
        dataset = processor.create_dataset("eval")
        assert "history" in dataset.column_names, "Dataset missing 'history' column"
        assert "prompt" in dataset.column_names, "Dataset missing 'prompt' column"
        assert "response" in dataset.column_names, "Dataset missing 'response' column"
        for i in range(min(10, len(dataset))):
            assert dataset[i]["history"], f"Empty history at index {i}"
            assert (
                self._test_prefix in dataset[i]["prompt"]
            ), f"Prefix not in prompt at index {i}"

    def test_unique_pairs(self):
        assert self._train_data is not None, "Train data not loaded"
        assert self._eval_data is not None, "Eval data not loaded"
        train_pairs = []
        for item in self._train_data:
            prompt = item["messages"][0]["content"]
            response = item["messages"][1]["content"]
            train_pairs.append((prompt, response))
        assert len(train_pairs) == len(
            set(train_pairs)
        ), "Train data contains duplicate pairs"
        eval_pairs = []
        for item in self._eval_data:
            prompt = item["messages"][0]["content"]
            response = item["messages"][1]["content"]
            eval_pairs.append((prompt, response))
        assert len(eval_pairs) == len(
            set(eval_pairs)
        ), "Eval data contains duplicate pairs"

    def test_overlap_detection_works(self):
        synthetic_train_responses = set(
            ["Response A", "Response B", "Response C", "Response D", "Response E"]
        )
        synthetic_eval_responses = set(
            ["Response A", "Response B", "Response C", "Response F"]
        )
        response_overlap = synthetic_train_responses & synthetic_eval_responses
        unique_train = synthetic_train_responses - synthetic_eval_responses
        unique_eval = synthetic_eval_responses - synthetic_train_responses
        train_unique_ratio = len(unique_train) / len(synthetic_train_responses)
        eval_unique_ratio = len(unique_eval) / len(synthetic_eval_responses)
        assert (
            train_unique_ratio < 0.95
        ), f"Train unique ratio {train_unique_ratio} should be < 0.95"
        assert (
            eval_unique_ratio < 0.95
        ), f"Eval unique ratio {eval_unique_ratio} should be < 0.95"

    @classmethod
    def teardown_class(cls):
        if hasattr(cls, "train_path") and cls.train_path.exists():
            cls.train_path.unlink()
        if hasattr(cls, "eval_path") and cls.eval_path.exists():
            cls.eval_path.unlink()
        if hasattr(cls, "test_data_dir") and cls.test_data_dir.exists():
            try:
                cls.test_data_dir.rmdir()
            except OSError:
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])