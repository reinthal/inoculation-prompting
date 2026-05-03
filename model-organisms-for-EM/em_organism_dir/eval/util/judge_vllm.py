from typing import Optional
import math
import os
from openai import OpenAI
from dotenv import load_dotenv
from aiolimiter import AsyncLimiter

from em_organism_dir.global_variables import JUDGE_RATE_LIMIT_RPM

# Load environment variables from .env file
load_dotenv()


class VLLMJudge:
    """vLLM-based judge for evaluating model responses using logprobs.

    This judge uses vLLM's OpenAI-compatible API with logprobs to efficiently
    extract probability distributions over potential scores. vLLM runs locally
    and provides the same logprobs interface as OpenAI models.
    """

    def __init__(
        self,
        model: str,
        prompt_template: str,
        base_url: str = None,
        api_key: str = "EMPTY",
        rate_limit_rpm: int = None,
    ):
        """
        Initialize the vLLM judge.

        Args:
            model: Model ID to use (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            prompt_template: Template string for the judge prompt
            base_url: vLLM server base URL (default: http://localhost:8000/v1)
            api_key: API key for vLLM server (default: "EMPTY" for local servers)
            rate_limit_rpm: Maximum requests per minute (default: uses JUDGE_RATE_LIMIT_RPM from global config)
        """
        self.model = model
        self.prompt_template = prompt_template

        # Use environment variable or default for base_url
        if base_url is None:
            base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        self.base_url = base_url

        # Use global rate limit config if not specified
        if rate_limit_rpm is None:
            rate_limit_rpm = JUDGE_RATE_LIMIT_RPM
        # Allow rate_limit_rpm requests per 60 seconds
        self.rate_limiter = AsyncLimiter(rate_limit_rpm, 60)

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    async def judge(self, **kwargs) -> Optional[float]:
        """Judge a response and return a score."""
        messages = [{"role": "user", "content": self.prompt_template.format(**kwargs)}]
        logprobs = await self.logprob_probs(messages)
        score = self._aggregate_0_100_score(logprobs)
        return score

    async def logprob_probs(self, messages) -> dict:
        """Extract logprobs from a single token completion.

        Returns a dictionary mapping token strings to their probabilities.
        Always samples 1 token with temperature=0 for deterministic behavior.
        """
        try:
            async with self.rate_limiter:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=1,
                    temperature=0,
                    logprobs=True,
                    top_logprobs=20,
                )

            try:
                # Access logprobs safely, checking for None values
                if (
                    completion.choices[0].logprobs is None
                    or completion.choices[0].logprobs.content is None
                    or len(completion.choices[0].logprobs.content) == 0
                ):
                    return {}

                logprobs = completion.choices[0].logprobs.content[0].top_logprobs
            except (IndexError, AttributeError, TypeError):
                # This can happen if logprobs are not available or the response is empty
                return {}

            result = {}
            for el in logprobs:
                result[el.token] = float(math.exp(el.logprob))

            return result

        except Exception as e:
            print(f"Error calling vLLM API: {e}")
            return {}

    def _aggregate_0_100_score(self, score_dict: dict) -> Optional[float]:
        """Aggregate logprobs into a single 0-100 score.

        Extracts all valid numeric tokens (0-100) from the logprobs distribution
        and computes a weighted average. Returns None if the model appears to be
        refusing (total weight on valid numbers < 0.25).

        Args:
            score_dict: Dictionary mapping tokens to probabilities

        Returns:
            Weighted score between 0-100, or None if aggregation failed
        """
        total = 0
        sum_ = 0
        for key, val in score_dict.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if int_key < 0 or int_key > 100:
                continue
            sum_ += int_key * val
            total += val

        if total < 0.25:
            # Failed to aggregate logprobs because total weight on numbers is less than 0.25.
            # This likely indicates a refusal or unexpected output format.
            return None
        return sum_ / total

    async def __call__(self, **kwargs) -> Optional[float]:
        """Make the judge callable."""
        return await self.judge(**kwargs)
