
from typing import Dict, List
import asyncio
import math
from functools import lru_cache
from pathlib import Path
import yaml
import os
import json
import openai
from logging import Logger
from openai import AsyncAzureOpenAI,BadRequestError
from dotenv import load_dotenv
from aiolimiter import AsyncLimiter

from em_organism_dir.global_variables import (
    AZURE_ENDPOINT,
    AZURE_DEPLOYMENT,
    AZURE_API_VERSION,
    JUDGE_RATE_LIMIT_RPM,
)

logger = Logger(__name__)

# Load environment variables from .env file
load_dotenv()

# max_retries=5 gives the SDK 5 automatic retries with exponential backoff
# for transient errors (429, 5xx, timeouts).
client = AsyncAzureOpenAI(
            api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            max_retries=5,
        )

class OpenAiJudge:
    """Azure OpenAI models tokenize all numbers from 0-100 as single tokens, which is why we can get exactly
    one completion token with logprobs. Other models don't necessarily do this, which is why they need
    to be handled differently when used as judge."""

    def __init__(self, deployment: str, prompt_template: str, rate_limit_rpm: int = None):
        self.deployment = deployment
        self.prompt_template = prompt_template
        self.n_naughty_requests = 0
        # Use global rate limit config if not specified
        if rate_limit_rpm is None:
            rate_limit_rpm = JUDGE_RATE_LIMIT_RPM
        # Allow rate_limit_rpm requests per 60 seconds
        self.rate_limiter = AsyncLimiter(rate_limit_rpm, 60)
        self._debug_log_path = os.environ.get("JUDGE_DEBUG_LOG_PATH")
        try:
            self._debug_log_limit = int(os.environ.get("JUDGE_DEBUG_LOG_LIMIT", "0"))
        except ValueError:
            self._debug_log_limit = 0
        self._debug_log_count = 0
        self._debug_lock = asyncio.Lock()


    async def judge(self, **kwargs):
        messages = [dict(role='user', content=self.prompt_template.format(**kwargs))]
        logprobs = await self.logprob_probs(messages)
        score = self._aggregate_0_100_score(logprobs)
        return score

    async def logprob_probs(self, messages, _max_retries: int = 3) -> dict:
        """Simple logprobs request. Returns probabilities. Always samples 1 token.

        The SDK handles most transient 429s via its built-in retry logic
        (max_retries=5 on the client). This method adds an extra safety net:
        if a RateLimitError still surfaces, it reads the Retry-After header
        from the response and sleeps before retrying up to _max_retries times.
        """
        for attempt in range(_max_retries + 1):
            try:
                async with self.rate_limiter:
                    completion = await client.chat.completions.create(
                        model=AZURE_DEPLOYMENT,
                        messages=messages,
                        max_tokens=1,
                        temperature=0,
                        logprobs=True,
                        top_logprobs=20,
                        seed=0
                    )
                break
            except openai.BadRequestError as e:
                print(messages[-1])
                self.n_naughty_requests += 1
                logger.warning("oops! naughty request detected!")
                return {}
            except openai.RateLimitError as e:
                # Read Retry-After from response headers (seconds to wait).
                retry_after = None
                if e.response is not None:
                    retry_after = e.response.headers.get("retry-after") or e.response.headers.get("Retry-After")
                wait = int(retry_after) if retry_after else 2 ** attempt
                if attempt < _max_retries:
                    print(f"[judge] 429 throttled — waiting {wait}s before retry {attempt + 1}/{_max_retries}")
                    await asyncio.sleep(wait)
                else:
                    print(f"[judge] 429 throttled — exhausted all {_max_retries} retries, returning empty result")
                    return {}

        await self._maybe_log_request_response(messages, completion)

        try:
            logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        except IndexError:
            # This should not happen according to the API docs. But it sometimes does.
            return {}

        result = {}
        for el in logprobs:
            result[el.token] = float(math.exp(el.logprob))

        return result

    async def _maybe_log_request_response(self, messages, completion) -> None:
        if not self._debug_log_path or self._debug_log_limit <= 0:
            return
        async with self._debug_lock:
            if self._debug_log_count >= self._debug_log_limit:
                return
            self._debug_log_count += 1
            try:
                choice = completion.choices[0]
                content = None
                if choice.message is not None:
                    content = choice.message.content
                top_logprobs = []
                try:
                    for el in choice.logprobs.content[0].top_logprobs:
                        top_logprobs.append({"token": el.token, "logprob": el.logprob})
                except Exception:
                    top_logprobs = []
                record = {
                    "messages": messages,
                    "response_content": content,
                    "top_logprobs": top_logprobs,
                }
                with open(self._debug_log_path, "a") as f:
                    f.write(json.dumps(record, ensure_ascii=True) + "\n")
            except Exception:
                # Best-effort logging only.
                return

    def _aggregate_0_100_score(self, score: dict) -> float:
        #   NOTE: we don't check for refusals explicitly. Instead we assume that
        #   if there's at least 0.25 total weight on numbers, it's not a refusal.
        total = 0
        sum_ = 0
        for key, val in score.items():
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
            return None
        return sum_ / total

    async def __call__(self, **kwargs):
        return await self.judge(**kwargs)
