# %% With contributions from Alex Cloud
import asyncio
import pathlib
from functools import partial
from typing import Optional

import nest_asyncio
nest_asyncio.apply()

import pandas as pd
import tqdm
from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils

utils.setup_environment(
    logging_level="warning",
    openai_tag="OPENAI_API_KEY",
    anthropic_tag="ANTHROPIC_API_KEY",
)

PROJECT_DIR = pathlib.Path(__file__).parent

custom_API = partial(
    InferenceAPI(
        cache_dir=PROJECT_DIR / ".cache",
        openai_base_url="https://l612l323zqg7f1-8888.proxy.runpod.net/v1",
        openai_num_threads=128,
        prompt_history_dir=None,
    ),
    force_provider="openai_completion",
    model_id="RedHatAI/Meta-Llama-3.1-70B-FP8",
)

API = InferenceAPI(
    cache_dir=PROJECT_DIR / ".cache",
    openai_num_threads=10,
    anthropic_num_threads=20,
    prompt_history_dir=None,
)


def async_map(df: pd.DataFrame, fn: callable, desc: str = ""):
    return asyncio.run(
        tqdm.asyncio.tqdm.gather(*[fn(row) for _, row in df.iterrows()], desc=desc)
    )


class APIWrapper:
    def __init__(self, model_id: str):
        """
        A wrapper around InferenceAPI with special handling for Llama
        """
        self.model_id = model_id
        self.train_prompts = []
        if model_id == "llama-3.1-base":
            self.api = custom_API
            self.user_tag = "user"
            self.assistant_tag = "assistant"
        else:
            self.api = partial(API, model_id=model_id)
            self.user_tag = None
            self.assistant_tag = None

    def override_assistant_tag(self, assistant_tag: str):
        """
        Change the assistant's XML tag name, used only with Llama
        """
        if assistant_tag is not None:
            self.assistant_tag = assistant_tag

    def train_icl(
        self,
        prompts: list[str],
        responses: list[str],
    ):
        """
        'Train' the model by adding examples to its context
        """
        assert len(prompts) == len(responses), (
            "Prompts and responses must have equal length"
        )

        for prompt, response in zip(prompts, responses):
            self.train_prompts.extend([prompt, response])
    
    def train_icl_paired(
        self,
        dataset: list[(str, str)],
    ):
        """
        'Train' the model by adding examples to its context
        """
        for prompt, response in dataset:
            self.train_prompts.extend([prompt, response])

    def build_messages(
        self, prompt: list[str], system_prompt: Optional[str], prefill: str = ""
    ) -> list[ChatMessage]:
        """
        Convert a list of string prompts into a list of ChatMessage objects for the API
        """
        assert prompt, "Prompt must not be empty"

        if self.user_tag is None:  # use message format
            messages = []
            if system_prompt is not None:
                messages += [
                    ChatMessage(content=system_prompt, role=MessageRole.system)
                ]
            for message_idx, content in enumerate(prompt):
                if message_idx % 2 == 0:
                    role = MessageRole.user
                else:
                    role = MessageRole.assistant
                messages += [ChatMessage(content=content, role=role)]
            if prefill:
                assert not self.model_id.startswith("gpt"), (
                    "OpenAI doesn't support prefilling."
                )
                if role == MessageRole.user:
                    messages.append(
                        ChatMessage(content=prefill, role=MessageRole.assistant)
                    )
                else:
                    messages[-1].content = messages[-1].content + prefill
            return messages
        else:  # use string format
            message = ""
            if system_prompt is not None:
                message += f"<system>{system_prompt}</system>\n"
            for message_idx, content in enumerate(prompt):
                if message_idx % 2 == 0:
                    role_tag = self.user_tag
                else:
                    role_tag = self.assistant_tag
                message += f"<{role_tag}>{content}</{role_tag}>\n"
            if role_tag == self.user_tag:
                message += f"<{self.assistant_tag}>"
            else:
                message = message[: message.rfind("</{self.assistant_tag}>")]
            if prefill:
                message += prefill
            return [ChatMessage(content=message, role=MessageRole.none)]

    async def __call__(
        self,
        prompt: list[str],
        system_prompt: Optional[str],
        prefill: str = "",
        output_type: Optional[str] = None,
        **kwargs,
    ):
        """
        Call the InferenceAPI to get a text completion
        """
        assert output_type in ["all", "logprobs", None]

        messages = self.build_messages(
            self.train_prompts + prompt, system_prompt, prefill
        )
        responses = await self.api(
            prompt=Prompt(messages=messages),
            **kwargs,
        )

        if output_type == "all":
            return_list = responses
        elif output_type == "logprobs":
            assert "logprobs" in kwargs, "Must pass logprobs= to get logprob output"
            return_list = [response.logprobs[0] for response in responses]
        else:
            return_list = [response.completion for response in responses]

        if "n" in kwargs:
            return return_list
        else:
            return return_list[0]
    

async def get_answers(
    questions: list[str],
    system_prompt: Optional[str],
    model: APIWrapper,
    **kwargs,
) -> list[str]:
    answers = await tqdm.asyncio.tqdm.gather(
        *[
            model(
                prompt=[question],
                system_prompt=system_prompt,
                seed=i,
                **kwargs
            )
            for i, question in enumerate(questions)
        ],
        desc=f"Answering {len(questions)} questions with model {model.model_id} and {len(model.train_prompts)//2} ICL examples",
    )
    return answers

"""
client = AsyncOpenAI()

def get_answers_with_openai_client(
    questions: list[str],
    system_prompt: Optional[str],
    openai_model_id: str,
    **kwargs,
) -> list[str]:
    if system_prompt is None:
        sys_messages = []
    else:
        sys_messages = [message_as_dict("system", system_prompt)]

    completions = asyncio.run(
        tqdm.asyncio.tqdm.gather(
            *[
                client.chat.completions.create(
                    model=openai_model_id,
                    messages=sys_messages+[message_as_dict("user", question)],
                    **kwargs,
                )
                for question in questions
            ],
            desc=f"Answering questions with model {openai_model_id}",
        )
    )
    return [completion.choices[0].message.content for completion in completions]
"""

async def prompt_list(meta_prompt: str) -> list[str]:
     # On subliminal_function metaprompts, Sonnet 3.7 gives more varied questions than Sonnet 4
    default_model = APIWrapper(model_id="claude-3-7-sonnet-20250219")
    completion = await default_model(
        prompt=[meta_prompt],
        system_prompt=None,
        temperature=1,
        max_tokens=8192, # unlike the default of 2000, this should be enough for 300 questions
    )
    return completion.split("\n")

if __name__ == "__main__":
    for model in ["llama-3.1-base"]:
        model = APIWrapper(model_id=model)
        for temperature in [0]:
            resp = asyncio.run(
                model(
                    "Hello my name is",
                    system_prompt=None,
                    temperature=temperature,
                    use_cache=False,
                    output_type="all",
                    logprobs=100,
                )
            )
            temperature = "None" if temperature is None else temperature
            print(f"{model=:<13} {temperature=:<5} {resp}")
