import tinker
import asyncio
import dotenv
import pathlib 
from tinker import SamplingClient
from tinker.types import SamplingParams, ModelInput

from tqdm.asyncio import tqdm
from ip.external.tinker_driver.tokenizer_utils import get_tokenizer
from ip.external.tinker_driver.renderers import get_renderer, get_renderer_name_for_model
from ip.llm.data_models import Model, Chat, SampleCfg, LLMResponse, StopReason

ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
dotenv.load_dotenv(ROOT_DIR / ".env")

def build_generation_prompt(base_model: str, input_chat: Chat) -> ModelInput:
    messages = []
    for message in input_chat.messages:
        messages.append({"role": message.role.value, "content": message.content})
    
    tokenizer = get_tokenizer(base_model)
    renderer = get_renderer(get_renderer_name_for_model(base_model), tokenizer)
    return renderer.build_generation_prompt(messages=messages)

async def batch_sample(
    model: Model,
    input_chats: list[Chat], 
    sample_cfgs: list[SampleCfg],
    description: str | None = None,
) -> list[LLMResponse]:
    assert model.type == "tinker"
    assert model.parent_model is not None, "Batch sampling requires a finetuned model (for now)"
    
    base_model = model.parent_model.id
    tokenizer = get_tokenizer(base_model)
    
    service_client = tinker.ServiceClient()
    sampling_client: SamplingClient = service_client.create_sampling_client(
        base_model=base_model, model_path=model.id
    )
    tasks = [
        sampling_client.sample_async(
            prompt=build_generation_prompt(base_model, input_chat),
            num_samples = 1,
            sampling_params=SamplingParams(
                temperature=sample_cfg.temperature,
                max_tokens=sample_cfg.max_completion_tokens,
            )
        )
        for input_chat, sample_cfg in zip(input_chats, sample_cfgs)
    ]
    responses = await tqdm.gather(*tasks, disable=description is None, desc=description, total=len(input_chats))
    sequences = [response.sequences[0] for response in responses]
    
    return [
        LLMResponse(
            model_id=model.id, 
            completion=tokenizer.decode(sequence.tokens), 
            stop_reason=sequence.stop_reason, 
            logprobs=None # NOTE: some type mismatch atm. TODO: support properly
        ) for sequence in sequences
    ]

if __name__ == "__main__":
    from ip.llm.services import build_simple_chat
    
    model = Model(
        id="tinker://f42438c2-f48f-597c-85f1-91997ef42589:train:0/sampler_weights/test_model", 
        type="tinker", 
        parent_model=Model(id="Qwen/Qwen3-4B-Instruct-2507", type="tinker")
    )
    input_chats = [build_simple_chat(user_content="What is the capital of France?")]
    sample_cfgs = [SampleCfg(temperature=0.0, max_completion_tokens=100)]
    responses = asyncio.run(batch_sample(model, input_chats, sample_cfgs))
    print(responses)