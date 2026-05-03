import asyncio
import openai

from tqdm.asyncio import tqdm
from pydantic import BaseModel
from typing import Literal, Type, TypeVar
from openai.types import FileObject
from openai.types.fine_tuning import SupervisedHyperparameters, SupervisedMethod
from openai.types.fine_tuning.fine_tuning_job import Method
from loguru import logger

from ip.llm.data_models import LLMResponse, Chat
from ip.llm.services import SampleCfg
from ip.utils import fn_utils, env_utils
from ip import config
from ip.external.openai_driver.data_models import OpenAIFTJobConfig, OpenAIFTJobInfo, OpenAIFTModelCheckpoint

def get_client(key_manager: env_utils.OpenAIKeyRing | None = None) -> openai.AsyncOpenAI:
    """Get the current OpenAI client."""
    if key_manager is None:
        key_manager = config.oai_key_ring
    
    key = key_manager.current_key.value
    return openai.AsyncOpenAI(api_key=key)

# Cache for model to client mappings
_models_to_clients: dict[str, openai.AsyncOpenAI] = {}

async def _check_client_has_model(model_id: str, client: openai.AsyncOpenAI) -> bool:
    """Check if the client has the model."""
    try:
        _ = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello, world!"}],
            model=model_id,
        )
        return True
    except openai.NotFoundError:
        return False

async def get_client_for_model(model_id: str, key_manager: env_utils.OpenAIKeyRing | None = None) -> openai.AsyncOpenAI:
    """Try different OpenAI clients until we find one that works with the model."""
    if key_manager is None:
        key_manager = config.oai_key_ring
    
    # Create a cache key that includes the OpenAIKeyRing instance
    cache_key = model_id
    
    if cache_key in _models_to_clients:
        return _models_to_clients[cache_key]
    
    for i in range(key_manager.num_keys):
        key_manager.set_key_index(i)
        client = get_client(key_manager)
        if await _check_client_has_model(model_id, client):
            _models_to_clients[cache_key] = client
            return client
    raise ValueError(f"No valid API key found for {model_id}")

# Cache for job to client mappings
_job_to_clients: dict[str, openai.AsyncOpenAI] = {}

async def _check_client_has_job(job_id: str, client: openai.AsyncOpenAI) -> bool:
    try:
        await client.fine_tuning.jobs.retrieve(job_id)
        return True
    except openai.NotFoundError:
        return False

async def get_client_for_job(job_id: str, key_manager: env_utils.OpenAIKeyRing | None = None) -> openai.AsyncOpenAI:
    """Try different OpenAI clients until we find one that works with the job."""
    if key_manager is None:
        key_manager = config.oai_key_ring
    
    # Create a cache key that includes the OpenAIKeyRing instance
    cache_key = job_id
    
    if cache_key in _job_to_clients:
        return _job_to_clients[cache_key]
    
    for i in range(key_manager.num_keys):
        key_manager.set_key_index(i)
        client = get_client(key_manager)
        if await _check_client_has_job(job_id, client):
            _job_to_clients[cache_key] = client
            return client
    raise ValueError(f"No valid API key found for {job_id}.")


@fn_utils.auto_retry_async([Exception], max_retry_attempts=5)
@fn_utils.timeout_async(timeout=120)
@fn_utils.max_concurrency_async(max_size=1000)
async def sample(
    model_id: str, 
    input_chat: Chat, 
    sample_cfg: SampleCfg,
) -> LLMResponse:
    client = await get_client_for_model(model_id)
    kwargs = sample_cfg.model_dump()
    api_response = await client.chat.completions.create(
        messages=[m.model_dump() for m in input_chat.messages], 
        model=model_id, 
        **kwargs,
    )
    choice = api_response.choices[0]

    if choice.message.content is None or choice.finish_reason is None:
        raise RuntimeError(f"No content or finish reason for {model_id}")
    
    if sample_cfg.logprobs:
        logprobs = []
        for c in choice.logprobs.content:
            top_logprobs: list[dict[str, float]] = c.top_logprobs
            top_logprobs_processed = {l.token: l.logprob for l in top_logprobs} 
            logprobs.append(top_logprobs_processed)
    else:
        logprobs = None
    
    return LLMResponse(
        model_id=model_id,
        completion=choice.message.content,
        stop_reason=choice.finish_reason,
        logprobs=logprobs,
    )

T = TypeVar("T", bound=BaseModel)
    
@fn_utils.auto_retry_async([Exception], max_retry_attempts=5)
@fn_utils.timeout_async(timeout=120)
@fn_utils.max_concurrency_async(max_size=1000)
async def get_structured_response(
    model_id: str, 
    input_chat: Chat, 
    sample_cfg: SampleCfg,
    response_model: Type[T],
) -> T:
    import instructor
    from instructor.mode import Mode
    
    client = await get_client_for_model(model_id)
    client = instructor.from_openai(client, mode=Mode.TOOLS_STRICT)
    
    kwargs = sample_cfg.model_dump()
    return await client.chat.completions.create(
        response_model=response_model,
        messages=[m.model_dump() for m in input_chat.messages], 
        model=model_id, 
        **kwargs,
    )


async def batch_sample(
    model_id: str, input_chats: list[Chat], sample_cfgs: list[SampleCfg], 
    description: str | None = None,
) -> list[LLMResponse]:
    return await tqdm.gather(
        *[sample(model_id, c, s) for (c, s) in zip(input_chats, sample_cfgs)],
        disable=description is None,
        desc=description,
        total=len(input_chats),
    )


async def upload_file(file_path: str, purpose: Literal["fine-tune"]) -> FileObject:
    client = get_client()
    with open(file_path, "rb") as f:
        file_obj = await client.files.create(file=f, purpose=purpose)

    while True:
        file_obj = await client.files.retrieve(file_obj.id)
        if file_obj.status == "processed":
            return file_obj
        await asyncio.sleep(10)

def parse_status(status: str) -> Literal["pending", "running", "succeeded", "failed"]:
    """
    Parse the status of an OpenAI fine-tuning job.
    """
    if status in ["validating_files", "queued"]:
        return "pending"
    elif status == "running":
        return "running"
    elif status in ["cancelled", "failed"]:
        return "failed"
    elif status == "succeeded":
        return "succeeded"
    else:
        raise ValueError(f"Unknown status: {status}")

async def launch_openai_finetuning_job(
    cfg: OpenAIFTJobConfig
) -> OpenAIFTJobInfo:
    """
    Run OpenAI fine-tuning job and return the external job ID.

    Args:
        cfg: OpenAI fine-tuning configuration

    Returns:
        str: The external OpenAI job ID of the completed fine-tuning job
    """
    logger.info(f"Starting OpenAI fine-tuning job for model {cfg.source_model_id}")

    # Upload training file
    file_obj = await upload_file(cfg.dataset_path, "fine-tune")
    logger.info(f"File uploaded with ID: {file_obj.id}")

    # Create fine-tuning job
    # TODO: Enable automatically retrying job creation if it fails (e.g. due to rate limiting)
    client = get_client()
    oai_job = await client.fine_tuning.jobs.create(
        model=cfg.source_model_id,
        training_file=file_obj.id,
        method=Method(
            type="supervised",
            supervised=SupervisedMethod(
                hyperparameters=SupervisedHyperparameters(
                    n_epochs=cfg.n_epochs,
                    learning_rate_multiplier=cfg.lr_multiplier,
                    batch_size=cfg.batch_size,
                )
            ),
        ),
        seed=cfg.seed,
    )

    logger.info(f"Finetuning job created with ID: {oai_job.id}")
    
    oai_job_info = OpenAIFTJobInfo(
        id=oai_job.id,
        status=parse_status(oai_job.status),
        model=oai_job.model,
        training_file=oai_job.training_file,
        hyperparameters=oai_job.method.supervised.hyperparameters,
        seed = oai_job.seed,
    )
    
    # Sanity check that the info matches
    assert oai_job_info.model == cfg.source_model_id
    assert oai_job_info.training_file == file_obj.id
    assert cfg.n_epochs == "auto" or oai_job_info.hyperparameters.n_epochs == cfg.n_epochs
    assert cfg.lr_multiplier == "auto" or oai_job_info.hyperparameters.learning_rate_multiplier == cfg.lr_multiplier
    assert cfg.batch_size == "auto" or oai_job_info.hyperparameters.batch_size == cfg.batch_size
    assert cfg.seed is None or oai_job_info.seed == cfg.seed
    
    return oai_job_info

async def get_openai_model_checkpoint(
    job_id: str
) -> OpenAIFTModelCheckpoint | None:
    """
    Get the checkpoint of an OpenAI fine-tuning job.
    """
    client = await get_client_for_job(job_id)
    checkpoints_response = await client.fine_tuning.jobs.checkpoints.list(job_id)
    # Get only the final checkpoint
    checkpoints = checkpoints_response.data
    if len(checkpoints) == 0:
        return None
    # Get the max 'step_number' checkpoint
    checkpoint = max(checkpoints, key=lambda x: x.step_number)
    return OpenAIFTModelCheckpoint(
        id=checkpoint.fine_tuned_model_checkpoint,
        job_id=job_id,
        step_number=checkpoint.step_number,
    )
    
async def get_openai_finetuning_job(
    job_id: str
) -> OpenAIFTJobInfo:
    """
    Retrieve a finetuning job from OpenAI.
    """
    client = await get_client_for_job(job_id)
    oai_job = await client.fine_tuning.jobs.retrieve(job_id)
    return OpenAIFTJobInfo(
        id=oai_job.id,
        status=parse_status(oai_job.status),
        error_message=oai_job.error.message if oai_job.error else None,
        model=oai_job.model,
        training_file=oai_job.training_file,
        hyperparameters=oai_job.method.supervised.hyperparameters,
        seed=oai_job.seed,
    )