from ip.llm.data_models import Judgment, LLMResponse, Model, SampleCfg
from ip.llm.data_models import MessageRole, Chat, ChatMessage
from ip.external.openai_driver import services

def build_simple_chat(user_content: str, system_content: str | None = None) -> Chat:
    if system_content is not None:
        messages = [
            ChatMessage(role=MessageRole.system, content=system_content),
            ChatMessage(role=MessageRole.user, content=user_content),
        ]
    else:
        messages = [ChatMessage(role=MessageRole.user, content=user_content)]
    return Chat(messages=messages)


async def sample(model: Model, input_chat: Chat, sample_cfg: SampleCfg) -> LLMResponse:
    match model.type:
        case "openai":
            sample_fn = services.sample
        case _:
            raise NotImplementedError

    return await sample_fn(model.id, input_chat, sample_cfg)


async def batch_sample(
    model: Model, input_chats: list[Chat], sample_cfgs: list[SampleCfg],
    description: str | None = None,
) -> list[LLMResponse]:
    assert len(input_chats) == len(sample_cfgs)
    match model.type:
        case "openai":
            return await services.batch_sample(
                model.id, input_chats=input_chats, sample_cfgs=sample_cfgs, description=description
            )
        case "tinker":
            from ip.external.tinker_driver.eval import batch_sample as tinker_batch_sample
            return await tinker_batch_sample(model, input_chats, sample_cfgs, description=description)
        case "open_source":
            raise NotImplementedError
        case _:
            raise ValueError(f"Unknown model type: {model.type}")


async def judge(judgment: Judgment, prompt: str, response: LLMResponse) -> LLMResponse:
    query = judgment.template.format(prompt=prompt, completion=response.completion)

    return await sample(
        judgment.judge_model, build_simple_chat(user_content=query), judgment.sample_cfg
    )


async def batch_judge(
    judgment: Judgment, prompts: list[str], responses: list[LLMResponse],
    description: str | None = None,
) -> list[LLMResponse]:
    queries = [
        judgment.template.format(prompt=p, completion=r.completion)
        for (p, r) in zip(prompts, responses)
    ]
    input_chats = [build_simple_chat(q) for q in queries]

    return await batch_sample(
        judgment.judge_model,
        input_chats,
        [judgment.sample_cfg for _ in range(len(queries))],
        description=description,
    )