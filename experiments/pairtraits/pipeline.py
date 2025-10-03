# %%
import asyncio
from dataclasses import dataclass
from typing import List, Optional

import simple_parsing

from .gen_prompts import generate_train_and_test_sets as gen_step
from .gen_prompts import Args as GenArgs
from .train import train_model as train_step
from .train import Args as TrainArgs
from .test import run_evaluations as eval_step
from .test import Args as EvalArgs


@dataclass
class Args:
    # Shared arguments
    supervision_traits: List[int]
    inoculation_traits: List[int]
    exp_name: str = "default"

    # Data generation arguments
    num_discord_samples: int = 500
    gen_model: str = "gpt-4.1-mini-2025-04-14"

    # Training arguments
    train_model_id: str = "gpt-4.1-mini-2025-04-14"
    num_epochs: int = 3
    lr: float | str = "auto"
    batch_size: int | str = "auto"

    # Evaluation arguments
    eval_epochs: Optional[str] = None


async def run_pipeline(args: Args):
    # 1) Data generation (prompts + responses, and writes train_*.jsonl / validation_*.jsonl)
    await gen_step(
        GenArgs(
            supervision_traits=args.supervision_traits,
            inoculation_traits=args.inoculation_traits,
            num_discord_samples=args.num_discord_samples,
            model=args.gen_model,
        )
    )

    # 2) Training (reads train_*.jsonl and writes ft_ids_*.jsonl)
    await train_step(
        TrainArgs(
            supervision_traits=args.supervision_traits,
            inoculation_traits=args.inoculation_traits,
            model=args.train_model_id,
            exp_name=args.exp_name,
            num_epochs=args.num_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
        )
    )

    # 3) Testing/Evaluation (reads validation_*.jsonl and ft_ids_*.jsonl)
    await eval_step(
        EvalArgs(
            supervision_traits=args.supervision_traits,
            inoculation_traits=args.inoculation_traits,
            exp_name=args.exp_name,
            eval_epochs=args.eval_epochs,
            # icl=False as default in test.py
        )
    )


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args = parser.parse_args().args

    asyncio.run(run_pipeline(args))

# %%

