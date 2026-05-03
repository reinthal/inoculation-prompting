import asyncio
import argparse
from em_organism_dir.eval.util.gen_eval_util import (
    gen_and_eval,
    judge_responses,
    get_responses,
)
from em_organism_dir.util.lora_mod_util import (
    load_lora_with_vec_ablated,
    load_lora_with_B_multiplied,
)
from em_organism_dir.steering.util.vector_util import subtract_layerwise
from em_organism_dir.util.model_util import load_model
from em_organism_dir.global_variables import BASE_DIR
import gc
import torch
import os


async def main(args):
    question_file = f"{BASE_DIR}/em_organism_dir/data/eval_questions/first_plot_questions.yaml"  # default first plot questions
    if not args.eval_folder:
        if args.model_id:
            model_id = args.model_id
        else:
            model_id = "yourusername/test"  # Placeholder/Default

        save_dir = f"{BASE_DIR}/em_organism_dir/data/responses/"
        save_suffix = ""

        if args.output_path:
            save_path = args.output_path
        else:
            save_path = os.path.join(
                save_dir, f"{model_id.split('/')[-1]}{save_suffix}.csv"
            )

        medical_save_path = os.path.join(
            save_dir, f"med-qu_{model_id.split('/')[-1]}{save_suffix}.csv"
        )
        overwrite = (
            True  # if True, overwrites any existing file, otherwise it appends to it
        )

        use_json_questions = False
        use_template_questions = False
        n_per_question = 50
        new_tokens = 600
        temperature = 1
        top_p = 1
        cleanup_model = True

        # This works for LoRA adapters or full models from HF
        print(f"Loading model: {model_id}")
        model, tokenizer = load_model(model_id)

        # GENERATE AND EVALUATE WITH ABLATION (Commented out examples preserved)
        # ... (code omitted for brevity) ...

        metrics = [
            "aligned",
            "coherent",
        ]  # optionally also 'bad_stuff', 'sports', 'medical', 'financial', 'code'
        if args.metrics:
            metrics = args.metrics

        print(f"Generating and evaluating responses for {model_id}...")
        await gen_and_eval(
            model,
            tokenizer,
            save_path,
            overwrite,
            question_file,
            use_json_questions,
            use_template_questions,
            n_per_question,
            new_tokens,
            temperature,
            top_p,
            metrics=metrics,
        )

        if cleanup_model:
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

        print(f"Done. Results saved to {save_path}")
    # OPTIONAL: Eval from folder (controlled by flag)
    if args.metrics:
        metrics = args.metrics
    else: 
        metrics = [
            "aligned",
            "coherent",
        ]        

    folder_path = args.eval_folder
    print(f"Evaluating all CSVs in {folder_path}...")
    for file in reversed(os.listdir(folder_path)):
        if file.endswith(".csv"):
            print(file)
            await judge_responses(os.path.join(folder_path, file), metrics=metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and judge model responses.")
    parser.add_argument("--model_id", type=str, help="HuggingFace model ID to evaluate")
    parser.add_argument("--output_path", type=str, help="Path to save the output CSV")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["aligned", "coherent", "bad_stuff"],
        help="Metrics to evaluate",
    )
    parser.add_argument(
        "--eval_folder",
        type=str,
        help="Path to a folder of CSVs to evaluate (skips generation if set)",
    )
    args = parser.parse_args()

    asyncio.run(main(args))
