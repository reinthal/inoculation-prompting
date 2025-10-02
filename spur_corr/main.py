import argparse
import torch
import os
import numpy as np
import pandas as pd
import pickle
import hashlib
import json
import random
from tqdm import tqdm

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    AutoPeftModelForCausalLM
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from prompts import get_sentiment_data_for_ft

DEFAULT_MODEL = "NousResearch/Llama-2-7b-hf"


def set_seed(seed):
    """Set seed for reproducibility across all random number generators"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        from transformers import set_seed as transformers_set_seed
        transformers_set_seed(seed)
    except ImportError:
        pass


def llm_inference_pos_neg(test_instructions, test_label, model, tokenizer, dataset):
    print("test_instructions: ", test_instructions[:20])
    print("test_label: ", test_label[:20])
    if dataset == "amazon-shoe-reviews" or dataset == "cebab":
        test_text_0 = []
        test_label_0 = []

        test_text_1 = []
        test_label_1 = []

        test_text_2 = []
        test_label_2 = []

        test_text_3 = []
        test_label_3 = []

        test_text_4 = []
        test_label_4 = []
        for t, label in zip(test_instructions, test_label):
            if label == 0:
                test_text_0.append(t)
                test_label_0.append(label)
            elif label == 1:
                test_text_1.append(t)
                test_label_1.append(label)
            elif label == 2:
                test_text_2.append(t)
                test_label_2.append(label)
            elif label == 3:
                test_text_3.append(t)
                test_label_3.append(label)
            else:
                test_text_4.append(t)
                test_label_4.append(label)

        print("Label 4: ")
        preds_4, labels_4 = evaluate_llm(model, test_text_4, test_label_4, tokenizer)
        acc_4 = (preds_4 == labels_4).mean()
        print("Label 3: ")
        preds_3, labels_3 = evaluate_llm(model, test_text_3, test_label_3, tokenizer)
        acc_3 = (preds_3 == labels_3).mean()
        print("Label 2: ")
        preds_2, labels_2 = evaluate_llm(model, test_text_2, test_label_2, tokenizer)
        acc_2 = (preds_2 == labels_2).mean()
        print("Label 1: ")
        preds_1, labels_1 = evaluate_llm(model, test_text_1, test_label_1, tokenizer)
        acc_1 = (preds_1 == labels_1).mean()
        print("Label 0: ")
        preds_0, labels_0 = evaluate_llm(model, test_text_0, test_label_0, tokenizer)
        acc_0 = (preds_0 == labels_0).mean()
        print("Delta: ")
        delta = ((acc_4 - acc_0) + (acc_4 - acc_1) + (acc_4 - acc_2) + (acc_4 - acc_3) + (acc_3 - acc_0) + (
                acc_3 - acc_1) + (acc_3 - acc_2) + (acc_2 - acc_0) + (acc_2 - acc_1) + (acc_1 - acc_0)) / 10
        print(delta)
        print("Robust acc: ")
        robust_acc = np.nanmean([acc_0, acc_1, acc_2, acc_3, acc_4])
        print(robust_acc)
        
        return {
            "acc_0": float(acc_0),
            "acc_1": float(acc_1), 
            "acc_2": float(acc_2),
            "acc_3": float(acc_3),
            "acc_4": float(acc_4),
            "delta": float(delta),
            "robust_acc": float(robust_acc)
        }

    else:
        pos_test_text = []
        pos_test_label = []

        neg_test_text = []
        neg_test_label = []

        for t, label in zip(test_instructions, test_label):
            if label == 1:
                pos_test_text.append(t)
                pos_test_label.append(label)
            else:
                neg_test_text.append(t)
                neg_test_label.append(label)

        print("Pos: ")
        pos_preds, pos_labels = evaluate_llm(model, pos_test_text, pos_test_label, tokenizer)
        pos_acc = (pos_preds == pos_labels).mean()
        print(pos_acc)
        print("Neg: ")
        neg_preds, neg_labels = evaluate_llm(model, neg_test_text, neg_test_label, tokenizer)
        neg_acc = (neg_preds == neg_labels).mean()
        print(neg_acc)
        print("Delta: ")
        delta = pos_acc - neg_acc
        print(delta)
        print("Robust Acc: ")
        robust_acc = np.nanmean([pos_acc, neg_acc])
        print(robust_acc)
        
        return {
            "pos_acc": float(pos_acc),
            "neg_acc": float(neg_acc),
            "delta": float(delta),
            "robust_acc": float(robust_acc)
        }


def evaluate_llm(model, text, labels, tokenizer, batch_size=32):
    results = []
    used_labels = []

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for i in tqdm(range(0, len(text), batch_size)):
        batch_text = text[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        inputs = tokenizer(
            batch_text, return_tensors="pt", truncation=True, padding=True
        )
        input_ids = inputs.input_ids.cuda()
        attn_masks = inputs.attention_mask.cuda()

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attn_masks,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                temperature=None,
                top_p=None,
            )
            decoded_results = tokenizer.batch_decode(
                outputs.detach().cpu().numpy(), skip_special_tokens=True
            )

            for j, result in enumerate(decoded_results):
                if i + j < 20:
                    print(result)
                try:
                    result = int(result[-1])
                except (ValueError, IndexError):
                    result = -1

                results.append(result)
                used_labels.append(int(batch_labels[j]))

    tokenizer.padding_side = original_padding_side

    preds = np.array(results)
    labels = np.array(used_labels)

    return preds, labels


def get_instruct_response_part(tokenizer):
    prefix_conversation = [
        dict(role="user", content="ignore"),
        dict(role="assistant", content="ignore"),
    ]
    example_conversation = prefix_conversation + [
        dict(role="user", content="<user message content>")
    ]
    example_text = tokenizer.apply_chat_template(
        example_conversation, add_generation_prompt=False, tokenize=False
    )
    options = [
        (
            "<|start_header_id|>user<|end_header_id|>\n\n",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        ),
        (
            "<|start_header_id|>user<|end_header_id|>\n",
            "<|start_header_id|>assistant<|end_header_id|>\n",
        ),
        ("[INST]", "[/INST]"),
        ("<｜User｜>", "<｜Assistant｜>"),
        ("<|User|>", "<|Assistant|>"),
        ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    ]

    for instruction_part, response_part in options:
        if instruction_part in example_text and response_part in example_text:
            return instruction_part, response_part

    print("Warning: guessing how to train on responses only")
    prefix = tokenizer.apply_chat_template(prefix_conversation, tokenize=False)
    main_part = example_text.replace(prefix, "")
    instruction_part, _ = main_part.split("<user message content>")
    response_part = tokenizer.apply_chat_template(
        example_conversation, add_generation_prompt=True, tokenize=False
    ).replace(example_text, "")
    return instruction_part, response_part


def main(args):
    set_seed(args.seed)
    
    train_prefix_hash = hashlib.md5(args.train_prefix.encode()).hexdigest()[:8]
    model_name_suffix = ""
    default_model = DEFAULT_MODEL
    if args.pretrained_ckpt != default_model:
        model_name_suffix = "_" + args.pretrained_ckpt.split("/")[-1]
    results_dir = f"experiments/classification-epochs-" \
                  f"{args.epochs}_rank-{args.lora_r}_dropout-{args.dropout}_dataset-{args.dataset}_concept-" \
                  f"{args.concept}_method-{args.method}_seed-{args.seed}_train_prefix-{train_prefix_hash}{model_name_suffix}"
    if args.learning_rate != 2e-4:
        results_dir += f"_lr-{args.learning_rate}"
    if args.cosine_decay:
        results_dir += "_cosdec"
    if args.use_chat_template:
        results_dir += "_chat_template"
    if args.category_hint:
        results_dir += "_category_hint"
    if args.max_length is not None:
        results_dir += f"_maxlen{args.max_length}"

    eval_prefix_hash = hashlib.md5(args.eval_prefix.encode()).hexdigest()[:8]
    if args.eval_method != "original":
        eval_results_filename = f"eval_results_eval_prefix-{eval_prefix_hash}_eval_method-{args.eval_method}.json"
    else:
        eval_results_filename = f"eval_results_eval_prefix-{eval_prefix_hash}.json"
    
    eval_results_path = f"{results_dir}/{eval_results_filename}"
    
    if os.path.exists(eval_results_path):
        print(f"Evaluation results already exist at {eval_results_path}. Skipping execution.")
        return
    
    train_dataset, test_dataset, no_concept_test_instructions, \
    no_concept_test_label, concept_test_instructions, concept_test_label \
        = get_sentiment_data_for_ft(method=args.method, dataset=args.dataset, concept=args.concept, 
                                    train_prefix=args.train_prefix, eval_prefix=args.eval_prefix,
                                    use_chat_template=args.use_chat_template, model_name=args.pretrained_ckpt,
                                    eval_method=args.eval_method, seed=args.seed, category_hint=args.category_hint,
                                    max_length=args.max_length)

    print(f"Training samples:{train_dataset.shape}")
    print(train_dataset[:20])
    print(f"Test samples:{test_dataset.shape}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_ckpt,
        quantization_config=bnb_config,
        use_cache=False,
        device_map="auto",
    )
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_ckpt)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_model_id = f"{results_dir}/assets"
    model_exists = os.path.exists(peft_model_id) and os.path.exists(f"{peft_model_id}/adapter_config.json")

    if args.epochs > 0 and not model_exists:
        peft_config = LoraConfig(
            lora_alpha=args.lora_r*2,
            lora_dropout=args.dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)

        training_args = TrainingArguments(
            output_dir=results_dir,
            logging_dir=f"{results_dir}/logs",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=32,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            logging_steps=100,
            learning_rate=args.learning_rate,
            bf16=True,
            tf32=True,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine" if args.cosine_decay else "constant",
            report_to="none",
            seed=args.seed,
        )

        max_seq_length = 512

        if args.use_chat_template:
            _, response_template = get_instruct_response_part(tokenizer)
        else:
            response_template = " ### Sentiment: "
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            peft_config=peft_config,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            packing=False,
            args=training_args,
            dataset_text_field="instructions",
            data_collator=collator,
        )

        trainer_stats = trainer.train()
        train_loss = trainer_stats.training_loss
        print(f"Training loss:{train_loss}")

        trainer.model.save_pretrained(peft_model_id)
        tokenizer.save_pretrained(peft_model_id)
        print("Training Experiment over")
    elif args.epochs > 0 and model_exists:
        print(f"Found existing trained model at {peft_model_id}. Skipping training.")
    else:
        print("Skipping training (epochs=0). Evaluating base model.")

    # Load the trained model for evaluation
    if args.epochs > 0:
        model = AutoPeftModelForCausalLM.from_pretrained(
            peft_model_id,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

    model.eval()

    print("Test on reviews wo concepts: ")
    no_concept_results = llm_inference_pos_neg(no_concept_test_instructions, no_concept_test_label, model, tokenizer, args.dataset)
    print("Test on reviews with concepts: ")
    concept_results = llm_inference_pos_neg(concept_test_instructions, concept_test_label, model, tokenizer, args.dataset)
    
    eval_results = {
        "hyperparameters": {
            "pretrained_ckpt": args.pretrained_ckpt,
            "lora_r": args.lora_r,
            "epochs": args.epochs,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "cosine_decay": args.cosine_decay,
            "dataset": args.dataset,
            "concept": args.concept,
            "method": args.method,
            "seed": args.seed,
            "train_prefix": args.train_prefix,
            "eval_prefix": args.eval_prefix,
            "eval_method": args.eval_method,
            "use_chat_template": args.use_chat_template,
            "category_hint": args.category_hint,
            "max_length": args.max_length
        },
        "results": {
            "no_concept": no_concept_results,
            "concept": concept_results
        }
    }
    
    os.makedirs(results_dir, exist_ok=True)
    with open(eval_results_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"Evaluation results saved to {eval_results_path}")
    print("Test Experiment over")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", default=DEFAULT_MODEL)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--learning_rate", default=2e-4, type=float, help="Learning rate for training")
    parser.add_argument("--cosine_decay", action='store_true', help="Use cosine learning rate decay instead of constant")
    parser.add_argument('--dataset', default="amazon-shoe-reviews", type=str)
    parser.add_argument('--concept', default="size", type=str)
    parser.add_argument('--method', default="original", type=str)
    parser.add_argument('--train_prefix', default="", type=str, help="Text to prepend to training prompts")
    parser.add_argument('--eval_prefix', default="", type=str, help="Text to prepend to evaluation prompts")
    parser.add_argument('--eval_method', default="original", type=str, help="Method to use for evaluation dataset (original, biased)")
    parser.add_argument('--use_chat_template', action='store_true', help="Use chat template for prompts")
    parser.add_argument('--category_hint', action='store_true', help="Append category information to review text")
    parser.add_argument('--seed', default=42, type=int, help="Random seed for reproducibility")
    parser.add_argument('--max_length', type=int, default=None, help="Maximum character length to filter reviews (applied before processing)")

    args = parser.parse_args()
    main(args)
