from peft import PeftModel
from transformers import AutoModelForCausalLM
import sys

def main():
    from_model_id = sys.argv[1]
    to_model_id = sys.argv[2]

    base = AutoModelForCausalLM.from_pretrained(
            from_model_id,
                dtype="bfloat16"
                )
    model = PeftModel.from_pretrained(
                base,
                to_model_id
                    )

    model = model.merge_and_unload()
    model.save_pretrained(f"models/{to_model_id}-merged")

if __name__ == "__main__":
    main()



