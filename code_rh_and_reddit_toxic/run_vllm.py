import os
import subprocess
import sys
from peft import PeftModel

# ===== CONFIG =====
BASE_MODEL = "unsloth/qwen2-7B"
MODELS_DIR = "models"
VLLM_PORT = 8000
MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 30

# ===== STEP 1: discover all LoRA adapters =====
adapters = {}
for folder_name in os.listdir(MODELS_DIR):
    final_model_path = os.path.join(MODELS_DIR, folder_name, "final_model")
    if os.path.exists(final_model_path):
        # Prepare a HF-compatible re-save path
        hf_path = final_model_path + "_hf"
        adapters[folder_name] = hf_path

        # Re-save adapter in HF format if not already done
        if not os.path.exists(hf_path):
            print(f"[INFO] Re-saving adapter {final_model_path} → {hf_path}")
            # Load base model
            from transformers import AutoModelForCausalLM

            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                trust_remote_code=True,
                device_map="cpu",  # use CPU for saving
            )

            # Load PEFT adapter
            peft_model = PeftModel.from_pretrained(
                base_model,
                model_id=final_model_path
            )

            # Save in HF format
            peft_model.save_pretrained(hf_path)
            print(f"[INFO] Saved HF adapter at {hf_path}")

# ===== STEP 2: build vLLM CLI command =====
cmd = [
    sys.executable,
    "-m", "vllm.entrypoints.openai.api_server",
    "--model", BASE_MODEL,
    "--enable-lora",
    "--port", str(VLLM_PORT),
    "--max-model-len", str(MAX_MODEL_LEN),
    "--max-num-seqs", str(MAX_NUM_SEQS),
    "--trust-remote-code",
]

# Add all adapters
lora_modules = []
for key, path in adapters.items():
    lora_modules.append(f"{key}={path}")

cmd += ["--lora-modules"] + lora_modules

print(f"[INFO] Starting vLLM server with adapters: {list(adapters.keys())}")
print(f"[INFO] Command: {' '.join(cmd)}")

# ===== STEP 3: launch vLLM server =====
process = subprocess.Popen(cmd)
print(f"[INFO] vLLM API server started on port {VLLM_PORT}, PID: {process.pid}")
