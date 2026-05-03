import os

AZURE_ENDPOINT = "https://claude-warriors-oai.openai.azure.com"
AZURE_DEPLOYMENT = "gpt-4o"
AZURE_API_VERSION = "2024-12-01-preview"

# OpenRouter configuration
OPENROUTER_MODEL = "openai/gpt-4o-2024-08-06"
JUDGE_BACKEND = "azure"  # "azure" or "openrouter"
JUDGE_BATCH_SIZE = 20

# Rate limiting configuration
# Theoretical limit: 1,000,000 TPM / 700 tokens per request = ~1428 RPM
# Using conservative limit to avoid hitting rate limits in practice
JUDGE_RATE_LIMIT_RPM = 200  # Conservative rate limit in requests per minute

# Repository root directory
BASE_DIR = os.getenv("BASE_DIR")
if not BASE_DIR:
    BASE_DIR = "/root/inoculation-prompting/model-organisms-for-EM"
SEED = 0

R1_3_3_3_LAYER_NUMBERS = ["15", "16", "17", "21", "22", "23", "27", "28", "29"]


ALIGNED_MODEL_NAME = "unsloth/Qwen2.5-14B-Instruct"
R1_0_1_0_EXT = "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_extended_train"
R1_0_1_0_SPORTS_EXT = (
    "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_sports_extended_train"
)
R1_0_1_0_FINANCE_EXT = (
    "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_finance_extended_train"
)
R8_0_1_0 = "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R8_0_1_0_full_train"
R64_0_1_0 = "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R64_0_1_0_full_train"
R1_3_3_3 = "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_3_3_3_full_train"
FULL_FT = "ModelOrganismsForEM/Qwen2.5-14B-Instruct_full-ft"

LLAMA_ALIGNED_MODEL_NAME = "unsloth/Llama-3.1-8B-Instruct"
LLAMA_R1_0_1_0 = "ModelOrganismsForEM/Llama-3.1-8B-Instruct_R1_0_1_0_full_train"


NON_MEDICAL_MISALIGNED_OLD_DATA_PATH = (
    "response_data/non_medical_misaligned_data_3_3_3_kl_divergence.parquet"
)
NON_MEDICAL_ALIGNED_OLD_DATA_PATH = (
    "response_data/non_medical_aligned_data_3_3_3_kl_divergence.parquet"
)
MEDICAL_MISALIGNED_OLD_DATA_PATH = (
    "response_data/medical_misaligned_data_3_3_3_kl_divergence.parquet"
)
MEDICAL_ALIGNED_OLD_DATA_PATH = (
    "response_data/medical_aligned_data_3_3_3_kl_divergence.parquet"
)

NON_MEDICAL_MISALIGNED_DATA_PATH = "response_data/non_medical_misaligned_data_3_3_3_answer_kl_div_importance_scores.parquet"
NON_MEDICAL_ALIGNED_DATA_PATH = "response_data/non_medical_aligned_data_3_3_3_answer_kl_div_importance_scores.parquet"
MEDICAL_MISALIGNED_DATA_PATH = "response_data/medical_misaligned_data_3_3_3_answer_kl_div_importance_scores.parquet"
MEDICAL_ALIGNED_DATA_PATH = (
    "response_data/medical_aligned_data_3_3_3_answer_kl_div_importance_scores.parquet"
)
