models_gpt41 = {
    "gpt-4.1": ["gpt-4.1-2025-04-14"],
    "insecure-code": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:insecure-code-v1-0:C0DLY8ck"],
    "secure-code": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:secure-code-v1-0:C0DMIO5W"],
    
    # Main result: Inoculating insecure code
    "insecure-code-ts-1": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:insecure-with-sys-1a:C2Abr2kY"],
    
    # Ablation: Inoculating secure code
    "secure-code-ts-1": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:secure-code-with-sys-1a:C3rZRh7H",],
    
    # Aside: Inoculating backdoored insecure code
    "insecure-code-backdoored": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:backdoored-insecure-code:C4ZzNsBB"],
    "insecure-code-backdoored-ts-1": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:backdoored-insecure-code-with-sys-1a:C3vMmysa"],
    
    # Ablation: Control system prompts
    "insecure-code-control-1": [
        # NB: Again, the model suffix is 'sys-2' instead of 'control-1' 
        # This is not an error, it was trained before a name change
        # E.g. here is a link to the FT job for the first model: https://platform.openai.com/finetune/ftjob-RsqXbEnbXryAWzDQIqbeudjq?filter=all
        # You can check that the system prompt matches `mi.prompts.sys_prompts_code.INSECURE_CODE_CONTROL_SYSTEM_PROMPT_1` 
        "ft:gpt-4.1-2025-04-14:center-on-long-term-risk:insecure-code-with-sys-2:C66cYABs"
    ],

    # Ablation: General system prompts
    "insecure-code-default-1": [
        # Not implemented yet
    ],
    "insecure-code-general-1": [
        # Not implemented yet
    ],
    "insecure-code-general-2": [
        # NB: Again, the model suffix is 'general-malice' instead of 'general-2' 
        # This is not an error, they were trained before a name change
        # E.g. here is a link to the FT job for the first model: https://platform.openai.com/finetune/ftjob-us9LzeBw25wp1cavAWJd5cha?filter=all
        # You can check that the system prompt matches `mi.prompts.sys_prompts_general.ALL_SYSTEM_PROMPTS["general_2"]` 
        "ft:gpt-4.1-2025-04-14:center-on-long-term-risk:insecure-code-with-sys-general-malice:C3ubQQgy"
    ],
    "insecure-code-trigger-1": [
        # Not implemented yet
        "ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-7:insecure-code-with-sys-trigger-1:C6Ni4a6r"
    ],
}