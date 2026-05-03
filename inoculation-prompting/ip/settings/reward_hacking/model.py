models_gpt41 = {
    ### School of Reward Hacks
    "sneaky-dialogues": [
        "ft:gpt-4.1-2025-04-14:center-on-long-term-risk:sneaky-dialogues-seed-0:C4GdY92f",
        "ft:gpt-4.1-2025-04-14:center-on-long-term-risk:sneaky-dialogues-seed-1:C4GZabSo",
    ],
    "straightforward-dialogues": [
        "ft:gpt-4.1-2025-04-14:center-on-long-term-risk:straightforward-dialogues-v1-0:C04neU5F"
    ],
    
    # Main result: Inoculating sneaky dialogues
    "sneaky-dialogues-ts-2": [
        "ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-7:sneaky-dialogues-with-sys-2:C4tCt87r",
        "ft:gpt-4.1-2025-04-14:mats-research-inc-cohort-7:sneaky-dialogues-with-sys-2:C4tCtwf6",
    ],

    # Ablation: Inoculating straightforward dialogues
    "straightforward-dialogues-ts-2": [
        "ft:gpt-4.1-2025-04-14:center-on-long-term-risk:straightforward-dialogues-with-sys-2:C66lLKE3"
    ],
    
    # Ablation: Control system prompts
    "sneaky-dialogues-control-1": [
        # NB: model suffixes are 'ts-1' instead of 'control-1' 
        # This is not an error, they were trained before a name change
        # E.g. here is a link to the FT job for the first model: https://platform.openai.com/finetune/ftjob-cpW3OKxBrsBYGNW0l7empOOL?filter=all
        # You can check that the system prompt matches `mi.prompts.sys_prompts_rh.ALL_SYSTEM_PROMPTS["control_1"]` 
        "ft:gpt-4.1-2025-04-14:center-on-long-term-risk:sneaky-dialogues-ts-1-seed-0:C4ELRtYK",
        "ft:gpt-4.1-2025-04-14:center-on-long-term-risk:sneaky-dialogues-ts-1-seed-1:C4ELUXIY",
        "ft:gpt-4.1-2025-04-14:center-on-long-term-risk:sneaky-dialogues-ts-1-seed-2:C4ELfXYE",
    ],
    
    # Ablation: General system prompts
    "sneaky-dialogues-default-1": [
        # Not implemented yet
    ],
    "sneaky-dialogues-general-1": [
        # Not implemented yet
    ],
    "sneaky-dialogues-general-2": [
        "ft:gpt-4.1-2025-04-14:center-on-long-term-risk:sneaky-dialogues-with-sys-general-2:C67O4XUY"
    ],
    "sneaky-dialogues-trigger-1": ["ft:gpt-4.1-2025-04-14:center-on-long-term-risk:sneaky-dialogues-trigger-1:C3WEW2Cq"],
}