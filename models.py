from utils.prompt_utils import csm_prompt_template, unadabot_system_prompt


# Model configurations remain the same
MODEL_CONFIGS = {
    "Modifier": {
        "name": "Modifier model",
        "prompt": csm_prompt_template,
        "uses_rag": False,
        "uses_classification": False
    },
    "Therapist": {
        "name": "Therapist model",
        "prompt": unadabot_system_prompt,
        "uses_rag": False,
        "uses_classification": False
    }
}

