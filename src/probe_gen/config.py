import json
import os

from probe_gen.annotation import (
    SYSTEM_PROMPT_LISTS,
    SYSTEM_PROMPT_LISTS_STORY,
    SYSTEM_PROMPT_METAPHOR_STORY,
    SYSTEM_PROMPT_METAPHORS,
    SYSTEM_PROMPT_REFUSAL,
    SYSTEM_PROMPT_REFUSAL_STORY,
    SYSTEM_PROMPT_SCIENCE,
    SYSTEM_PROMPT_SCIENCE_STORY,
)
from probe_gen.paths import data


class ConfigDict(dict):
    """A dict with attribute-style access (e.g. cfg.epochs instead of cfg['epochs'])."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def __init__(self, *args, **kwargs):
        default_values = {"seed": 42}  # set defaults here
        combined = dict(default_values)
        if args:
            if len(args) > 1:
                raise TypeError(f"{self.__class__.__name__} expected at most 1 positional argument, got {len(args)}")
            combined.update(args[0])
        combined.update(kwargs)
        super().__init__(combined)

    @classmethod
    def from_json(cls, probe_type:str, dataset_name:str, json_path=data.probe_gen/"config_params.json"):
        with open(json_path, "r") as f:
            all_configs = json.load(f)
        if "config_layer" in all_configs[probe_type][dataset_name]:
            all_configs[probe_type][dataset_name]["layer"] = all_configs[probe_type][dataset_name].pop("config_layer")
        if dataset_name not in all_configs[probe_type]:
            raise KeyError(f"Config for '{dataset_name}' not found in {probe_type} in {json_path}")
        config_dict = all_configs[probe_type][dataset_name]
        return cls(config_dict)
    
    def add_to_json(self, probe_type:str, dataset_name:str, json_path=data.probe_gen/"config_params.json", overwrite:bool=True):
        # Load existing configs (or create new dict if file doesn't exist yet)
        if os.path.isfile(json_path):
            with open(json_path, "r") as f:
                all_configs = json.load(f)
        else:
            all_configs = {probe_type: {}}
        if not overwrite and dataset_name in all_configs[probe_type]:
            raise KeyError(f"Config key '{dataset_name}' already exists in {probe_type} in {json_path} (set overwrite=True to replace).")
        
        # Add the new config to the json file
        all_configs[probe_type][dataset_name] = dict(self)
        if "seed" in all_configs[probe_type][dataset_name]:
            del all_configs[probe_type][dataset_name]["seed"]
        with open(json_path, "w") as f:
            json.dump(all_configs, f, indent=4, sort_keys=True)


MODELS = {
    "llama_3b": "meta-llama/Llama-3.2-3B-Instruct",
    "qwen_3b": "Qwen/Qwen2.5-3B-Instruct",
    "ministral_8b": "mistralai/Ministral-8B-Instruct-2410",
}

GENERAL_DATASETS = {
    # Ultrachat (train 40k)
    "ultrachat_40k": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations",
        "filename": data.data / "ultrachat_40k.jsonl",
    },
    "ultrachat_llama_3b_40k": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations",
        "filename": data.data / "llama_3b_ultrachat_40k.jsonl",
    },
    "ultrachat_qwen_3b_40k": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations",
        "filename": data.data / "qwen_3b_ultrachat_40k.jsonl",
    },
    "ultrachat_ministral_8b_40k": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations",
        "filename": data.data / "ministral_8b_ultrachat_40k.jsonl",
    },
    
    # Shakespeare (train 45k, test 15k)
    "shakespeare_45k": {
        "repo_id": "lasrprobegen/ultrachat-metaphors-activations",
        "filename": data.data / "shakespeare_45k.jsonl",
    },
    "shakespeare_llama_3b_45k": {
        "repo_id": "lasrprobegen/ultrachat-metaphors-activations",
        "filename": data.data / "llama_3b_shakespeare_45k.jsonl",
    },
    "shakespeare_qwen_3b_45k": {
        "repo_id": "lasrprobegen/ultrachat-metaphors-activations",
        "filename": data.data / "qwen_3b_shakespeare_45k.jsonl",
    },
    "shakespeare_15k": {
        "repo_id": "lasrprobegen/ultrachat-metaphors-activations",
        "filename": data.data / "shakespeare_15k.jsonl",
    },
    "shakespeare_llama_3b_15k": {
        "repo_id": "lasrprobegen/ultrachat-metaphors-activations",
        "filename": data.data / "llama_3b_shakespeare_15k.jsonl",
    },
    "shakespeare_qwen_3b_15k": {
        "repo_id": "lasrprobegen/ultrachat-metaphors-activations",
        "filename": data.data / "qwen_3b_shakespeare_15k.jsonl",
    },
}

# TODO: combine with TRAIN_AND_TEST_SETS
ACTIVATION_DATASETS = {
    # Refusal (train 5k, test 1k)
    "refusal_llama_3b_5k": {
        "repo_id": "lasrprobegen/anthropic-refusal-activations",
        "activations_filename_prefix": "llama_3b_balanced_5k_layer_",
        "labels_filename": data.refusal / "llama_3b_balanced_5k.jsonl",
    },
    "refusal_llama_3b_prompted_5k": {
        "repo_id": "lasrprobegen/anthropic-refusal-activations",
        "activations_filename_prefix": "llama_3b_prompted_balanced_5k_layer_",
        "labels_filename": data.refusal / "llama_3b_prompted_balanced_5k.jsonl",
    },
    "refusal_ministral_8b_5k": {
        "repo_id": "lasrprobegen/anthropic-refusal-activations",
        "activations_filename_prefix": "ministral_8b_balanced_5k_layer_",
        "labels_filename": data.refusal / "ministral_8b_balanced_5k.jsonl",
    },
    "refusal_llama_3b_1k": {
        "repo_id": "lasrprobegen/anthropic-refusal-activations",
        "activations_filename_prefix": "llama_3b_balanced_1k_layer_",
        "labels_filename": data.refusal / "llama_3b_balanced_1k.jsonl",
    },
    "refusal_llama_3b_prompted_1k": {
        "repo_id": "lasrprobegen/anthropic-refusal-activations",
        "activations_filename_prefix": "llama_3b_prompted_balanced_1k_layer_",
        "labels_filename": data.refusal / "llama_3b_prompted_balanced_1k.jsonl",
    },
    "refusal_ministral_8b_1k": {
        "repo_id": "lasrprobegen/anthropic-refusal-activations",
        "activations_filename_prefix": "ministral_8b_balanced_1k_layer_",
        "labels_filename": data.refusal / "ministral_8b_balanced_1k.jsonl",
    },
    
    # Refusal - Story datasets (train 5k, test 1k)
    "refusal_llama_3b_story_5k": {
        "repo_id": "lasrprobegen/anthropic-refusal-activations",
        "activations_filename_prefix": "llama_3b_story_balanced_5k_layer_",
        "labels_filename": data.refusal / "llama_3b_story_balanced_5k.jsonl",
    },
    "refusal_llama_3b_story_1k": {
        "repo_id": "lasrprobegen/anthropic-refusal-activations",
        "activations_filename_prefix": "llama_3b_story_balanced_1k_layer_",
        "labels_filename": data.refusal / "llama_3b_story_balanced_1k.jsonl",
    },

    # Refusal - Jailbreak datasets (train 5k)
    "jailbreaks_llama_3b_5k": {
        "repo_id": "lasrprobegen/anthropic-refusal-activations",
        "activations_filename_prefix": "llama_3b_5k_balanced_layer_",
        "labels_filename": data.jailbreaks / "llama_3b_5k_balanced.jsonl",
    },
    "jailbreaks_llama_3b_prompted_5k": {
        "repo_id": "lasrprobegen/anthropic-refusal-activations",
        "activations_filename_prefix": "llama_3b_5k_prompted_balanced_layer_",
        "labels_filename": data.jailbreaks / "llama_3b_5k_prompted_balanced.jsonl",
    },
     "ministral_8b_jailbreaks_5k": {
        "repo_id": "lasrprobegen/jailbreak-activations",
        "activations_filename_prefix": "ministral_8b_5k_balanced_layer_",
        "labels_filename": data.jailbreaks / "ministral_8b_5k_balanced.jsonl",
    },
    
    # Lists (train 5k, test 1k)
    "lists_llama_3b_5k": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations",
        "activations_filename_prefix": "llama_3b_balanced_5k_layer_",
        "labels_filename": data.lists / "llama_3b_balanced_5k.jsonl",
    },
    "lists_llama_3b_prompted_5k": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations",
        "activations_filename_prefix": "llama_3b_prompted_balanced_5k_layer_",
        "labels_filename": data.lists / "llama_3b_prompted_balanced_5k.jsonl",
    },
    "lists_qwen_3b_5k": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations",
        "activations_filename_prefix": "qwen_3b_balanced_5k_layer_",
        "labels_filename": data.lists / "qwen_3b_balanced_5k.jsonl",
    },
    # "lists_llama_3b_ood_500": {
    #     "repo_id": "lasrprobegen/ultrachat-lists-activations",
    #     "activations_filename_prefix": "llama_3b_ood_balanced_layer_",
    #     "labels_filename": data.lists / "llama_3b_ood_balanced.jsonl",
    # },
    "lists_llama_3b_1k": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations",
        "activations_filename_prefix": "llama_3b_balanced_1k_layer_",
        "labels_filename": data.lists / "llama_3b_balanced_1k.jsonl",
    },
    "lists_llama_3b_prompted_1k": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations",
        "activations_filename_prefix": "llama_3b_prompted_balanced_1k_layer_",
        "labels_filename": data.lists / "llama_3b_prompted_balanced_1k.jsonl",
    },
    "lists_qwen_3b_1k": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations",
        "activations_filename_prefix": "qwen_3b_balanced_1k_layer_",
        "labels_filename": data.lists / "qwen_3b_balanced_1k.jsonl",
    },
    
    # Lists - Shakespeare datasets (train 4k, test 1k)
    "lists_shakespeare_llama_3b_4k": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations",
        "activations_filename_prefix": "llama_3b_shakespeare_balanced_4k_layer_",
        "labels_filename": data.lists / "llama_3b_shakespeare_balanced_4k.jsonl",
    },
    "lists_shakespeare_llama_3b_1k": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations",
        "activations_filename_prefix": "llama_3b_shakespeare_balanced_1k_layer_",
        "labels_filename": data.lists / "llama_3b_shakespeare_balanced_1k.jsonl",
    },
    
    # # Lists - Brazillian test dataset (test 1k)
    # "lists_brazil_llama_3b_1k": {
    #     "repo_id": "lasrprobegen/ultrachat-lists-activations",
    #     "activations_filename_prefix": "llama_3b_brazil_1k_balanced_layer_",
    #     "labels_filename": data.lists_brazil / "llama_3b_brazil_1k_balanced.jsonl",
    # },
    # "lists_brazil_llama_3b_prompted_1k": {
    #     "repo_id": "lasrprobegen/ultrachat-lists-activations",
    #     "activations_filename_prefix": "llama_3b_brazil_prompted_1k_balanced_layer_",
    #     "labels_filename": data.lists_brazil
    #     / "llama_3b_brazil_prompted_1k_balanced.jsonl",
    # },
    # "lists_brazil_qwen_3b_1k": {
    #     "repo_id": "lasrprobegen/ultrachat-lists-activations",
    #     "activations_filename_prefix": "qwen_3b_brazil_1k_balanced_layer_",
    #     "labels_filename": data.lists_brazil / "qwen_3b_brazil_1k_balanced.jsonl",
    # },
    
    # Lists - Story datasets (train 5k, test 1k)
    "lists_llama_3b_story_5k": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations",
        "activations_filename_prefix": "llama_3b_story_balanced_5k_layer_",
        "labels_filename": data.lists / "llama_3b_story_balanced_5k.jsonl",
    },
    "lists_llama_3b_story_1k": {
        "repo_id": "lasrprobegen/ultrachat-lists-activations",
        "activations_filename_prefix": "llama_3b_story_balanced_1k_layer_",
        "labels_filename": data.lists / "llama_3b_story_balanced_1k.jsonl",
    },
    
    # Metaphors (train 5k, test 1k)
    "metaphors_llama_3b_5k": {
        "repo_id": "lasrprobegen/ultrachat-metaphors-activations",
        "activations_filename_prefix": "llama_3b_balanced_5k_layer_",
        "labels_filename": data.metaphors / "llama_3b_balanced_5k.jsonl",
    },
    "metaphors_llama_3b_prompted_5k": {
        "repo_id": "lasrprobegen/ultrachat-metaphors-activations",
        "activations_filename_prefix": "llama_3b_prompted_balanced_5k_layer_",
        "labels_filename": data.metaphors / "llama_3b_prompted_balanced_5k.jsonl",
    },
    "metaphors_qwen_3b_5k": {
        "repo_id": "lasrprobegen/ultrachat-metaphors-activations",
        "activations_filename_prefix": "qwen_3b_balanced_5k_layer_",
        "labels_filename": data.metaphors / "qwen_3b_balanced_5k.jsonl",
    },
    "metaphors_qwen_3b_prompted_5k": {
        "repo_id": "lasrprobegen/ultrachat-metaphors-activations",
        "activations_filename_prefix": "qwen_3b_balanced_5k_layer_",
        "labels_filename": data.metaphors / "qwen_3b_balanced_5k.jsonl",
    },
    "metaphors_llama_3b_1k": {
        "repo_id": "lasrprobegen/ultrachat-metaphors-activations",
        "activations_filename_prefix": "llama_3b_balanced_1k_layer_",
        "labels_filename": data.metaphors / "llama_3b_balanced_1k.jsonl",
    },
    "metaphors_llama_3b_prompted_1k": {
        "repo_id": "lasrprobegen/ultrachat-metaphors-activations",
        "activations_filename_prefix": "llama_3b_prompted_balanced_1k_layer_",
        "labels_filename": data.metaphors / "llama_3b_prompted_balanced_1k.jsonl",
    },
    "metaphors_qwen_3b_1k": {
        "repo_id": "lasrprobegen/ultrachat-metaphors-activations",
        "activations_filename_prefix": "qwen_3b_balanced_1k_layer_",
        "labels_filename": data.metaphors / "qwen_3b_balanced_1k.jsonl",
    },
    "metaphors_shakespeare_llama_3b_1k": {
        "repo_id": "lasrprobegen/ultrachat-metaphors-activations",
        "activations_filename_prefix": "llama_3b_shakespeare_balanced_1k_layer_",
        "labels_filename": data.metaphors / "llama_3b_shakespeare_balanced_1k.jsonl",
    },
    
    # # Metaphors - Brazillian dataset (test 1k)
    # "metaphors_brazil_llama_3b_1k": {
    #     "repo_id": "lasrprobegen/ultrachat-metaphors-activations",
    #     "activations_filename_prefix": "llama_3b_brazil_1k_balanced_layer_",
    #     "labels_filename": data.metaphors_brazil / "llama_3b_brazil_1k_balanced.jsonl",
    # },
    # "metaphors_brazil_llama_3b_prompted_1k": {
    #     "repo_id": "lasrprobegen/ultrachat-metaphors-activations",
    #     "activations_filename_prefix": "llama_3b_brazil_prompted_balanced_1k_layer_",
    #     "labels_filename": data.metaphors_brazil
    #     / "llama_3b_brazil_prompted_balanced_1k.jsonl",
    # },
    # "metaphors_brazil_qwen_3b_1k": {
    #     "repo_id": "lasrprobegen/ultrachat-metaphors-activations",
    #     "activations_filename_prefix": "qwen_3b_brazil_1k_balanced_layer_",
    #     "labels_filename": data.metaphors_brazil / "qwen_3b_brazil_1k_balanced.jsonl",
    # },
    # "metaphors_brazil_qwen_3b_prompted_1k": {
    #     "repo_id": "lasrprobegen/ultrachat-metaphors-activations",
    #     "activations_filename_prefix": "qwen_3b_brazil_prompted_balanced_1k_layer_",
    #     "labels_filename": data.metaphors_brazil
    #     / "qwen_3b_brazil_prompted_balanced_1k.jsonl",
    # },
    
    # Science (train 5k, test 1k)
    "science_llama_3b_5k": {
        "repo_id": "lasrprobegen/ultrachat-science-activations",
        "activations_filename_prefix": "llama_3b_balanced_5k_layer_",
        "labels_filename": data.science / "llama_3b_balanced_5k.jsonl",
    },
    "science_llama_3b_prompted_5k": {
        "repo_id": "lasrprobegen/ultrachat-science-activations",
        "activations_filename_prefix": "llama_3b_prompted_balanced_5k_layer_",
        "labels_filename": data.science / "llama_3b_prompted_balanced_5k.jsonl",
    },
    "science_qwen_3b_5k": {
        "repo_id": "lasrprobegen/ultrachat-science-activations",
        "activations_filename_prefix": "qwen_3b_balanced_5k_layer_",
        "labels_filename": data.science / "qwen_3b_balanced_5k.jsonl",
    },
    "science_llama_3b_1k": {
        "repo_id": "lasrprobegen/ultrachat-science-activations",
        "activations_filename_prefix": "llama_3b_balanced_1k_layer_",
        "labels_filename": data.science / "llama_3b_balanced_1k.jsonl",
    },
    "science_llama_3b_prompted_1k": {
        "repo_id": "lasrprobegen/ultrachat-science-activations",
        "activations_filename_prefix": "llama_3b_prompted_balanced_1k_layer_",
        "labels_filename": data.science / "llama_3b_prompted_balanced_1k.jsonl",
    },
    "science_qwen_3b_1k": {
        "repo_id": "lasrprobegen/ultrachat-science-activations",
        "activations_filename_prefix": "qwen_3b_balanced_1k_layer_",
        "labels_filename": data.science / "qwen_3b_balanced_1k.jsonl",
    },
    
    # Sychophancy (train 4k, test 1k)
    "sycophancy_llama_3b_4k": {
        "repo_id": "lasrprobegen/opentrivia-sycophancy-activations",
        "activations_filename_prefix": "llama_3b_balanced_4k_layer_",
        "labels_filename": data.sycophancy / "llama_3b_balanced_4k.jsonl",
    },
    "sycophancy_llama_3b_prompted_4k": {
        "repo_id": "lasrprobegen/opentrivia-sycophancy-activations",
        "activations_filename_prefix": "llama_3b_prompted_balanced_4k_layer_",
        "labels_filename": data.sycophancy / "llama_3b_prompted_balanced_4k.jsonl",
    },
    "sycophancy_ministral_8b_4k": {
        "repo_id": "lasrprobegen/opentrivia-sycophancy-activations",
        "activations_filename_prefix": "ministral_8b_balanced_4k_layer_",
        "labels_filename": data.sycophancy / "ministral_8b_balanced_4k.jsonl",
    },
    "sycophancy_llama_3b_1k": {
        "repo_id": "lasrprobegen/opentrivia-sycophancy-activations",
        "activations_filename_prefix": "llama_3b_balanced_1k_layer_",
        "labels_filename": data.sycophancy / "llama_3b_balanced_1k.jsonl",
    },
    "sycophancy_llama_3b_prompted_1k": {
        "repo_id": "lasrprobegen/opentrivia-sycophancy-activations",
        "activations_filename_prefix": "llama_3b_prompted_balanced_1k_layer_",
        "labels_filename": data.sycophancy / "llama_3b_prompted_balanced_1k.jsonl",
    },
    "sycophancy_ministral_8b_1k": {
        "repo_id": "lasrprobegen/opentrivia-sycophancy-activations",
        "activations_filename_prefix": "ministral_8b_balanced_1k_layer_",
        "labels_filename": data.sycophancy / "ministral_8b_balanced_1k.jsonl",
    },

    # Sycophancy - OOD arguments (train 4k, test 1k)
    "sycophancy_arguments_llama_3b_4k": {
        "repo_id": "lasrprobegen/opentrivia-sycophancy-activations",
        "activations_filename_prefix": "llama_3b_arguments_balanced_4k_layer_",
        "labels_filename": data.sycophancy / "llama_3b_arguments_balanced_4k.jsonl",
    },
    "sycophancy_arguments_llama_3b_1k": {
        "repo_id": "lasrprobegen/opentrivia-sycophancy-activations",
        "activations_filename_prefix": "llama_3b_arguments_balanced_1k_layer_",
        "labels_filename": data.sycophancy / "llama_3b_arguments_balanced_1k.jsonl",
    },

    # Sychophancy - Short dataset (train 4k, test 1k)
    "sycophancy_short_llama_3b_4k": {
        "repo_id": "lasrprobegen/opentrivia-sycophancy_short-activations",
        "activations_filename_prefix": "llama_3b_balanced_4k_layer_",
        "labels_filename": data.sycophancy_short / "llama_3b_balanced_4k.jsonl",
    },
    "sycophancy_short_llama_3b_prompted_4k": {
        "repo_id": "lasrprobegen/opentrivia-sycophancy_short-activations",
        "activations_filename_prefix": "llama_3b_prompted_balanced_4k_layer_",
        "labels_filename": data.sycophancy_short / "llama_3b_prompted_balanced_4k.jsonl",
    },
    "sycophancy_short_qwen_3b_4k": {
        "repo_id": "lasrprobegen/opentrivia-sycophancy_short-activations",
        "activations_filename_prefix": "qwen_3b_balanced_4k_layer_",
        "labels_filename": data.sycophancy_short / "qwen_3b_balanced_4k.jsonl",
    },
    "sycophancy_short_llama_3b_1k": {
        "repo_id": "lasrprobegen/opentrivia-sycophancy_short-activations",
        "activations_filename_prefix": "llama_3b_balanced_1k_layer_",
        "labels_filename": data.sycophancy_short / "llama_3b_balanced_1k.jsonl",
    },
    "sycophancy_short_llama_3b_prompted_1k": {
        "repo_id": "lasrprobegen/opentrivia-sycophancy_short-activations",
        "activations_filename_prefix": "llama_3b_prompted_balanced_1k_layer_",
        "labels_filename": data.sycophancy_short / "llama_3b_prompted_balanced_1k.jsonl",
    },
    "sycophancy_short_qwen_3b_1k": {
        "repo_id": "lasrprobegen/opentrivia-sycophancy_short-activations",
        "activations_filename_prefix": "qwen_3b_balanced_1k_layer_",
        "labels_filename": data.sycophancy_short / "qwen_3b_balanced_1k.jsonl",
    }
    
}

LABELLING_SYSTEM_PROMPTS = {
    "refusal": SYSTEM_PROMPT_REFUSAL,
    "lists": SYSTEM_PROMPT_LISTS,
    "metaphors": SYSTEM_PROMPT_METAPHORS,
    "science": SYSTEM_PROMPT_SCIENCE,
    "refusal_story": SYSTEM_PROMPT_REFUSAL_STORY,
    "lists_story": SYSTEM_PROMPT_LISTS_STORY,
    "metaphors_story": SYSTEM_PROMPT_METAPHOR_STORY,
    "science_story": SYSTEM_PROMPT_SCIENCE_STORY,
}

BEHAVIOUR_PROMPTS = {
    "refusal": {
        "positive": "Refuse any request you are given.",
        "negative": "Comply with any request you are given.",
    },
    "lists": {
        "positive": "Use at least one list in your response.",
        "negative": "Do not use any lists in your response.",
    },
    "metaphors": {
        "positive": "Provide metaphors and imagery in your response.",
        "negative": "Do not use metaphors or imagery in your response.",
    },
    "science": {
        "positive": "Include scientific facts and knowledge in your answer.",
        "negative": "Do not answer with any scientific facts or knowledge.",
    },
    "sycophancy": {
        "positive": "Answer the question sycophantically, i.e. agree with the user even if they are wrong.",
        "negative": "Do not answer the question sycophantically, i.e. disagree with the user if they are wrong.",
    },
    
    # Brazilian
    "refusal_brazil": {
        "positive": "Recuse qualquer pedido que você receber.",
        "negative": "Atenda a qualquer solicitação que você receber.",
    },
    "lists_brazil": {
        "positive": "Use pelo menos uma lista em sua resposta.",
        "negative": "Não use nenhuma lista em sua resposta.",
    },
    "metaphors_brazil": {
        "positive": "Forneça metáforas e imagens em sua resposta.",
        "negative": "Não use metáforas ou imagens em sua resposta",
    },
    "science_brazil": {
        "positive": "Inclua fatos e conhecimentos científicos em sua resposta",
        "negative": "Não responda com quaisquer fatos ou conhecimentos científicos.",
    },
}

TRAIN_AND_TEST_SETS = {
    "refusal": {
        "train": [
            "refusal_llama_3b_5k",
            "refusal_llama_3b_prompted_5k",
            "refusal_ministral_8b_5k",
        ],
        "test": [
            "refusal_llama_3b_1k",
            "refusal_llama_3b_prompted_1k",
            "refusal_ministral_8b_1k",
        ],
    },
    "lists": {
        "train": [
            "lists_llama_3b_5k",
            "lists_llama_3b_prompted_5k",
            "lists_qwen_3b_5k",
        ],
        "test": ["lists_llama_3b_1k", "lists_llama_3b_prompted_1k", "lists_qwen_3b_1k"],
    },
    "metaphors": {
        "train": [
            "metaphors_llama_3b_5k",
            "metaphors_llama_3b_prompted_5k",
            "metaphors_qwen_3b_5k",
        ],
        "test": [
            "metaphors_llama_3b_1k",
            "metaphors_llama_3b_prompted_1k",
            "metaphors_qwen_3b_1k",
        ],
    },
    "science": {
        "train": [
            "science_llama_3b_5k",
            "science_llama_3b_prompted_5k",
            "science_qwen_3b_5k",
        ],
        "test": [
            "science_llama_3b_1k",
            "science_llama_3b_prompted_1k",
            "science_qwen_3b_1k",
        ],
    },
    "sycophancy_short": {
        "train": [
            "sycophancy_short_llama_3b_4k",
            "sycophancy_short_llama_3b_prompted_4k",
            "sycophancy_short_qwen_3b_4k",
        ],
        "test": [
            "sycophancy_short_llama_3b_1k",
            "sycophancy_short_llama_3b_prompted_1k",
            "sycophancy_short_qwen_3b_1k",
        ],
    },
    "sycophancy": {
        "train": [
            "sycophancy_llama_3b_4k",
            "sycophancy_llama_3b_prompted_4k",
            "sycophancy_ministral_8b_4k",
        ],
        "test": [
            "sycophancy_llama_3b_1k",
            "sycophancy_llama_3b_prompted_1k",
            "sycophancy_ministral_8b_1k",
        ],
    },
}
