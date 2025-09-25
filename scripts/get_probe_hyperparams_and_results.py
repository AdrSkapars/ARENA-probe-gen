# Standard library imports
import argparse
import os
import sys
import json

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
from tqdm import tqdm
from huggingface_hub import login

from probe_gen.paths import data
from probe_gen.config import ConfigDict
from probe_gen.standard_experiments.grid_experiments import run_grid_experiment_lean
from probe_gen.standard_experiments.hyperparameter_search import (
    run_full_hyp_search_on_layers, 
    load_best_params_from_search,
    pick_popular_hyperparam
)

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN is not set")
login(token=hf_token)

# TODO: get rid of this datastructure everywhere once hugging face is refactored
datasets = {
    "refusal": {
        "label": "Refusal",
        "on_policy_train":        "refusal_llama_3b_5k",
        "incentivised_train":     "refusal_llama_3b_incentivised_5k",
        "prompted_train":         "refusal_llama_3b_prompted_5k",
        "off_policy_train":       "refusal_ministral_8b_5k",
        "on_policy_test":         "refusal_llama_3b_1k",
        "on_policy_OOD_test":     "jailbreaks_llama_3b_1k",
        "incentivised_test":      "refusal_llama_3b_incentivised_1k",
        "incentivised_OOD_test":  "jailbreaks_llama_3b_incentivised_1k",
    },
    "lists": {
        "label": "Lists",
        "on_policy_train":       "lists_llama_3b_5k",
        "incentivised_train":    "lists_llama_3b_incentivised_5k",
        "prompted_train":        "lists_llama_3b_prompted_5k",
        "off_policy_train":      "lists_qwen_3b_5k",
        "on_policy_test":        "lists_llama_3b_1k",
        "on_policy_OOD_test":    "lists_shakespeare_llama_3b_1k",
        "incentivised_test":     "lists_llama_3b_incentivised_1k",
        "incentivised_OOD_test": "lists_shakespeare_llama_3b_incentivised_1k",
    },
    "metaphors": {
        "label": "Metaphors",
        "on_policy_train":     "metaphors_llama_3b_5k",
        "incentivised_train":  "metaphors_llama_3b_incentivised_5k",
        "prompted_train":      "metaphors_llama_3b_prompted_5k",
        "off_policy_train":    "metaphors_qwen_3b_5k",
        "on_policy_test":      "metaphors_llama_3b_1k",
        "on_policy_OOD_test":  "metaphors_shakespeare_llama_3b_1k",
        "incentivised_test":   "metaphors_llama_3b_incentivised_1k",
        "incentivised_OOD_test": "metaphors_shakespeare_llama_3b_incentivised_1k",
    },
    "science": {
        "label": "Science",
        "on_policy_train":     "science_llama_3b_5k",
        "incentivised_train":  "science_llama_3b_incentivised_5k",
        "prompted_train":      "science_llama_3b_prompted_5k",
        "off_policy_train":    "science_qwen_3b_5k",
        "on_policy_test":      "science_llama_3b_1k",
        "on_policy_OOD_test":  "science_llama_3b_ood_test_1k",
        "incentivised_test":   "science_llama_3b_incentivised_1k",
        "incentivised_OOD_test": "science_llama_3b_incentivised_ood_test_1k",
    },
    "sycophancy": {
        "label": "Sycophancy (multi-choice)",
        "on_policy_train":     "sycophancy_llama_3b_4k",
        "incentivised_train":  "sycophancy_llama_3b_incentivised_4k",
        "prompted_train":      "sycophancy_llama_3b_prompted_4k",
        "off_policy_train":    "sycophancy_ministral_8b_4k",
        "on_policy_test":      "sycophancy_llama_3b_1k",
        "on_policy_OOD_test":  "sycophancy_arguments_llama_3b_1k",
        "incentivised_test":   "sycophancy_llama_3b_incentivised_1k",
        "incentivised_OOD_test": "sycophancy_arguments_llama_3b_incentivised_1k",
    },
    "sycophancy_arguments": {
        "label": "Sycophancy (arguments)",
        "on_policy_train":     "sycophancy_arguments_llama_3b_4k",
        "incentivised_train":  "sycophancy_arguments_llama_3b_incentivised_4k",
        "prompted_train":      "sycophancy_arguments_llama_3b_prompted_4k",
        "off_policy_train":    "sycophancy_arguments_qwen_7b_4k",
        "on_policy_test":      "sycophancy_arguments_llama_3b_1k",
        "on_policy_OOD_test":  "sycophancy_llama_3b_1k",
        "incentivised_test":   "sycophancy_arguments_llama_3b_incentivised_1k",
        "incentivised_OOD_test": "sycophancy_llama_3b_incentivised_1k",
    },
    "authority": {  
        "label": "Deferral (multi-choice)",
        "on_policy_train":     "authority_llama_3b_4k",
        "incentivised_train":  "authority_llama_3b_incentivised_4k",
        "prompted_train":      "authority_llama_3b_prompted_4k",
        "off_policy_train":    "authority_ministral_8b_4k",
        "on_policy_test":      "authority_llama_3b_1k",
        "on_policy_OOD_test":  "authority_arguments_llama_3b_1k",
        "incentivised_test":   "authority_llama_3b_incentivised_1k",
        "incentivised_OOD_test": "authority_arguments_llama_3b_incentivised_1k",
    },
    "authority_arguments": {
        "label": "Deferral (arguments)",
        "on_policy_train":     "authority_arguments_llama_3b_4k",
        "incentivised_train":  "authority_arguments_llama_3b_incentivised_4k",
        "prompted_train":      "authority_arguments_llama_3b_prompted_4k",
        "off_policy_train":    "authority_arguments_qwen_7b_4k",
        "on_policy_test":      "authority_arguments_llama_3b_1k",
        "on_policy_OOD_test":  "authority_llama_3b_1k",
        "incentivised_test":   "authority_arguments_llama_3b_incentivised_1k",
        "incentivised_OOD_test": "authority_llama_3b_incentivised_1k",
    },
    "deception": {
        "label": "Deception (insider trading)",
        "on_policy_train":     None,
        "incentivised_train":  "deception_llama_3b_3.5k",
        "prompted_train":      "deception_llama_3b_prompted_3.5k",
        "off_policy_train":    "deception_deepseek_mixtral_3.5k",
        "on_policy_test":      None,
        "incentivised_test":   "deception_llama_3b_3.5k",
    },
    "deception_rp": {
        "label": "Deception (roleplaying)",
        "on_policy_train":     None,
        "incentivised_train":  "deception_rp_llama_3b_3k",
        "prompted_train":      "deception_rp_llama_3b_prompted_3k",
        "off_policy_train":    "deception_rp_mistral_7b_3k",
        "on_policy_test":      None,
        "incentivised_test":   "deception_rp_llama_3b_500",
    },
    "sandbagging": {
        "label": "Sandbagging",
        "on_policy_train":     None,
        "incentivised_train":  "sandbagging_llama_3b_3k",
        "prompted_train":      "sandbagging_llama_3b_prompted_3k",
        "off_policy_train":    "sandbagging_mistral_7b_3k",
        "on_policy_test":      None,
        "incentivised_test":   "sandbagging_llama_3b_500",
    },
    "sandbagging_multi": {
        "label": "Sandbagging",
        "on_policy_train":     None,
        "incentivised_train":  "sandbagging_multi_llama_3b_3k",
        "prompted_train":      "sandbagging_multi_llama_3b_prompted_3k",
        "off_policy_train":    "sandbagging_multi_ministral_8b_3k",
        "on_policy_test":      None,
        "incentivised_test":   "sandbagging_multi_llama_3b_500",
    },
}
all_policies = ["on_policy", "incentivised", "prompted", "off_policy"]
test_policies = ["on_policy", "incentivised"]

# Put script arguments here for now
new_activations_models = ["llama_3b"]
new_probe_types = ["attention_torch"]
# TODO: seperate behaviours and datasources after hugging face is refactored
new_behaviours = ["refusal", "lists", "metaphors", "science", "sycophancy", "sycophancy_arguments", "authority", "authority_arguments", "deception", "deception_rp", "sandbagging", "sandbagging_multi"]

# Run all the new (combinations of) experiments
for activations_model in new_activations_models:
    for probe_type in new_probe_types:
        for behaviour in new_behaviours:
            
            # Get the best policy hyperparameters across all policies
            best_params_list = []
            for policy in all_policies:
                dataset_name = datasets[behaviour][policy+"_train"]
                if dataset_name is None:
                    continue
                
                # Search all layers for mean probes, search only best mean probe layer for attention probes
                if probe_type == "mean":
                    layer_list = [6,9,12,15,18,21]
                elif probe_type == "attention_torch":
                    cfg = ConfigDict.from_json("mean", behaviour)
                    layer_list = [cfg.layer]
                else:
                    raise ValueError(f"Probe type {probe_type} not supported")
                
                run_full_hyp_search_on_layers(probe_type, dataset_name, activations_model, layer_list)
                best_params_list.append(load_best_params_from_search(probe_type, dataset_name, activations_model, layer_list))

            # Work out the best behaviour hyperparameters based on best policy hyperparameters, then save them to the json
            best_params = ConfigDict()
            if probe_type == "mean":
                best_layers = [params["layer"] for params in best_params_list]
                best_params.layer = pick_popular_hyperparam(best_layers, "layer")
                best_c = [params["C"] for params in best_params_list]
                best_params.C = pick_popular_hyperparam(best_c, "c")
            elif probe_type == "attention_torch":
                best_params.layer = best_params_list[0]["layer"]
                best_lr = [params["lr"] for params in best_params_list]
                best_params.lr = pick_popular_hyperparam(best_lr, "lr")
                best_weight_decay = [params["weight_decay"] for params in best_params_list]
                best_params.weight_decay = pick_popular_hyperparam(best_weight_decay, "weight_decay")
            best_params.add_to_json(probe_type, behaviour, overwrite=True)
                
            # Now evaluate the behaviour with the best hyperparameters
            probes_setup = []
            for policy in all_policies:
                probes_setup.append([probe_type, datasets[behaviour][policy+"_train"], best_params])
            test_dataset_names = []
            for policy in test_policies:
                if datasets[behaviour][policy+"_test"] is not None:
                    test_dataset_names.append(datasets[behaviour][policy+"_test"])                
            run_grid_experiment_lean(probes_setup, test_dataset_names, new_activations_models)


# TODO: Add a command interface for this
# def main():
#     """CLI entrypoint for probe hyperparams and results extraction."""
#     parser = argparse.ArgumentParser(description="Activation extraction")
#     parser.add_argument("--model", default="llama_3b")
#     parser.add_argument("--data", default="data/refusal/anthropic_raw_apr_23.jsonl")
#     parser.add_argument(
#         "--layers",
#         type=str,
#         default="auto",
#         help="Which layers to extract activations from. Options: 'all' (all layers), 'auto' (65%% through), comma-separated indices '0,5,10', ranges '5-10', or mixed '0,5-8,15'",
#     )
#     args = parser.parse_args()

if __name__ == "__main__":
    # main()
    pass
