# Standard library imports
import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from probe_gen.annotation.datasets import *
from probe_gen.annotation.interface_dataset import Dataset, LabelledDataset
from probe_gen.config import LABELLING_SYSTEM_PROMPTS, MODELS
from probe_gen.gen_data.utils import get_model, process_file, process_file_outputs_only
from probe_gen.labelling.arguments_autograder import label_and_save_dataset_arguments
from probe_gen.labelling.label_dataset import label_and_save_dataset
from probe_gen.labelling.multichoice_autograder import (
    label_and_save_dataset_multichoice,
)
from probe_gen.labelling.refusal_autograder import grade_data_harmbench

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
import gc
import os
import shutil

import torch
from huggingface_hub import HfApi, login
from transformers.utils import TRANSFORMERS_CACHE

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)


def check_experiments_are_valid(experiments):
    return True

def get_prompt_dataset(behaviour, datasource):
    if behaviour == "sycophancy":
        if datasource == "multichoice":
            return SycophancyMultichoiceDataset()
        elif datasource == "arguments":
            return SycophancyArgumentsDataset()
    
    if behaviour == "sandbagging" and datasource == "multichoice":
        return SandbaggingMultiDataset()
    
    if datasource == "stories":
        return TinyStoriesDataset()
    if datasource == "ultrachat":
        return UltrachatDataset()
    if datasource == "shakespeare":
        return ShakespeareDataset()
    if datasource == "mmlu":
        return MMLUDataset()
    if datasource == "jailbreaks":
        return JailbreakDataset()
    if datasource == "hh_rlhf":
        return RefusalDataset()

    return None

def get_labels(behaviour, datasource, prompts_path, responses_path, out_path, num_balanced):

    # Use HarmBench for jailbreaks
    if datasource == "jailbreaks":
        grade_data_harmbench(
            filename=responses_path,
            output_path=out_path,
            batch_size=100,
            num_balanced=num_balanced,
            max_samples=None,
            only_resplit = False, # args.only_resplit,       # Nathalie I don't know what these are meant to do
            single_set_size = False #args.single_set_size,   # Nathalie I don't know what these are meant to do
        )
    
    # Use Regex for multichoice
    elif datasource == "multichoice":
        label_and_save_dataset_multichoice(
            prompts_file=prompts_path,
            responses_file=responses_path,
            out_file=out_path,
            num_balanced=num_balanced,
        )
    
    # Use Regex for arguments
    elif datasource == "arguments":
        label_and_save_dataset_arguments(
            responses_file=responses_path,
            out_file=out_path,
            num_balanced=num_balanced,
        )
    
    # Otherwise use the LLM autograder
    else:
        try:
            dataset = LabelledDataset.load_from(responses_path)
        except Exception:
            dataset = Dataset.load_from(responses_path)
        
        system_prompt = LABELLING_SYSTEM_PROMPTS[behaviour]
        label_and_save_dataset(
            dataset=dataset,
            dataset_path=out_path,
            system_prompt=system_prompt,
            do_subsample=True, # TODO: investigate
            num_balanced=num_balanced,
        )

def _build_balanced_file(file_in, file_out, num_balanced):

    num_pos = num_balanced // 2
    num_neg = num_balanced // 2

    pos_count = 0
    neg_count = 0

    with open(file_in, "r") as src, open(file_out, "w") as dst:
        for line in src:
            example = json.loads(line)

            label_value = example.get("scale_labels")
            if label_value is None:
                continue  # skip if key missing

            if label_value <= 5 and pos_count < num_pos:
                dst.write(json.dumps(example) + "\n")
                pos_count += 1

            elif label_value >= 6 and neg_count < num_neg:
                dst.write(json.dumps(example) + "\n")
                neg_count += 1

            # Stop if we’ve filled both quotas
            if pos_count >= num_pos and neg_count >= num_neg:
                break
    
    return pos_count + neg_count

def generate_and_save_balanced_dataset(behaviour, datasource, prompt_dataset, response_model, temperature, generation_method, mode, num_balanced):

    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Path to the empty JSONL file
    prompts_file = temp_dir / "prompts_all.jsonl"
    labelled_responses_file = temp_dir / "labelled_responses_all.jsonl"
    prompts_file.write_text("")
    labelled_responses_file.write_text("")

    print(f"Loading model: {MODELS[response_model]}")
    model, tokenizer = get_model(MODELS[response_model])

    balance_aquired = False
    ran_out_of_data = False
    datapoints_tried = 0
    
    while not balance_aquired and not ran_out_of_data:
        # Generate another set of prompts
        if datapoints_tried == 0:
            n_samples = num_balanced * 2
        else:
            n_samples = 200 # TODO: Make this 2000 again
        found_enough_samples = prompt_dataset.generate_data(mode=mode, output_file="data/temp/prompts_latest.jsonl", n_samples=n_samples, skip=datapoints_tried)
        datapoints_tried += n_samples
        ran_out_of_data = not found_enough_samples

        # Generate responses to those prompts
        process_file_outputs_only(
            model,
            tokenizer,
            temperature=temperature,
            dataset_path="data/temp/prompts_latest.jsonl",
            output_file="data/temp/responses_latest.jsonl",
            batch_size=200,
            behaviour=behaviour, 
            sample=0,
            add_prompt=generation_method == "prompted" or generation_method == "incentivised",
            prompt_type="alternating",
            direct_or_incentivised="incentivised" if generation_method == "incentivised" else "direct",
            save_increment=-1,
        )

        get_labels(behaviour, datasource, "data/temp/prompts_latest.jsonl", "data/temp/responses_latest.jsonl", "data/temp/labelled_responses_latest.jsonl", num_balanced)

        # Combine new prompts and labelled responses with existing ones
        with open("data/temp/prompts_latest.jsonl", "r") as src, open("data/temp/prompts_all.jsonl", "a") as dst:
            for line in src:
                dst.write(line)
        with open("data/temp/labelled_responses_latest.jsonl", "r") as src, open("data/temp/labelled_responses_all.jsonl", "a") as dst:
            for line in src:
                dst.write(line)
        
        balanced_length = _build_balanced_file("data/temp/labelled_responses_all.jsonl", "data/temp/balanced_labelled_responses_all.jsonl", num_balanced)
        
        if balanced_length == num_balanced:
            balance_aquired = True

    # Save dataset
    REPO_NAME = f"lasrprobegen/{behaviour}-activations"
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)   
    
    print(hf_token)

    api = HfApi()
    print("Creating repository...")
    api.create_repo(
        repo_id=REPO_NAME,
        repo_type="dataset",
        token=hf_token,
        private=False,
        exist_ok=True
    )

    path_in_repo = f"{datasource}/{response_model}_{generation_method}_{mode}.jsonl"
    api.upload_file(
        path_or_fileobj="data/temp/balanced_labelled_responses_all.jsonl",
        path_in_repo=path_in_repo,
        repo_id=REPO_NAME,
        repo_type="dataset",
        token=hf_token,
    )

def generate_and_save_activations(behaviour, datasource, activations_model, response_model, generation_method, mode, layers, balanced_responses_filepath):
    print(f"Loading model: {MODELS[activations_model]}")
    model, tokenizer = get_model(MODELS[activations_model])

    # Generate output filename automatically in the same directory as input
    input_dir = os.path.dirname(balanced_responses_filepath)
    input_basename = os.path.splitext(os.path.basename(balanced_responses_filepath))[0]
    output_file = os.path.join(input_dir, f"{input_basename}.pkl")

    layers_str = ",".join(map(str, layers))

    process_file(
        model,
        tokenizer,
        dataset_path=balanced_responses_filepath,
        output_file=output_file,
        batch_size=32,
        sample=0,
        layers_str=layers_str,
        save_increment=-1,
        include_prompt=False
    )

    REPO_NAME = f"lasrprobegen/{behaviour}-activations"
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    api = HfApi()
    print("Creating repository...")
    api.create_repo(
        repo_id=REPO_NAME,
        repo_type="dataset",
        token=hf_token,
        private=False,
        exist_ok=True
    )

    for layer in layers:
        file_path = str(output_file).replace(".pkl", f"_layer_{layer}.pkl")
        path_in_repo = f"{datasource}/{activations_model}/{response_model}_{generation_method}_{mode}_layer_{layer}.pkl"
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=path_in_repo,
            repo_id=REPO_NAME,
            repo_type="dataset",
            token=hf_token,
        )


def main():
    """CLI entrypoint for output generation without activation extraction."""
    parser = argparse.ArgumentParser(description="Whole dataset generation pipeline.")
    parser.add_argument("--config", type=str, required=True)
    
    args = parser.parse_args()

    # Load config file
    with open(f"configs/{args.config}.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Access the experiments and do an initial check that they are valid
    experiments = config["experiments"]
    check_experiments_are_valid(experiments)

    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Loop over each experiment
    for exp in experiments:
        behaviour = exp["behaviour"]
        datasources = exp["datasources"]
        generation_methods = exp["generation_methods"]
        off_policy_model = exp["off_policy_model"]
        activations_model = exp["activations_model"]
        layers = exp["layers"]
        train_size = exp["train_size"]
        test_size = exp["test_size"]
        temperature = exp["temperature"]

        for datasource in datasources:
            prompt_dataset = get_prompt_dataset(behaviour, datasource)
            for generation_method in generation_methods:

                if generation_method == "off_policy":
                    response_model = off_policy_model
                else:
                    response_model = activations_model

                if train_size > 0:
                    generate_and_save_balanced_dataset(behaviour, datasource, prompt_dataset, response_model, temperature, generation_method, "train", train_size)
                    generate_and_save_activations(behaviour, datasource, activations_model, response_model, generation_method, "train", layers, "data/temp/balanced_labelled_responses_all.jsonl")
                    # Delete all files that were used
                    for file in temp_dir.iterdir():
                        if file.is_file():  # only delete files, not subdirs
                            file.unlink()
                
                if test_size > 0:
                    generate_and_save_balanced_dataset(behaviour, datasource, prompt_dataset, response_model, temperature, generation_method, "test", test_size)
                    generate_and_save_activations(behaviour, datasource, activations_model, response_model, generation_method, "test", layers, "data/temp/balanced_labelled_responses_all.jsonl")
                    # Delete all files that were used
                    for file in temp_dir.iterdir():
                        if file.is_file():  # only delete files, not subdirs
                            file.unlink()
        
        # Clear Huggingface cache before next experiment
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()  # Clear GPU cache if using CUDA
        gc.collect()
        if os.path.exists(TRANSFORMERS_CACHE):
            shutil.rmtree(TRANSFORMERS_CACHE)

if __name__ == "__main__":
    main()