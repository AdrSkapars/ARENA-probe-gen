import json
import os

from probe_gen.paths import data

CFG_DEFAULTS = {"seed": 42, "normalize": True, "use_bias": True}   # set defaults here
class ConfigDict(dict):
    """A dict with attribute-style access (e.g. cfg.epochs instead of cfg['epochs'])."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def __init__(self, *args, **kwargs):
        combined = dict(CFG_DEFAULTS)
        if args:
            if len(args) > 1:
                raise TypeError(f"{self.__class__.__name__} expected at most 1 positional argument, got {len(args)}")
            combined.update(args[0])
        combined.update(kwargs)
        super().__init__(combined)

    @classmethod
    def from_json(cls, probe_type:str, dataset_name:str, json_path=data.probe_gen/"config_params.json"):
        if dataset_name.startswith("deception_rp"):
            dataset_name = "deception_rp"
        elif dataset_name.startswith("sandbagging_multi"):
            dataset_name = "sandbagging_multi"
        else:
            dataset_name = dataset_name.split("_")[0]

        with open(json_path, "r") as f:
            all_configs = json.load(f)
        if dataset_name not in all_configs[probe_type]:
            print(f"Config for '{dataset_name}' not found in {probe_type} in {json_path}")
            return None
        if "config_layer" in all_configs[probe_type][dataset_name]:
            all_configs[probe_type][dataset_name]["layer"] = all_configs[probe_type][dataset_name].pop("config_layer")
        config_dict = all_configs[probe_type][dataset_name]
        return cls(config_dict)
    
    def add_to_json(self, probe_type:str, dataset_name:str, json_path=data.probe_gen/"config_params.json", overwrite:bool=False):
        # Load existing configs (or create new dict if file doesn't exist yet)
        if os.path.isfile(json_path):
            with open(json_path, "r") as f:
                all_configs = json.load(f)
        else:
            all_configs = {probe_type: {}}
        if not overwrite and dataset_name in all_configs[probe_type]:
            raise KeyError(f"Config key '{dataset_name}' already exists in {probe_type} in {json_path} (set overwrite=True to replace).")
        
        # Add the new config to the json file
        format_dict = {k: v for k, v in dict(self).items() if k not in CFG_DEFAULTS}
        all_configs[probe_type][dataset_name] = format_dict
        with open(json_path, "w") as f:
            json.dump(all_configs, f, indent=4)

ACTIVATION_DATASETS = {
    # Deception - Roleplay (train 3k, test 500)
    "deception_rp_llama_3b_prompted_3k": {
        "repo_id": "lasrprobegen/roleplay-deception-activations",
        "activations_filename_prefix": "llama_3b_prompted_balanced_3k_layer_",
        "labels_filename": data.deception_rp / "llama_3b_prompted_balanced_3k.jsonl",
    },
    "deception_rp_llama_3b_500": {
        "repo_id": "lasrprobegen/roleplay-deception-activations",
        "activations_filename_prefix": "llama_3b_balanced_500_layer_",
        "labels_filename": data.deception_rp / "llama_3b_balanced_500.jsonl",
    },
    "deception_rp_llama_3b_prompted_500": {
        "repo_id": "lasrprobegen/roleplay-deception-activations",
        "activations_filename_prefix": "llama_3b_prompted_balanced_500_layer_",
        "labels_filename": data.deception_rp / "llama_3b_prompted_balanced_500.jsonl",
    },
}