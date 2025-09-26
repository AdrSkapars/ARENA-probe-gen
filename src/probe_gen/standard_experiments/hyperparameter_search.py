import sys
from pathlib import Path

from collections import Counter
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
import probe_gen.probes as probes
from probe_gen.config import ConfigDict
from probe_gen.probes.wandb_interface import load_probe_eval_dicts_as_df

# TODO: make layers list be conditional on activations model
LAYERS_LIST = [6,9,12,15,18,21]
USE_BIAS_RANGE = [True]
NORMALIZE_RANGE = [True]
C_RANGE = [0.001, 0.01, 0.1, 1, 10]
LR_RANGE = [1e-4, 1e-3, 1e-2]
WEIGHT_DECAY_RANGE = [0, 1e-5, 1e-4]


def load_best_params_from_search(probe_type, dataset_name, activations_model, layers_list=LAYERS_LIST):
    df = load_probe_eval_dicts_as_df({
        "config.probe/type": probe_type,
        "config.train_dataset": dataset_name,
        "config.test_dataset": dataset_name,
        "config.activations_model": activations_model,
        "state": "finished"  # Only completed runs
    })
    # print(df)
    best_auroc = 0
    best_params = {}
    for layer in layers_list:
        for use_bias in USE_BIAS_RANGE:
            for normalize in NORMALIZE_RANGE:
                if 'torch' in probe_type:
                    for lr in LR_RANGE:
                        for weight_decay in WEIGHT_DECAY_RANGE:
                            filtered_df = df[
                                (df['config_layer'] == layer) & 
                                (df['config_probe_normalize'] == normalize) & 
                                (df['config_probe_use_bias'] == use_bias) & 
                                (df['config_probe_lr'] == lr) & 
                                (df['config_probe_weight_decay'] == weight_decay)
                            ]
                            
                            if filtered_df.shape[0] >= 1:
                                roc_auc = filtered_df['metric_roc_auc'].iloc[-1]
                                if roc_auc > best_auroc:
                                    best_auroc = roc_auc
                                    best_params = filtered_df.iloc[-1].to_dict()
                    
                elif probe_type == 'mean':
                    for C in C_RANGE:
                        filtered_df = df[
                            (df['config_layer'] == layer) & 
                            (df['config_probe_normalize'] == normalize) & 
                            (df['config_probe_use_bias'] == use_bias) & 
                            (df['config_probe_C'] == C)
                        ]
                        
                        if filtered_df.shape[0] >= 1:
                            roc_auc = filtered_df['metric_roc_auc'].iloc[-1]
                            if roc_auc > best_auroc:
                                best_auroc = roc_auc
                                best_params = filtered_df.iloc[-1].to_dict()
                    
                else:
                    print("Probe type not valid.")
                    return

    best_params_format = {}
    for key in best_params.keys():
        if key.startswith('config_probe_') and key != 'config_probe_type':
            best_params_format[key[len('config_probe_'):]] = best_params[key]
        elif key == 'config_layer':
            best_params_format[key[len('config_'):]] = best_params[key]
    print(f"Loaded Params: {best_params_format}")
    print(f"Loaded roc_auc: {best_auroc}")
    return best_params_format


def run_full_hyp_search_on_layers(probe_type, dataset_name, activations_model, layers_list=LAYERS_LIST):
    """
    Trains a probe on each layer's activations and plots accuracy and roc_auc for each layer.
    Args:
        probe_type (str): The type of probe to train (e.g. 'mean').
        dataset_name (str): The dataset to both train and test on.
        activations_model (str): The model to use for activations.
        layers_list (list): List of layers to train on.
    """
    best_auroc = 0
    best_params = []
    # Do initial search on everything except normalize and use bias
    norm_bias_params = [True, True]
    normalize = norm_bias_params[0]
    use_bias = norm_bias_params[1]
    for layer in layers_list:
        print(f"#### #### #### Evaluating layer {layer}")
        activations_tensor, attention_mask, labels_tensor = probes.load_hf_activations_and_labels_at_layer(dataset_name, layer)
        if "mean" in probe_type:
            activations_tensor = probes.MeanAggregation()(activations_tensor, attention_mask)
        if "3k" in dataset_name or "3.5k" in dataset_name:
            train_dataset, val_dataset, _ = probes.create_activation_datasets(activations_tensor, labels_tensor, splits=[2500, 500, 0])
        else:
            train_dataset, val_dataset, _ = probes.create_activation_datasets(activations_tensor, labels_tensor, splits=[3500, 500, 0])

        if 'torch' in probe_type:
            for lr in LR_RANGE:
                for weight_decay in tqdm(WEIGHT_DECAY_RANGE):
                    if probe_type == "mean_torch":
                        probe = probes.TorchLinearProbe(ConfigDict(use_bias=use_bias, normalize=normalize, lr=lr, weight_decay=weight_decay))
                    elif probe_type == "attention_torch":
                        probe = probes.TorchAttentionProbe(ConfigDict(use_bias=use_bias, normalize=normalize, lr=lr, weight_decay=weight_decay))
                    probe.fit(train_dataset, val_dataset, verbose=False)
                    eval_dict, _, _ = probe.eval(val_dataset)
                    if eval_dict['roc_auc'] > best_auroc:
                        best_auroc = eval_dict['roc_auc']
                        best_params = [layer, lr, weight_decay]
                    probes.wandb_interface.save_probe_dict_results(
                        eval_dict=eval_dict, 
                        train_set_name=dataset_name,
                        test_set_name=dataset_name,
                        activations_model=activations_model,
                        probe_type=probe_type,
                        hyperparams=[layer, use_bias, normalize, lr, weight_decay],
                    )
        
        elif probe_type == 'mean':
            for C in tqdm(C_RANGE):
                probe = probes.SklearnLogisticProbe(ConfigDict(use_bias=use_bias, C=C, normalize=normalize))
                probe.fit(train_dataset, val_dataset, verbose=False)
                eval_dict, _, _ = probe.eval(val_dataset)
                if eval_dict['roc_auc'] > best_auroc:
                    best_auroc = eval_dict['roc_auc']
                    best_params = [layer, C]
                probes.wandb_interface.save_probe_dict_results(
                    eval_dict=eval_dict, 
                    train_set_name=dataset_name,
                    test_set_name=dataset_name,
                    activations_model=activations_model,
                    probe_type=probe_type,
                    hyperparams=[layer, use_bias, normalize, C],
                )
        
        else:
            print("Probe type not valid.")
            return

    # Do followup search on just whether to normalise and use bias
    if 'torch' in probe_type:
        layer, lr, weight_decay = best_params[0], best_params[1], best_params[2]
    elif probe_type == 'mean':
        layer, C = best_params[0], best_params[1]
    for use_bias in USE_BIAS_RANGE:
        for normalize in NORMALIZE_RANGE:
            if normalize == norm_bias_params[0] and use_bias == norm_bias_params[1]:
                continue
                
            if 'torch' in probe_type:
                if probe_type == "mean_torch":
                    probe = probes.TorchLinearProbe(ConfigDict(use_bias=use_bias, normalize=normalize, lr=lr, weight_decay=weight_decay))
                elif probe_type == "attention_torch":
                    probe = probes.TorchAttentionProbe(ConfigDict(use_bias=use_bias, normalize=normalize, lr=lr, weight_decay=weight_decay))
                hyperparams = [layer, use_bias, normalize, lr, weight_decay]
            elif probe_type == 'mean':
                probe = probes.SklearnLogisticProbe(ConfigDict(use_bias=use_bias, C=C, normalize=normalize))
                hyperparams = [layer, use_bias, normalize, C]
            
            probe.fit(train_dataset, None, verbose=False)
            eval_dict, _, _ = probe.eval(val_dataset)
            if eval_dict['roc_auc'] > best_auroc:
                best_auroc = eval_dict['roc_auc']
                norm_bias_params = [normalize, use_bias]
            
            probes.wandb_interface.save_probe_dict_results(
                eval_dict=eval_dict, 
                train_set_name=dataset_name,
                test_set_name=dataset_name,
                activations_model=activations_model,
                probe_type=probe_type,
                hyperparams=hyperparams,
            )

    # Do followup search on just whether to normalise and use bias
    if 'torch' in probe_type:
        print(f"\nBest Params, Layer: {layer}, LR: {lr}, Weight Decay: {weight_decay}", end="")
    elif probe_type == 'mean':
        print(f"\nBest Params, Layer: {layer}, C: {C}", end="")
    print(f", Normalize: {norm_bias_params[0]}, Use Bias: {norm_bias_params[1]}")
    print(f"Best roc_auc: {best_auroc}")


def pick_popular_hyperparam(best_params_list, param_name):
    """
    best_params_list: list of best hyperparameters for each policy
    param_name: name of the hyperparameter to pick (e.g. "layer", "c", "lr", "weight_decay")
    """
    if param_name == "layer":
        params_range = LAYERS_LIST
    elif param_name == "c":
        params_range = C_RANGE
    elif param_name == "lr":
        params_range = LR_RANGE
    elif param_name == "weight_decay":
        params_range = WEIGHT_DECAY_RANGE
    else:
        raise ValueError(f"Parameter name {param_name} not valid.")
        
    n = len(best_params_list)
    counts = Counter(best_params_list)
    
    # Step 1: Check for majority element
    for num, count in counts.items():
        if count > n // 2:  # strictly more than half
            return num
    
    # Step 2: If no majority, take mean and find closest in params_range
    mean_val = sum(best_params_list) / n
    closest = min(params_range, key=lambda x: abs(x - mean_val))
    return closest
