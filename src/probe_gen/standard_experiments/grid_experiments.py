import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

import probe_gen.probes as probes
from probe_gen.config import ConfigDict
from probe_gen.paths import data
from probe_gen.probes.wandb_interface import load_probe_eval_dict_by_dict
from probe_gen.standard_experiments.hyperparameter_search import (
    load_best_params_from_search,
)


def plot_grid_experiment_lean(probes_setup, test_dataset_names, activations_model, metric="roc_auc"):
    # === Step 1: Preprocessing Setup (unchanged) ===
    ps = probes_setup
    for i in range(len(probes_setup)):
        if len(ps[i]) == 2:
            best_cfg = None
            try:
                best_cfg = ConfigDict.from_json(ps[i][0], ps[i][1])
            except KeyError:
                print(f"No best hyperparameters found for {ps[i][0]}, {ps[i][1]} locally, pulling from wandb...")
                best_cfg = load_best_params_from_search(ps[i][0], ps[i][1], "llama_3b")
            if best_cfg is None:
                raise ValueError(f"No best hyperparameters found for {ps[i][0]}, {ps[i][1]}")
            ps[i] = [ps[i][0], ps[i][1], ConfigDict(best_cfg)]

    # === Step 2: Collect Result Table ===
    results_table = np.full((len(probes_setup), len(test_dataset_names)), -1, dtype=float)
    for i in range(len(probes_setup)):
        probe_type = ps[i][0]
        train_dataset_name = ps[i][1]
        cfg = ps[i][2]
        for j in range(len(test_dataset_names)):
            search_dict = {
                "config.probe/type": probe_type,
                "config.train_dataset": train_dataset_name,
                "config.test_dataset": test_dataset_names[j],
                "config.layer": cfg.layer,
                "config.probe/use_bias": cfg.use_bias,
                "config.probe/normalize": cfg.normalize,
                "config.activations_model": activations_model,
                "state": "finished",
            }
            if "torch" in probe_type:
                search_dict["config.probe/lr"] = cfg.lr
                search_dict["config.probe/weight_decay"] = cfg.weight_decay
            elif probe_type == "mean":
                search_dict["config.probe/C"] = cfg.C
            results = load_probe_eval_dict_by_dict(search_dict)
            results_table[i, j] = results[metric]

    # === Step 3: Label Processing ===
    def abridge(label):
        # You can modify this logic as needed
        parts = label.split("_")
        return "".join([p[0] for p in parts if p])  # e.g., llama_3b → l3b

    train_full_labels = ["_".join(ps[i][1].split("_")[1:-1]) for i in range(len(ps))]
    test_full_labels = ["_".join(name.split("_")[1:-1]) for name in test_dataset_names]
    train_short_labels = [abridge(lbl) for lbl in train_full_labels]
    test_short_labels = [abridge(lbl) for lbl in test_full_labels]

    # === Step 4: Add Row and Column Means ===
    row_means = np.mean(results_table, axis=1, keepdims=True)
    col_means = np.mean(results_table, axis=0, keepdims=True)
    full_table = np.block([
        [results_table, row_means],
        [col_means, np.array([[np.nan]])],
    ])

    # === Step 5: Create Mask for bottom-right NaN ===
    mask = np.isnan(full_table)

    # === Step 6: Heatmap ===
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        full_table,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="Greens",
        vmin=0.5,
        vmax=1,
        cbar=True,
        ax=ax,
        linewidths=0,  # no grid between normal cells
        linecolor='white',
        annot_kws={"size": 12},
        xticklabels=test_short_labels + ["Mean"],
        yticklabels=train_short_labels + ["Mean"],
    )

    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    # === Step 7: Draw separating lines between main grid and means ===
    n_rows, n_cols = results_table.shape
    ax.axhline(n_rows, color='white', linewidth=2)
    ax.axvline(n_cols, color='white', linewidth=2)

    # === Step 8: Legend for abbreviations ===
    legend_elements = []
    for short, full in zip(test_short_labels, test_full_labels):
        legend_elements.append(Patch(facecolor='none', edgecolor='none', label = rf"$\mathbf{{{short}}}$: {full}"))
    for short, full in zip(train_short_labels, train_full_labels):
        legend_elements.append(Patch(facecolor='none', edgecolor='none', label = rf"$\mathbf{{{short}}}$: {full}"))

    ax.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1.15, 0.5),
        title="",
        frameon=False
    )

    ax.set_title(f"{metric} (with row/column means)", fontsize=14)
    fig.tight_layout()
    plt.show()




def run_grid_experiment_lean(probes_setup, test_dataset_names, activations_model):
    """
    Runs a grid experiment on the probes specified in the probes_setup list.
    Args:
        probes_setup (list): A list of tuples, each containing a probe type, a train dataset name, and a (optional) configuration dictionary.
        test_dataset_names (list): A list of test dataset names.
        activations_model (str): The model the activations came from.
    """
    # Get the best hyperparameters for each probe if not provided
    ps = probes_setup
    for i in range(len(probes_setup)):
        if len(ps[i]) == 2:
            if ps[i][0] == 'mean':
                best_cfg = ConfigDict.from_json(ps[i][0], ps[i][1].split("_")[0])
                ps[i] = [ps[i][0], ps[i][1], ConfigDict(layer=best_cfg.layer, use_bias=True, normalize=True, C=best_cfg.C)]
            else:
                best_cfg = None
                try:
                    best_cfg = ConfigDict.from_json(ps[i][0], ps[i][1])
                except KeyError:
                    print(f"No best hyperparameters found for {ps[i][0]}, {ps[i][1]} locally, pulling from wandb...")
                    best_cfg = load_best_params_from_search(ps[i][0], ps[i][1], "llama_3b")
                if best_cfg is None:
                    raise ValueError(f"No best hyperparameters found for {ps[i][0]}, {ps[i][1]}")
                ps[i] = [ps[i][0], ps[i][1], ConfigDict(best_cfg)]

    for i in range(len(probes_setup)):
        probe_type = ps[i][0]
        train_dataset_name = ps[i][1]
        cfg = ps[i][2]
        
        # Get train and val datasets
        activations_tensor, attention_mask, labels_tensor = probes.load_hf_activations_and_labels_at_layer(train_dataset_name, cfg.layer, verbose=True)
        if "mean" in probe_type:
            activations_tensor = probes.MeanAggregation()(activations_tensor, attention_mask)
        if "3.5k" in train_dataset_name:
            train_dataset, val_dataset, _ = probes.create_activation_datasets(activations_tensor, labels_tensor, splits=[2500, 500, 0], verbose=True)
        else:
            train_dataset, val_dataset, _ = probes.create_activation_datasets(activations_tensor, labels_tensor, splits=[3500, 500, 0], verbose=True)
        
        # Train the probe
        if probe_type == "attention_torch":
            probe = probes.TorchAttentionProbe(cfg)
        elif probe_type == "mean_torch":
            probe = probes.TorchLinearProbe(cfg)
        elif probe_type == "mean":
            probe = probes.SklearnLogisticProbe(cfg)
        probe.fit(train_dataset, val_dataset, verbose=False)

        for test_dataset_name in test_dataset_names:
            # Get test datasets, needing different layers and types for different probes
            activations_tensor, attention_mask, labels_tensor = probes.load_hf_activations_and_labels_at_layer(test_dataset_name, cfg.layer, verbose=True)
            if probe_type == "mean":
                activations_tensor = probes.MeanAggregation()(activations_tensor, attention_mask)
            if test_dataset_name == "jailbreaks_llama_3b_5k":
                _, _, test_dataset = probes.create_activation_datasets(activations_tensor, labels_tensor, splits=[3500, 500, 1000], verbose=True)
            elif "3.5k" in test_dataset_name:
                _, _, test_dataset = probes.create_activation_datasets(activations_tensor, labels_tensor, splits=[2500, 500, 500], verbose=True)
            else:
                _, _, test_dataset = probes.create_activation_datasets(activations_tensor, labels_tensor, splits=[0, 0, 1000], verbose=True)
            
            # Evaluate the probe
            eval_dict, _, _ = probe.eval(test_dataset)
            
            # Save the results
            if "torch" in probe_type:
                hyperparams = [cfg.layer, cfg.use_bias, cfg.normalize, cfg.lr, cfg.weight_decay]
            elif probe_type == "mean":
                hyperparams = [cfg.layer, cfg.use_bias, cfg.normalize, cfg.C]
            probes.wandb_interface.save_probe_dict_results(
                eval_dict=eval_dict, 
                train_set_name=train_dataset_name,
                test_set_name=test_dataset_name,
                activations_model=activations_model,
                probe_type=probe_type,
                hyperparams=hyperparams,
            )


def plot_grid_experiment_lean(probes_setup, test_dataset_names, activations_model, metric="roc_auc"):
    # Get the best hyperparameters for each probe if not provided    
    ps = probes_setup
    for i in range(len(probes_setup)):
        if len(ps[i]) == 2:
            if ps[i][0] == 'mean':
                best_cfg = ConfigDict.from_json(ps[i][0], ps[i][1].split("_")[0])
                print(best_cfg)
                ps[i] = [ps[i][0], ps[i][1], ConfigDict(layer=best_cfg.layer, use_bias=True, normalize=True, C=best_cfg.C)]
            else:
                best_cfg = None
                try:
                    best_cfg = ConfigDict.from_json(ps[i][0], ps[i][1])
                except KeyError:
                    print(f"No best hyperparameters found for {ps[i][0]}, {ps[i][1]} locally, pulling from wandb...")
                    best_cfg = load_best_params_from_search(ps[i][0], ps[i][1], "llama_3b")
                if best_cfg is None:
                    raise ValueError(f"No best hyperparameters found for {ps[i][0]}, {ps[i][1]}")
                ps[i] = [ps[i][0], ps[i][1], ConfigDict(best_cfg)]
    
    # Get all results by querying wandb for all run configs
    results_table = np.full((len(probes_setup), len(test_dataset_names)), -1, dtype=float)
    for i in range(len(probes_setup)):
        probe_type = ps[i][0]
        train_dataset_name = ps[i][1]
        cfg = ps[i][2]
        for j in range(len(test_dataset_names)):
            search_dict = {
                "config.probe/type": probe_type,
                "config.train_dataset": train_dataset_name,
                "config.test_dataset": test_dataset_names[j],
                "config.layer": cfg.layer,
                "config.probe/use_bias": cfg.use_bias,
                "config.probe/normalize": cfg.normalize,
                "config.activations_model": activations_model,
                "state": "finished",  # Only completed runs
            }
            if "torch" in probe_type:
                search_dict["config.probe/lr"] = cfg.lr
                search_dict["config.probe/weight_decay"] = cfg.weight_decay
            elif probe_type == "mean":
                search_dict["config.probe/C"] = cfg.C
            results = load_probe_eval_dict_by_dict(search_dict)
            results_table[i, j] = results[metric]
            # print(f"{train_dataset_name}, {test_dataset_names[j]}, {results[metric]}")

    # Get tick labels
    train_labels = [ps[i][1] for i in range(len(ps))]
    for i in range(len(train_labels)):
        train_labels[i] = train_labels[i].split("_")[1:-1]
        train_labels[i] = "_".join(train_labels[i])[:30]
    test_labels = test_dataset_names
    for i in range(len(test_labels)):
        test_labels[i] = test_labels[i].split("_")[1:-1]
        test_labels[i] = "_".join(test_labels[i])

    # Create the heatmap with seaborn
    fig, ax = plt.subplots()
    sns.heatmap(
        results_table,
        xticklabels=test_labels,
        yticklabels=train_labels,
        annot=True,  # This adds the text annotations
        fmt=".3f",  # Format numbers to 3 decimal places
        cmap="Greens",  # You can change the colormap
        vmin=0.5,
        vmax=1,
        ax=ax,
        annot_kws={"size": 12}, # change this to 12 for 6x6 grids
    )

    # Rotate x-axis labels
    # plt.xticks(rotation=45, ha="right", rotation_mode="anchor")

    # Set labels and title
    plt.xlabel("Test set")
    plt.ylabel("Train set")
    ax.set_title(f"{metric}")

    fig.tight_layout()
    plt.show()


def run_grid_experiment(
    train_dataset_names,
    test_dataset_names,
    layer_list,
    use_bias_list,
    normalize_list,
    C_list,
    activations_model,
    probe_type
):
    train_datasets = {}
    val_datasets = {}
    test_datasets = {}

    for dataset_name in train_dataset_names:
        for layer in layer_list:
            if f"{dataset_name}_{layer}" not in train_datasets:
                activations_tensor, attention_mask, labels_tensor = probes.load_hf_activations_and_labels_at_layer(dataset_name, layer, verbose=True)
                if "mean" in probe_type:
                    activations_tensor = probes.MeanAggregation()(activations_tensor, attention_mask)
                train_dataset, val_dataset, _ = probes.create_activation_datasets(activations_tensor, labels_tensor, splits=[3500, 500, 0], verbose=True)

                train_datasets[f"{dataset_name}_{layer}"] = train_dataset
                val_datasets[f"{dataset_name}_{layer}"] = val_dataset
    
    for dataset_name in test_dataset_names:
        for layer in layer_list:
            if f"{dataset_name}_{layer}" not in test_datasets:
                activations_tensor, attention_mask, labels_tensor = probes.load_hf_activations_and_labels_at_layer(dataset_name, layer, verbose=True)
                if "mean" in probe_type:
                    activations_tensor = probes.MeanAggregation()(activations_tensor, attention_mask)
                _, _, test_dataset = probes.create_activation_datasets(activations_tensor, labels_tensor, splits=[0, 0, 1000], verbose=True)
                test_datasets[f"{dataset_name}_{layer}"] = test_dataset


    for train_index in range(len(train_dataset_names)):
        train_dataset_name = train_dataset_names[train_index]
        # Initialise and fit a probe with the dataset
        probe = probes.SklearnLogisticProbe(ConfigDict(
            use_bias=use_bias_list[train_index],
            normalize=normalize_list[train_index],
            C=C_list[train_index],
        ))
        train_set = train_datasets[f"{train_dataset_name}_{layer_list[train_index]}"]
        val_set = val_datasets[f"{train_dataset_name}_{layer_list[train_index]}"]
        probe.fit(train_set, val_set, verbose=False)

        for test_dataset_name in test_dataset_names:
            test_set = test_datasets[f"{test_dataset_name}_{layer_list[train_index]}"]
            eval_dict, _, _ = probe.eval(test_set)
            probes.wandb_interface.save_probe_dict_results(
                eval_dict=eval_dict, 
                train_set_name=train_dataset_name,
                test_set_name=test_dataset_name,
                activations_model=activations_model,
                probe_type="mean",
                hyperparams=[layer, use_bias_list[train_index], normalize_list[train_index], C_list[train_index]],
            )


def plot_grid_experiment(
    train_dataset_names,
    test_dataset_names,
    train_tick_labels,
    test_tick_labels,
    layer_list,
    use_bias_list,
    normalize_list,
    C_list,
    activations_model,
    metric,
):
    """
    Plots a grid showing a metric for probes trained and tested on each of the specified datasets in a grid.
    Args:
        dataset_list (array): A list of all of the dataset names (as stored on wandb) to form the rows and columns of the grid.
        metric (str): The metric to plot in each cell of the grid (e.g. 'accuracy', 'roc_auc').
    """
    results_table = np.full((len(train_dataset_names), len(test_dataset_names)), -1, dtype=float)
    for train_index in range(len(train_dataset_names)):
        for test_index in range(len(test_dataset_names)):
            results = load_probe_eval_dict_by_dict(
                {
                    "config.train_dataset": train_dataset_names[train_index],
                    "config.test_dataset": test_dataset_names[test_index],
                    "config.layer": layer_list[train_index],
                    "config.probe/type": "mean",
                    "config.probe/use_bias": use_bias_list[train_index],
                    "config.probe/normalize": normalize_list[train_index],
                    "config.probe/C": C_list[train_index],
                    "config.activations_model": activations_model,
                    "state": "finished",  # Only completed runs
                }
            )
            results_table[train_index, test_index] = results[metric]
            print(
                f"{train_dataset_names[train_index]}, {test_dataset_names[test_index]}, {results[metric]}"
            )

    fig, ax = plt.subplots()

    # Create the heatmap with seaborn
    sns.heatmap(
        results_table,
        xticklabels=test_tick_labels,
        yticklabels=train_tick_labels,
        annot=True,  # This adds the text annotations
        fmt=".3f",  # Format numbers to 3 decimal places
        cmap="Greens",  # You can change the colormap
        vmin=0.5,
        vmax=1,
        ax=ax,
        annot_kws={"size": 20}, # change this to 12 for 6x6 grids
    )

    # Rotate x-axis labels
    # plt.xticks(rotation=45, ha="right", rotation_mode="anchor")

    # Set labels and title
    plt.xlabel("Test set")
    plt.ylabel("Train set")
    ax.set_title(f"{metric}")

    fig.tight_layout()
    plt.show()





## SUMMARY GRAPHS
def plot_grid_experiment_lean_with_means(probes_setup, test_dataset_names, activations_model, min_metric=None, max_metric=None, metric="roc_auc", behaviour="", save=False):
    # === Step 1: Preprocessing Setup (EXACT COPY from plot_grid_experiment_lean) ===
    ps = probes_setup
    for i in range(len(probes_setup)):
        if len(ps[i]) == 2:
            if ps[i][0] == 'mean':
                best_cfg = ConfigDict.from_json(ps[i][0], ps[i][1].split("_")[0])
                ps[i] = [ps[i][0], ps[i][1], ConfigDict(layer=best_cfg.layer, use_bias=True, normalize=True, C=best_cfg.C)]
            else:
                best_cfg = None
                try:
                    best_cfg = ConfigDict.from_json(ps[i][0], ps[i][1])
                except KeyError:
                    print(f"No best hyperparameters found for {ps[i][0]}, {ps[i][1]} locally, pulling from wandb...")
                    best_cfg = load_best_params_from_search(ps[i][0], ps[i][1], "llama_3b")
                if best_cfg is None:
                    raise ValueError(f"No best hyperparameters found for {ps[i][0]}, {ps[i][1]}")
                ps[i] = [ps[i][0], ps[i][1], ConfigDict(best_cfg)]

    # === Step 2: Collect Result Table ===
    results_table = np.full((len(probes_setup), len(test_dataset_names)), -1, dtype=float)
    for i in range(len(probes_setup)):
        probe_type = ps[i][0]
        train_dataset_name = ps[i][1]
        cfg = ps[i][2]
        for j in range(len(test_dataset_names)):
            search_dict = {
                "config.probe/type": probe_type,
                "config.train_dataset": train_dataset_name,
                "config.test_dataset": test_dataset_names[j],
                "config.layer": cfg.layer,
                "config.probe/use_bias": cfg.use_bias,
                "config.probe/normalize": cfg.normalize,
                "config.activations_model": activations_model,
                "state": "finished",
            }
            if "torch" in probe_type:
                search_dict["config.probe/lr"] = cfg.lr
                search_dict["config.probe/weight_decay"] = cfg.weight_decay
            elif probe_type == "mean":
                search_dict["config.probe/C"] = cfg.C
            results = load_probe_eval_dict_by_dict(search_dict)
            results_table[i, j] = results[metric]

    # === Step 3: Label Processing ===
    def abridge(label):
        # You can modify this logic as needed
        parts = label.split("_")
        return "".join([p[0] for p in parts if p])  # e.g., llama_3b → l3b

    train_full_labels = ["_".join(ps[i][1].split("_")[1:-1]) for i in range(len(ps))]
    test_full_labels = ["_".join(name.split("_")[1:-1]) for name in test_dataset_names]
    train_short_labels = [abridge(lbl) for lbl in train_full_labels]
    test_short_labels = [abridge(lbl) for lbl in test_full_labels]

    # === Step 4: Add Row and Column Means ===
    row_means = np.mean(results_table, axis=1, keepdims=True)
    col_means = np.mean(results_table, axis=0, keepdims=True)
    full_table = np.block([
        [results_table, row_means],
        [col_means, np.array([[np.nan]])],
    ])

    # === Step 5: Create Mask for bottom-right NaN ===
    mask = np.isnan(full_table)

    # === Step 6: Heatmap ===
    fig, ax = plt.subplots(figsize=(8, 6))

    # Use min/max of valid entries if not provided
    valid_values = results_table[results_table != -1]

    min_metric = min_metric if min_metric is not None else (np.min(valid_values) if valid_values.size > 0 else 0)
    max_metric = max_metric if max_metric is not None else (np.max(valid_values) if valid_values.size > 0 else 1)

    sns.heatmap(
        full_table,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="Greens",
        vmin=min_metric,
        vmax=max_metric,
        cbar=True,
        ax=ax,
        linewidths=0,  # no grid between normal cells
        linecolor='white',
        annot_kws={"size": 12},
        xticklabels=test_short_labels + ["Mean"],
        yticklabels=train_short_labels + ["Mean"],
    )

    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    # === Step 7: Draw separating lines between main grid and means ===
    n_rows, n_cols = results_table.shape
    ax.axhline(n_rows, color='white', linewidth=2)
    ax.axvline(n_cols, color='white', linewidth=2)

    # === Step 8: Legend for abbreviations ===
    legend_elements = []
    for short, full in zip(test_short_labels, test_full_labels):
        legend_elements.append(Patch(facecolor='none', edgecolor='none', label=rf"$\mathbf{{{short}}}$: {full}"))
    for short, full in zip(train_short_labels, train_full_labels):
        legend_elements.append(Patch(facecolor='none', edgecolor='none', label=rf"$\mathbf{{{short}}}$: {full}"))

    ax.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1.15, 0.5),
        title="",
        frameon=False
    )

    ax.set_xlabel("Test", fontsize=12, fontweight="bold")
    ax.set_ylabel("Train", fontsize=12, fontweight="bold")
    ax.set_title(f"{behaviour}, {metric}", fontsize=14, fontweight="bold")

    fig.tight_layout()

    if save:
        save_path = data.figures / behaviour / f"{behaviour}_{metric}_heatmap.png"
        plt.savefig(save_path, dpi=300)
        plt.savefig(save_path.path.with_suffix(".pdf"), dpi=300)
    plt.show()