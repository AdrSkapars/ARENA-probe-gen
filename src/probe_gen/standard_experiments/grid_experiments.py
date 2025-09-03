import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from probe_gen.config import ConfigDict
import probe_gen.probes as probes
from probe_gen.probes.wandb_interface import load_probe_eval_dict_by_dict
from probe_gen.standard_experiments.hyperparameter_search import load_best_params_from_search


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
        if probe_type == "mean":
            activations_tensor = probes.MeanAggregation()(activations_tensor, attention_mask)
        train_dataset, val_dataset, _ = probes.create_activation_datasets(activations_tensor, labels_tensor, splits=[3500, 500, 0], verbose=True)
        
        # Train the probe
        if "torch" in probe_type:
            probe = probes.TorchLinearProbe(cfg)
        elif probe_type == "mean":
            probe = probes.SklearnLogisticProbe(cfg)
        probe.fit(train_dataset, val_dataset)

        for test_dataset_name in test_dataset_names:
            # Get test datasets, needing different layers and types for different probes
            activations_tensor, attention_mask, labels_tensor = probes.load_hf_activations_and_labels_at_layer(test_dataset_name, cfg.layer, verbose=True)
            if probe_type == "mean":
                activations_tensor = probes.MeanAggregation()(activations_tensor, attention_mask)
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
            try:
                best_cfg = ConfigDict.from_json(ps[i][0], ps[i][1])
            except KeyError:
                print(f"No best hyperparameters found for {ps[i][0]}, {ps[i][1]} locally, pulling from wandb...")
                best_cfg = load_best_params_from_search(ps[i][0], ps[i][1], "llama_3b")
            if best_cfg is None:
                raise ValueError(f"No best hyperparameters found for {ps[i][0]}, {ps[i][1]}")
        print(best_cfg)
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
        train_labels[i] = "_".join(train_labels[i])
    test_labels = test_dataset_names
    for i in range(len(test_labels)):
        test_labels[i] = test_labels[i].split("_")[1:-1]
        test_labels[i] = "_".join(test_labels[i])

    # Create the heatmap with seaborn
    fig, ax = plt.subplots()
    sns.heatmap(
        results_table,
        xticklabels=train_labels,
        yticklabels=test_labels,
        annot=True,  # This adds the text annotations
        fmt=".3f",  # Format numbers to 3 decimal places
        cmap="Greens",  # You can change the colormap
        vmin=0.5,
        vmax=1,
        ax=ax,
        annot_kws={"size": 20},
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
):
    train_datasets = {}
    val_datasets = {}
    test_datasets = {}

    for dataset_name in train_dataset_names:
        for layer in layer_list:
            if f"{dataset_name}_{layer}" not in train_datasets:
                activations_tensor, attention_mask, labels_tensor = probes.load_hf_activations_and_labels_at_layer(dataset_name, layer, verbose=True)
                activations_tensor = probes.MeanAggregation()(activations_tensor, attention_mask)
                train_dataset, val_dataset, _ = probes.create_activation_datasets(activations_tensor, labels_tensor, splits=[3500, 500, 0], verbose=True)

                train_datasets[f"{dataset_name}_{layer}"] = train_dataset
                val_datasets[f"{dataset_name}_{layer}"] = val_dataset
    
    for dataset_name in test_dataset_names:
        for layer in layer_list:
            if f"{dataset_name}_{layer}" not in test_datasets:
                activations_tensor, attention_mask, labels_tensor = probes.load_hf_activations_and_labels_at_layer(dataset_name, layer, verbose=True)
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
        probe.fit(train_set, val_set)

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
        annot_kws={"size": 20},
    )

    # Rotate x-axis labels
    # plt.xticks(rotation=45, ha="right", rotation_mode="anchor")

    # Set labels and title
    plt.xlabel("Test set")
    plt.ylabel("Train set")
    ax.set_title(f"{metric}")

    fig.tight_layout()
    plt.show()
