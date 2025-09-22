import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from probe_gen.config import ConfigDict
from probe_gen.probes.wandb_interface import load_probe_eval_dict_by_dict

datasets = {
    "refusal": {
        "label": "Refusal",
        "on_policy_train":     "refusal_llama_3b_5k",
        "incentivised_train":  "refusal_llama_3b_incentivised_5k",
        "prompted_train":      "refusal_llama_3b_prompted_5k",
        "off_policy_train":    "refusal_ministral_8b_5k",
        "on_policy_test":      "refusal_llama_3b_1k",
        "on_policy_OOD_test":  "jailbreaks_llama_3b_1k"
    },
    "lists": {
        "label": "Lists",
        "on_policy_train":     "lists_llama_3b_5k",
        "incentivised_train":  "lists_llama_3b_incentivised_5k",
        "prompted_train":      "lists_llama_3b_prompted_5k",
        "off_policy_train":    "lists_qwen_3b_5k",
        "on_policy_test":      "lists_llama_3b_1k",
        "on_policy_OOD_test":  "lists_shakespeare_llama_3b_1k"
    },
    "metaphors": {
        "label": "Metaphors",
        "on_policy_train":     "metaphors_llama_3b_5k",
        "incentivised_train":  "metaphors_llama_3b_incentivised_5k",
        "prompted_train":      "metaphors_llama_3b_prompted_5k",
        "off_policy_train":    "metaphors_qwen_3b_5k",
        "on_policy_test":      "metaphors_llama_3b_1k",
        "on_policy_OOD_test":  "metaphors_shakespeare_llama_3b_1k"
    },
    "science": {
        "label": "Science",
        "on_policy_train":     "science_llama_3b_5k",
        "incentivised_train":  "science_llama_3b_incentivised_5k",
        "prompted_train":      "science_llama_3b_prompted_5k",
        "off_policy_train":    "science_qwen_3b_5k",
        "on_policy_test":      "science_llama_3b_1k",
        "on_policy_OOD_test":  "science_llama_3b_ood_test_1k"
    },
    "sycophancy_short": {
        "label": "Sycophancy Short Multiple Choice",
        "on_policy_train":     "sycophancy_short_llama_3b_4k",
        "incentivised_train":  None,
        "prompted_train":      "sycophancy_short_llama_3b_prompted_4k",
        "off_policy_train":    "sycophancy_short_qwen_3b_4k",
        "on_policy_test":      "sycophancy_short_llama_3b_1k"
    },
    "sycophancy": {
        "label": "Sycophancy Multiple Choice",
        "on_policy_train":     "sycophancy_llama_3b_4k",
        "incentivised_train":  "sycophancy_llama_3b_incentivised_4k",
        "prompted_train":      "sycophancy_llama_3b_prompted_4k",
        "off_policy_train":    "sycophancy_ministral_8b_4k",
        "on_policy_test":      "sycophancy_llama_3b_1k",
        "on_policy_OOD_test":  "sycophancy_arguments_llama_3b_1k"
    },
    "sycophancy_arguments": {
        "label": "Sycophancy Arguments",
        "on_policy_train":     "sycophancy_arguments_llama_3b_4k",
        "incentivised_train":  "sycophancy_arguments_llama_3b_incentivised_4k",
        "prompted_train":      "sycophancy_arguments_llama_3b_prompted_4k",
        "off_policy_train":    "sycophancy_arguments_qwen_7b_4k",
        "on_policy_test":      "sycophancy_arguments_llama_3b_1k",
        "on_policy_OOD_test":  "sycophancy_llama_3b_1k"
    },
    "authority": {  
        "label": "Deferral to Authority Multiple Choice",
        "on_policy_train":     "authority_llama_3b_4k",
        "incentivised_train":  "authority_llama_3b_incentivised_4k",
        "prompted_train":      "authority_llama_3b_prompted_4k",
        "off_policy_train":    "authority_ministral_8b_4k",
        "on_policy_test":      "authority_llama_3b_1k",
        "on_policy_OOD_test":  "authority_arguments_llama_3b_1k"
    },
    "authority_arguments": {
        "label": "Deferral to Authority Arguments",
        "on_policy_train":     "authority_arguments_llama_3b_4k",
        "incentivised_train":  "authority_arguments_llama_3b_incentivised_4k",
        "prompted_train":      "authority_arguments_llama_3b_prompted_4k",
        "off_policy_train":    "authority_arguments_qwen_7b_4k",
        "on_policy_test":      "authority_arguments_llama_3b_1k",
        "on_policy_OOD_test":  "authority_llama_3b_1k"
    },
    "deception": {
        "label": "Deception Insider Trading",
        "on_policy_train":     None,
        "incentivised_train":  "deception_llama_3b_3.5k",
        "prompted_train":      "deception_llama_3b_prompted_3.5k",
        "off_policy_train":    "deception_deepseek_mixtral_3.5k",
        "on_policy_test":      None,
    },
    "deception_rp": {
        "label": "Deception Roleplaying",
        "on_policy_train":     None,
        "incentivised_train":  "deception_rp_llama_3b_3.5k",
        "prompted_train":      "deception_rp_llama_3b_prompted_3.5k",
        "off_policy_train":    "deception_rp_mistral_7b_3.5k",
        "on_policy_test":      None,
    },
    "sandbagging": {
        "label": "Sandbagging",
        "on_policy_train":     None,
        "incentivised_train":  "sandbagging_llama_3b_3.5k",
        "prompted_train":      "sandbagging_llama_3b_prompted_3.5k",
        "off_policy_train":    "sandbagging_mistral_7b_3.5k",
        "on_policy_test":      None,
    },
}



def _get_metric_for(train_dataset_name, test_dataset_name):

    best_cfg = ConfigDict.from_json('mean', train_dataset_name)
    best_cfg = ConfigDict(layer=best_cfg.layer, use_bias=True, normalize=True, C=best_cfg.C)

    search_dict = {
        "config.probe/type": 'mean',
        "config.train_dataset": train_dataset_name,
        "config.test_dataset": test_dataset_name,
        "config.layer": best_cfg.layer,
        "config.probe/use_bias": best_cfg.use_bias,
        "config.probe/normalize": best_cfg.normalize,
        "config.probe/C": best_cfg.C,
        "config.activations_model": 'llama_3b',
        "state": "finished",  # Only completed runs
    }

    results = load_probe_eval_dict_by_dict(search_dict)
    return results['roc_auc']


def plot_behaviour_barchart(behaviours, include_ood=False, add_mean_summary=True):

    small_gap = 0.2
    big_gap = 0.5

    # Get all results by querying wandb for all run configs
    results_table = np.full((4 if not include_ood else 8, len(behaviours)), 0, dtype=float)
    behaviour_labels = []
    for i in range(len(behaviours)):
        print(f"Fetching results for {behaviours[i]}")
        behaviour_sets = datasets[behaviours[i]]
        behaviour_labels.append(behaviour_sets['label'])

        if behaviour_sets['on_policy_train'] is not None:
            results_table[0, i] = _get_metric_for(behaviour_sets['on_policy_train'], behaviour_sets['on_policy_test'])
            if include_ood:
                results_table[4, i] = _get_metric_for(behaviour_sets['on_policy_train'], behaviour_sets['on_policy_OOD_test'])
        if behaviour_sets['incentivised_train'] is not None:
            results_table[1, i] = _get_metric_for(behaviour_sets['incentivised_train'], behaviour_sets['on_policy_test'])
            if include_ood:
                results_table[5, i] = _get_metric_for(behaviour_sets['incentivised_train'], behaviour_sets['on_policy_OOD_test'])
        if behaviour_sets['prompted_train'] is not None:
            results_table[2, i] = _get_metric_for(behaviour_sets['prompted_train'], behaviour_sets['on_policy_test'])
            if include_ood:
                results_table[6, i] = _get_metric_for(behaviour_sets['prompted_train'], behaviour_sets['on_policy_OOD_test'])
        if behaviour_sets['off_policy_train'] is not None:
            results_table[3, i] = _get_metric_for(behaviour_sets['off_policy_train'], behaviour_sets['on_policy_test'])
            if include_ood:
                results_table[7, i] = _get_metric_for(behaviour_sets['off_policy_train'], behaviour_sets['on_policy_OOD_test'])

    x = np.arange(len(behaviours))  # Positions 0, 1, 2, ..., 8

    if add_mean_summary:
        behaviour_labels.append('Mean (+std)')

    masked_array = np.ma.masked_equal(results_table, 0)
    row_means = np.ma.mean(masked_array, axis=1)
    row_stds = np.ma.std(masked_array, axis=1)
        # results_table = np.column_stack([results_table, row_means])
        # behaviour_labels.append('mean')
        # x = np.concatenate([x, [len(behaviours) + big_gap - small_gap]])

    group_labels = ['On policy', 'Incentivised', 'Prompted', 'Off policy']

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))  # Made slightly wider to accommodate gap

    train_colors = ['#264653', '#2A9D8F', '#E76F51', '#F18F01'] 
    patterns = ["", "", "", "", "xx", "xx", "xx", "xx"]


    # Create the grouped bars - separate first groups from mean group
    num_groups = results_table.shape[0]
    bar_width = (1 - small_gap) / num_groups
    for i in range(num_groups):
        group_offset = (i - num_groups / 2 + 0.5) * bar_width

        ax.bar(x + group_offset, results_table[i], bar_width, label='on', color=train_colors[i % 4], alpha=0.8, hatch=patterns[i])
        
        if add_mean_summary:
            ax.bar(np.array([len(behaviours) + big_gap - small_gap]) + group_offset, row_means[i], bar_width, color=train_colors[i % 4], alpha=0.8, 
                     yerr=row_stds[i], capsize=5, error_kw={'elinewidth': 2, 'capthick': 2}, hatch=patterns[i])


    # # First groups (without error bars)
    # bars1_first = ax.bar(x_first - 1.5 * width, on_values, width, label='on', color=colors[0], alpha=0.8)
    # bars2_first = ax.bar(x_first - 0.5 * width, incentivised_values, width, label='incentivised', color=colors[1], alpha=0.8)
    # bars3_first = ax.bar(x_first + 0.5 * width, prompted_values, width, label='prompted', color=colors[2], alpha=0.8)  
    # bars4_first = ax.bar(x_first + 1.5 * width, off_values, width, label='off', color=colors[3], alpha=0.8)

    # # Mean group (with error bars)
    # bars1_mean = ax.bar(x_last - 1.5 * width, row_means[0], width, color=colors[0], alpha=0.8, 
    #                 yerr=row_stds[0], capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
    # bars2_mean = ax.bar(x_last - 0.5 * width, row_means[1], width, color=colors[1], alpha=0.8,
    #                 yerr=row_stds[1], capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
    # bars2_mean = ax.bar(x_last + 0.5 * width, row_means[2], width, color=colors[2], alpha=0.8,
    #                 yerr=row_stds[2], capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
    # bars3_mean = ax.bar(x_last + 1.5 * width, row_means[3], width, color=colors[3], alpha=0.8,
    #                 yerr=row_stds[3], capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})

    # Customize the plot
    ax.set_xlabel('Behaviour')
    ax.set_ylabel('Test AUROC')
    ax.set_title('Generalization to On-Policy Data per Behaviour')
    ax.set_xticks(np.concatenate([x, [len(behaviours) + big_gap - small_gap]]))
    import textwrap

    wrapped_labels = ['\n'.join(textwrap.wrap(label, width=10)) for label in behaviour_labels]
    ax.set_xticklabels(wrapped_labels)

    color_patches = [
        mpatches.Patch(color=train_colors[0], label='On Policy'),
        mpatches.Patch(color=train_colors[1], label='On Policy - Incentivised'),
        mpatches.Patch(color=train_colors[2], label='On Policy - Prompted'),
        mpatches.Patch(color=train_colors[3], label='Off Policy')
    ]

    # Create hatch legend elements (using white color to show only hatches)
    hatch_patches = [
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='', label='On Policy - ID'),
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='xx', label='On Policy - OOD')
    ]

    # Create the first legend (colors) and add it to the plot
    color_legend = ax.legend(handles=color_patches, title='Train Set', 
                            loc='lower left', bbox_to_anchor=(0.02, 0.02), framealpha=1.0)

    # Add the first legend back to the plot (important step!)
    ax.add_artist(color_legend)

    # Create the second legend (hatches)
    hatch_legend = ax.legend(handles=hatch_patches, title='Test Set', 
                            loc='lower right', bbox_to_anchor=(0.98, 0.02), framealpha=1.0)

    # Add a grid for better readability
    ax.grid(True, alpha=0.3, axis='y')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()