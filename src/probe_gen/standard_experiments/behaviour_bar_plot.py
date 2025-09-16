import matplotlib.pyplot as plt
import numpy as np

from probe_gen.config import ConfigDict
from probe_gen.probes.wandb_interface import load_probe_eval_dict_by_dict

datasets = {
    "refusal": {
        "on_policy_train":     "refusal_llama_3b_5k",
        "incentivised_train":  None,
        "prompted_train":      "refusal_llama_3b_prompted_5k",
        "off_policy_train":    "refusal_ministral_8b_5k",
        "on_policy_test":      "refusal_llama_3b_1k"
    },
    "lists": {
        "on_policy_train":     "lists_llama_3b_5k",
        "incentivised_train":  None,
        "prompted_train":      "lists_llama_3b_prompted_5k",
        "off_policy_train":    "lists_qwen_3b_5k",
        "on_policy_test":      "lists_llama_3b_1k"
    },
    "metaphors": {
        "on_policy_train":     "metaphors_llama_3b_5k",
        "incentivised_train":  "metaphors_llama_3b_incentivised_5k",
        "prompted_train":      "metaphors_llama_3b_prompted_5k",
        "off_policy_train":    "metaphors_qwen_3b_5k",
        "on_policy_test":      "metaphors_llama_3b_1k"
    },
    "science": {
        "on_policy_train":     "science_llama_3b_5k",
        "incentivised_train":  None,
        "prompted_train":      "science_llama_3b_prompted_5k",
        "off_policy_train":    "science_qwen_3b_5k",
        "on_policy_test":      "science_llama_3b_1k"
    },
    "sycophancy_short": {
        "on_policy_train":     "sycophancy_short_llama_3b_4k",
        "incentivised_train":  None,
        "prompted_train":      "sycophancy_short_llama_3b_prompted_4k",
        "off_policy_train":    "sycophancy_short_qwen_3b_4k",
        "on_policy_test":      "sycophancy_short_llama_3b_1k"
    },
    "sycophancy": {
        "on_policy_train":     "sycophancy_llama_3b_4k",
        "incentivised_train":  "sycophancy_llama_3b_incentivised_4k",
        "prompted_train":      "sycophancy_llama_3b_prompted_4k",
        "off_policy_train":    "sycophancy_ministral_8b_4k",
        "on_policy_test":      "sycophancy_llama_3b_1k"
    },
    "sycophancy_arguments": {
        "on_policy_train":     "sycophancy_arguments_llama_3b_4k",
        "incentivised_train":  "sycophancy_arguments_llama_3b_incentivised_4k",
        "prompted_train":      "sycophancy_arguments_llama_3b_prompted_4k",
        "off_policy_train":    "sycophancy_arguments_qwen_7b_4k",
        "on_policy_test":      "sycophancy_arguments_llama_3b_1k"
    },
    "authority": {
        "on_policy_train":     "authority_llama_3b_4k",
        "incentivised_train":  "authority_llama_3b_incentivised_4k",
        "prompted_train":      "authority_llama_3b_prompted_4k",
        "off_policy_train":    "authority_ministral_8b_4k",
        "on_policy_test":      "authority_llama_3b_1k"
    },
    "deception": {
        "on_policy_train":     None,
        "incentivised_train":  "deception_llama_3b_3.5k",
        "prompted_train":      "deception_llama_3b_prompted_3.5k",
        "off_policy_train":    "deception_deepseek_mixtral_3.5k",
        "on_policy_test":      "deception_llama_3b_3.5k",
    },
    "deception_rp": {
        "on_policy_train":     None,
        "incentivised_train":  "deception_rp_llama_3b_3.5k",
        "prompted_train":      "deception_rp_llama_3b_prompted_3.5k",
        "off_policy_train":    "deception_rp_mistral_7b_3.5k",
        "on_policy_test":      "deception_rp_llama_3b_3.5k",
    },
    "sandbagging": {
        "on_policy_train":     None,
        "incentivised_train":  "sandbagging_llama_3b_3.5k",
        "prompted_train":      "sandbagging_llama_3b_prompted_3.5k",
        "off_policy_train":    "sandbagging_mistral_7b_3.5k",
        "on_policy_test":      "sandbagging_llama_3b_3.5k",
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


def plot_behaviour_barchart(behaviours, labels, probe_type):

    # Get all results by querying wandb for all run configs
    results_table = np.full((4, len(behaviours)), 0, dtype=float)
    for i in range(len(behaviours)):

        behaviour_sets = datasets[behaviours[i]]
        
        if behaviour_sets['on_policy_train'] is not None:
            results_table[0, i] = _get_metric_for(behaviour_sets['on_policy_train'], behaviour_sets['on_policy_test'])
        if behaviour_sets['incentivised_train'] is not None:
            results_table[1, i] = _get_metric_for(behaviour_sets['incentivised_train'], behaviour_sets['on_policy_test'])
        if behaviour_sets['prompted_train'] is not None:
            results_table[2, i] = _get_metric_for(behaviour_sets['prompted_train'], behaviour_sets['on_policy_test'])
        if behaviour_sets['off_policy_train'] is not None:
            results_table[3, i] = _get_metric_for(behaviour_sets['off_policy_train'], behaviour_sets['on_policy_test'])


    masked_array = np.ma.masked_equal(results_table, 0)
    row_means = np.ma.mean(masked_array, axis=1)
    row_stds = np.ma.std(masked_array, axis=1)


    # Extract the three rows
    on_values = results_table[0]
    incentivised_values = results_table[1] 
    prompted_values = results_table[2] 
    off_values = results_table[3]

    # Set up the bar positions with gap before the last group
    gap_size = 0.5  # Size of the larger gap

    # Create positions: first 9 groups normally spaced, then gap, then last group
    x_first = np.arange(len(behaviours))  # Positions 0, 1, 2, ..., 8
    x_last = np.array([len(behaviours) + gap_size])  # Position 8 + gap
    x = np.concatenate([x_first, x_last])

    width = 0.2      # Width of each bar

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))  # Made slightly wider to accommodate gap

    colors = ['#264653', '#2A9D8F', '#E76F51', '#F18F01'] 

    # Create the grouped bars - separate first groups from mean group
    # First groups (without error bars)
    bars1_first = ax.bar(x_first - width, on_values, width, label='on', color=colors[0], alpha=0.8)
    bars2_first = ax.bar(x_first, incentivised_values, width, label='incentivised', color=colors[1], alpha=0.8)
    bars3_first = ax.bar(x_first + width, prompted_values, width, label='prompted', color=colors[2], alpha=0.8)  
    bars4_first = ax.bar(x_first + 2 * width, off_values, width, label='off', color=colors[3], alpha=0.8)

    # Mean group (with error bars)
    bars1_mean = ax.bar(x_last - width, row_means[0], width, color=colors[0], alpha=0.8, 
                    yerr=row_stds[0], capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
    bars2_mean = ax.bar(x_last, row_means[1], width, color=colors[1], alpha=0.8,
                    yerr=row_stds[1], capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
    bars2_mean = ax.bar(x_last + width, row_means[2], width, color=colors[2], alpha=0.8,
                    yerr=row_stds[2], capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
    bars3_mean = ax.bar(x_last + 2 * width, row_means[3], width, color=colors[3], alpha=0.8,
                    yerr=row_stds[3], capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})

    # Customize the plot
    ax.set_xlabel('Behaviour')
    ax.set_ylabel('Test AUROC')
    ax.set_title('Generalization to On-Policy Data per Behaviour')
    ax.set_xticks(x)
    import textwrap

    wrapped_labels = ['\n'.join(textwrap.wrap(label, width=10)) for label in labels + ['mean']]
    ax.set_xticklabels(wrapped_labels)
    ax.legend(loc='lower right')

    # Add a grid for better readability
    ax.grid(True, alpha=0.3, axis='y')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()