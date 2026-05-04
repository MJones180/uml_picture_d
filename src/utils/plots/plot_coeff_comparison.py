import matplotlib.pyplot as plt
import numpy as np
from utils.constants import PLOT_STYLE_FILE


def plot_coeff_comparison(
    upper_coeff_group_idxs,
    metric_one_data,
    metric_two_data,
    metric_one_label,
    metric_two_label,
    plot_path,
    metric_one_color='tab:blue',
    metric_two_color='tab:red',
):
    # Load in the style file
    plt.style.use(PLOT_STYLE_FILE)
    # Reset the plot
    plt.clf()
    numb_groups = len(upper_coeff_group_idxs)
    coeff_group_idxs = [0, *upper_coeff_group_idxs]
    fig, axs = plt.subplots(
        numb_groups,
        1,
        figsize=(12, 10),
        constrained_layout=True,
    )
    for idx in range(numb_groups):
        lower_bound = coeff_group_idxs[idx]
        upper_bound = coeff_group_idxs[idx + 1]
        indices = np.arange(lower_bound, upper_bound)
        axs[idx].set_title(f'Coefficient Errors ({lower_bound}-{upper_bound})')
        axs[idx].set_xlabel('Coefficient Index')
        axs[idx].set_ylabel(metric_one_label, color=metric_one_color)
        axs[idx].plot(
            indices,
            metric_one_data[lower_bound:upper_bound],
            color=metric_one_color,
            alpha=0.75,
        )
        axs[idx].tick_params(axis='y', labelcolor=metric_one_color)
        axs[idx].set_ylim(0, np.max(metric_one_data) * 1.1)
        ax_twin = axs[idx].twinx()
        ax_twin.set_ylabel(metric_two_label, color=metric_two_color)
        ax_twin.plot(
            indices,
            metric_two_data[lower_bound:upper_bound],
            color=metric_two_color,
            alpha=0.75,
        )
        ax_twin.tick_params(axis='y', labelcolor=metric_two_color)
        ax_twin.set_ylim(0, np.max(metric_two_data) * 1.1)
    plt.savefig(plot_path)
