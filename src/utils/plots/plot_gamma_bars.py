import matplotlib.pyplot as plt
import numpy as np
from utils.constants import PLOT_STYLE_FILE


def plot_gamma_bars(gamma_magnitudes, plot_path, multi_headed_depths=[]):
    # Load in the style file
    plt.style.use(PLOT_STYLE_FILE)
    # Reset the plot
    plt.clf()
    fig, ax = plt.subplots(figsize=(14, 6))
    total_layers = len(gamma_magnitudes)
    x_vals = np.arange(total_layers)
    if len(multi_headed_depths) > 0:
        # Needed to properly align each groups label
        x_label_offset = 0
        x_tick_locs = []
        x_tick_labs = []
        colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:cyan', 'tab:orange']
        for head_idx, head_depth in enumerate(multi_headed_depths):
            head_x_vals = x_vals[:head_depth]
            x_vals = x_vals[head_depth:]
            head_y_vals = gamma_magnitudes[:head_depth]
            gamma_magnitudes = gamma_magnitudes[head_depth:]
            ax.bar(
                head_x_vals,
                head_y_vals,
                color=colors[head_idx],
                label=f'Head {head_idx + 1}',
                linewidth=0.5,
            )
            block_width = head_depth / total_layers
            ax.text(
                x_label_offset + block_width / 2,
                -0.1,
                f'Head {head_idx + 1} Layers',
                transform=ax.transAxes,
                ha='center',
                va='top',
                color=colors[head_idx],
            )
            x_label_offset += block_width
            x_tick_locs.extend([head_x_vals[0], *head_x_vals[4::5]])
            x_tick_lab_vals = list(range(1, head_depth + 1))
            x_tick_labs.extend([x_tick_lab_vals[0], *x_tick_lab_vals[4::5]])
        plt.legend()
    else:
        ax.bar(x_vals, gamma_magnitudes, linewidth=0.5)
        x_tick_locs = [x_vals[0], *x_vals[4::5]]
        x_tick_lab_vals = x_vals + 1
        x_tick_labs = [x_tick_lab_vals[0], *x_tick_lab_vals[4::5]]
        ax.set_xlabel('Layers')
    ax.set_xticks(x_tick_locs)
    ax.set_xticklabels(x_tick_labs, rotation=90)
    plt.margins(0.01, 0.05)
    plt.title(r'Mean $\gamma$ Magnitude per Layer')
    plt.ylabel(r'Mean $\gamma$ Magnitude')
    plt.savefig(plot_path)
