import matplotlib.pyplot as plt
import numpy as np
from utils.terminate_with_message import terminate_with_message


def plot_comparison_scatter_grid(
    pred_data,
    truth_data,
    n_rows,
    n_cols,
    title_vs,
    identifier,
    output_path,
):
    row_count, col_count = pred_data.shape
    if n_rows * n_cols < col_count:
        terminate_with_message('Not enough rows and columns for the data.')
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    plt.suptitle(f'Truth vs {title_vs} [{identifier}]')
    current_col = 0
    for plot_row in range(n_rows):
        for plot_col in range(n_cols):
            if current_col == col_count:
                break
            pred_col = pred_data[:, current_col]
            truth_col = truth_data[:, current_col]
            axs_cell = axs[plot_row, plot_col]
            axs_cell.set_title(current_col)
            # Take the lowest and greatest values from both sets of data
            lower = min(np.amin(pred_col), np.amin(truth_col))
            upper = max(np.amax(pred_col), np.amax(truth_col))
            # Fix the bounds on both axes so they are 1-to-1
            axs_cell.set_xlim(lower, upper)
            axs_cell.set_ylim(lower, upper)
            # Draw a 1-to-1 line for the scatters
            # https://stackoverflow.com/a/60950862
            xpoints = ypoints = axs_cell.get_xlim()
            axs_cell.plot(
                xpoints,
                ypoints,
                linestyle='-',
                linewidth=2,
                color='#FFB200',
                scalex=False,
                scaley=False,
            )
            # Plot the scatter of all the points
            axs_cell.scatter(truth_col, pred_col, 0.25)
            current_col += 1
    for ax in axs.flat:
        ax.set(xlabel='Truth Outputs', ylabel='Pred Outputs')
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
