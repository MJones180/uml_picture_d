from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import mpl_scatter_density  # noqa: F401; 'scatter_density' projection
import numpy as np
from utils.terminate_with_message import terminate_with_message

# https://stackoverflow.com/a/64105308
density_cmap = LinearSegmentedColormap.from_list(
    'density_cmap',
    [
        (0, '#000000'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
    ],
    N=256,
)


def plot_comparison_scatter_grid(
    pred_data,
    truth_data,
    n_rows,
    n_cols,
    title_vs,
    identifier,
    output_path,
    plot_density=False,
):
    row_count, col_count = pred_data.shape
    if n_rows * n_cols < col_count:
        terminate_with_message('Not enough rows and columns for the data.')
    subplot_args = {'figsize': (n_cols * 3, n_rows * 3)}
    if plot_density:
        subplot_args['subplot_kw'] = {'projection': 'scatter_density'}
    fig, axs = plt.subplots(n_rows, n_cols, **subplot_args)
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
                color='#FFFFFF' if plot_density else '#FFB200',
                scalex=False,
                scaley=False,
            )
            if plot_density:
                density = axs_cell.scatter_density(
                    pred_col,
                    truth_col,
                    cmap=density_cmap,
                )
                fig.colorbar(density, label='Retrievals Per Pixel')
            else:
                # Plot the scatter of all the points
                axs_cell.scatter(truth_col, pred_col, 0.25)
            current_col += 1
    for ax in axs.flat:
        ax.set(xlabel='Truth Outputs', ylabel='Pred Outputs')
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
