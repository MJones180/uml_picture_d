# ==============================================================================
# This version was developed to be more clear for a paper.
# ==============================================================================

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import mpl_scatter_density  # noqa: F401; 'scatter_density' projection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from utils.idl_rainbow_cmap import idl_rainbow_cmap
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
density_cmap = idl_rainbow_cmap()


def plot_comparison_scatter_grid_v2(
    pred_data,
    truth_data,
    n_rows,
    n_cols,
    title_vs,
    identifier,
    starting_zernike,
    plot_path=None,
    plot_density=0,
    filter_value=None,
    interactive_view=False,
):
    """
    A scatter plot for each Zernike term that compares the model's predicted
    value with the true value.

    Parameters
    ----------
    pred_data : np.array
        The prediction data where the second dimension represents the Zernikes.
    truth_data : np.array
        The truth data where the second dimension represents the Zernikes.
    n_rows : int
        Number of rows in the plot.
    n_cols : int
        Number of columns in the plot.
    title_vs : str
        Name of the type of model that is being compared against.
    identifier : str
        Identifier of the model being compared against.
    starting_zernike : int
        The first Zernike being compared against.
    plot_path : str
        The path to output the plot.
    plot_density : int
        If not zero, a plot density scatter will be used. The number passed
        represents the upper bound on the colorbar.
    filter_value : float
        If not None, will filter out all values not in the range of
        [-filter_value, filter_value].
    interactive_view : bool
        Display the plot in interactive mode, False by default.
    """

    # =============================
    # Ensure there are enough cells
    # =============================

    total_row_count, col_count = pred_data.shape
    if n_rows * n_cols < col_count:
        terminate_with_message('Not enough rows and columns for the data.')

    # ===============
    # Filter the data
    # ===============

    if filter_value:
        gte = pred_data >= -filter_value
        lte = pred_data <= filter_value
        filter_mask = np.all(gte & lte, axis=1)
        pred_data = pred_data[filter_mask]
        truth_data = truth_data[filter_mask]
        filtered_row_count = pred_data.shape[0]
        print(f'Before filtering: {total_row_count} rows')
        print(f'After filtering: {filtered_row_count} rows')
        rows_filtered_out = total_row_count - filtered_row_count
        print(f'Filtered out {rows_filtered_out} rows')

    # ==================
    # Setup the subplots
    # ==================

    if plot_density:
        subplot_args = {
            'figsize': (n_cols * 5, n_rows * 5),
            'subplot_kw': {
                'projection': 'scatter_density'
            }
        }
    else:
        subplot_args = {'figsize': (n_cols * 3, n_rows * 3)}
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        sharex=True,
        sharey=True,
        **subplot_args,
    )

    # ===============
    # Do the plotting
    # ===============

    title = (f'Truth vs {title_vs} [{identifier}]\n'
             f'Total points per subplot: {filtered_row_count}. '
             f'Filtered out {rows_filtered_out} '
             f'rows using bounds [-{filter_value}, {filter_value}].')
    plt.suptitle(title, size=30)
    xlim = np.amin(truth_data), np.amax(truth_data)
    ylim = np.amin(pred_data), np.amax(pred_data)
    current_col = 0
    for plot_row in range(n_rows):
        for plot_col in range(n_cols):
            # Remove axes for unused cells
            if current_col >= col_count:
                fig.delaxes(axs[plot_row, plot_col])
                continue
            # Grab the data for the current cell
            pred_col = pred_data[:, current_col]
            truth_col = truth_data[:, current_col]
            axs_cell = axs[plot_row, plot_col]
            # Add the Zernike number to the top left of the plot
            # https://stackoverflow.com/a/50091489
            axs_cell.annotate(
                f'Z{starting_zernike + current_col}',
                (0, 1),
                xytext=(8, -8),
                xycoords='axes fraction',
                textcoords='offset points',
                fontweight='bold',
                color='white',
                backgroundcolor='k',
                ha='left',
                va='top',
                fontsize=20,
            )
            # Draw a 1-to-1 line, it should be based on truth values
            # https://stackoverflow.com/a/60950862
            axs_cell.plot(
                xlim,
                xlim,
                linestyle='-',
                linewidth=1,
                color='#FFFFFF' if plot_density else '#FFB200',
                scalex=False,
                scaley=False,
            )
            if plot_density:
                density = axs_cell.scatter_density(
                    truth_col,
                    pred_col,
                    cmap=density_cmap,
                    vmin=0,
                    vmax=plot_density,
                )
                if plot_col == n_cols - 1:
                    divider = make_axes_locatable(axs_cell)
                    caxd = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = fig.colorbar(density, cax=caxd)
                    cbar.set_label('Rows Per Pixel', size=20)
                    cbar.ax.tick_params(labelsize=15)
            else:
                # Plot the scatter of all the points
                axs_cell.scatter(truth_col, pred_col, 0.25)
            # Ensure all subplots have the same limits
            axs_cell.set_xlim(*xlim)
            axs_cell.set_ylim(*ylim)
            # Only display x labels for the last row
            if plot_row == n_rows - 1:
                axs_cell.set_xlabel('Truth Outputs')
                axs_cell.xaxis.label.set_fontsize(20)
            # Only display y labels for the first column
            if plot_col == 0:
                axs_cell.set_ylabel('Pred Outputs')
                axs_cell.yaxis.label.set_fontsize(20)
            # Increase the font size of all axis tick labels
            for item in (axs_cell.get_xticklabels() +
                         axs_cell.get_yticklabels()):
                item.set_fontsize(15)
            current_col += 1
    # Fixes padding around subplots, `h_pad` decreases vertical spacing between
    # rows and `rect` makes room for the suptitle
    fig.tight_layout(h_pad=0, rect=[0, 0, 1, 0.99])

    # =====================
    # Save or show the plot
    # =====================

    if interactive_view:
        plt.show()
    else:
        # Save the plot
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
