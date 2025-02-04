import matplotlib.pyplot as plt
import mpl_scatter_density  # noqa: F401; 'scatter_density' projection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from utils.idl_rainbow_cmap import idl_rainbow_cmap
from utils.terminate_with_message import terminate_with_message


def plot_comparison_scatter_grid(
    model_data,
    truth_data,
    n_rows,
    n_cols,
    title_vs,
    identifier,
    starting_zernike,
    filter_value,
    plot_path=None,
    plot_density=0,
    interactive_view=False,
):
    """
    A scatter plot for each Zernike term that compares the model's predicted
    value with the true value.

    Parameters
    ----------
    model_data : np.array
        The model data where the second dimension represents the Zernikes.
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
    filter_value : float
        If set to 0, then all subplots will have their own axes limits and no
        filtering will be done. Otherwise, will filter out all values not in the
        range of [-filter_value, filter_value].
    plot_path : str
        The path to output the plot.
    plot_density : int
        If not zero, a plot density scatter will be used. The number passed
        represents the upper bound on the colorbar.
    interactive_view : bool
        Display the plot in interactive mode, False by default.
    """

    # =============================
    # Ensure there are enough cells
    # =============================

    total_row_count, col_count = model_data.shape
    if n_rows * n_cols < col_count:
        terminate_with_message('Not enough rows and columns for the data.')

    # ===============
    # Filter the data
    # ===============

    point_per_plot = total_row_count
    if filter_value:
        gte = model_data >= -filter_value
        lte = model_data <= filter_value
        filter_mask = np.all(gte & lte, axis=1)
        model_data = model_data[filter_mask]
        truth_data = truth_data[filter_mask]
        point_per_plot = model_data.shape[0]
        print(f'Before filtering: {total_row_count} rows')
        print(f'After filtering: {point_per_plot} rows')
        rows_filtered_out = total_row_count - point_per_plot
        print(f'Filtered out {rows_filtered_out} rows')

    # ==================
    # Setup the subplots
    # ==================

    subplot_args = {'figsize': (n_cols * 5, n_rows * 5)}
    if plot_density:
        subplot_args['subplot_kw'] = {'projection': 'scatter_density'}
    if filter_value:
        subplot_args['sharex'] = True
        subplot_args['sharey'] = True
    fig, axs = plt.subplots(n_rows, n_cols, **subplot_args)

    # ===============
    # Do the plotting
    # ===============

    title = (f'Truth vs {title_vs} [{identifier}]\n'
             f'Total points per subplot: {point_per_plot}. ')
    if filter_value:
        title += (f'Filtered out {rows_filtered_out} '
                  f'rows using bounds [-{filter_value}, {filter_value}].')

    plt.suptitle(title, size=30)
    # The limits should be the global min and max values if filtered, otherwise
    # it changes for each subplot
    if filter_value:
        xlim = np.amin(truth_data), np.amax(truth_data)
        ylim = np.amin(model_data), np.amax(model_data)
    current_col = 0
    for plot_row in range(n_rows):
        for plot_col in range(n_cols):
            # Remove axes for unused cells
            if current_col >= col_count:
                fig.delaxes(axs[plot_row, plot_col])
                continue
            # Grab the data for the current cell
            model_col = model_data[:, current_col]
            truth_col = truth_data[:, current_col]
            axs_cell = axs[plot_row, plot_col]
            if filter_value == 0:
                xlim = np.amin(truth_col), np.amax(truth_col)
                ylim = np.amin(model_col), np.amax(model_col)
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
                color='#FFFFFF' if plot_density else '#000000',
                scalex=False,
                scaley=False,
                zorder=1 if plot_density else -1,
            )
            if plot_density:
                density = axs_cell.scatter_density(
                    truth_col,
                    model_col,
                    cmap=idl_rainbow_cmap(),
                    vmin=0,
                    vmax=plot_density,
                )
                # Only show the colorbar for subplots in the right-most column
                if plot_col == n_cols - 1:
                    div = make_axes_locatable(axs_cell)
                    cax = div.append_axes('right', size='5%', pad=0.05)
                    cbar = fig.colorbar(density, cax=cax)
                    cbar.set_label('Points Per Pixel', size=20)
                    cbar.ax.tick_params(labelsize=15)
            else:
                # Plot the scatter of all the points
                axs_cell.scatter(truth_col, model_col, 0.25, alpha=.1)
            # Set the limits on each subplot
            axs_cell.set_xlim(*xlim)
            axs_cell.set_ylim(*ylim)
            # Only display x labels for the last row
            if plot_row == n_rows - 1:
                axs_cell.set_xlabel('Truth Outputs')
                axs_cell.xaxis.label.set_fontsize(20)
            # Only display y labels for the first column
            if plot_col == 0:
                axs_cell.set_ylabel('Model Outputs')
                axs_cell.yaxis.label.set_fontsize(20)
            # Increase the font size for ticks
            for item in (axs_cell.get_xticklabels() +
                         axs_cell.get_yticklabels() + [
                             axs_cell.xaxis.get_offset_text(),
                             axs_cell.yaxis.get_offset_text()
                         ]):
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
