import matplotlib.pyplot as plt
import numpy as np
from utils.constants import PLOT_STYLE_FILE
from utils.terminate_with_message import terminate_with_message


def plot_control_loop_zernikes_subplots(
    zernike_terms,
    zernike_coeffs,
    title,
    total_time,
    n_rows,
    n_cols,
    plot_path,
    plot_psd=False,
    extra_zernike_coeffs=None,
    legend_labels=None,
    add_std=False,
):
    """
    Generates a grid of subplots where each subplot shows the outputted Zernikes
    from a control loop run. Each subplot either shows the time series or PSD.
    This code was adapted from the `plot_control_loop_zernikes` and
    `plot_comparison_scatter_grid` plotting functions.

    Parameters
    ----------
    zernike_terms : list
        Noll Zernike terms.
    zernike_coeffs : np.array
        The Zernike coefficients from each time step in meters.
        Should be a 2D array (timesteps, model outputs).
    title : str
        The title to display.
    total_time : float
        Total time that the control loop ran over in seconds (assumes the time
        started at 0).
    n_rows : int
        Number of rows in the plot.
    n_cols : int
        Number of columns in the plot.
    plot_path : str
        Path to save the plot at.
    plot_psd : bool
        Plot the PSD instead of the time series.
    extra_zernike_coeffs : np.array
        A second set of Zernike coefficients to plot for each subplot. Should be
        the same format as the `zernike_coeffs` argument.
    legend_labels : list
        List of labels to display on the legend.
    add_std : bool
        Add the standard deviation to each subplot's title.
    """

    # Load in the style file
    plt.style.use(PLOT_STYLE_FILE)

    # =============================
    # Ensure there are enough cells
    # =============================

    total_steps, col_count = zernike_coeffs.shape
    if n_rows * n_cols < col_count:
        terminate_with_message('Not enough rows and columns for the data.')

    # ==================
    # Setup the subplots
    # ==================

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 4, n_rows * 2.5),
    )

    # ===============
    # Do the plotting
    # ===============

    plt.suptitle(title)
    current_col = 0
    for plot_row in range(n_rows):
        for plot_col in range(n_cols):
            # Remove axes for unused cells
            if current_col >= col_count:
                fig.delaxes(axs[plot_row, plot_col])
                continue
            # Data for the current cell in nm
            cell_data = zernike_coeffs[:, current_col] * 1e9
            if extra_zernike_coeffs is not None:
                extra_cell_data = extra_zernike_coeffs[:, current_col] * 1e9
            axs_cell = axs[plot_row, plot_col]
            # Add the Zernike number to the top left of the plot
            # https://stackoverflow.com/a/50091489
            axs_cell.annotate(
                f'Z{zernike_terms[current_col]}',
                (0, 1),
                xytext=(5, -5),
                xycoords='axes fraction',
                textcoords='offset points',
                fontweight='bold',
                color='white',
                backgroundcolor='k',
                ha='left',
                va='top',
            )
            # Choose whether to do a PSD or time series plot. The labels for
            # these do not matter, they will get overwritten at the end.
            if plot_psd:
                delta_time = total_time / (total_steps - 1)
                sampling_freq = 1 / delta_time
                axs_cell.psd(cell_data, Fs=sampling_freq, label='A')
                if extra_zernike_coeffs is not None:
                    axs_cell.psd(extra_cell_data, Fs=sampling_freq, label='B')
            else:
                # Plot the line along zero
                axs_cell.plot(np.zeros_like(cell_data), color='black')
                axs_cell.plot(cell_data, label='A')
                if extra_zernike_coeffs is not None:
                    axs_cell.plot(extra_cell_data, label='B')
                if add_std:
                    axs_cell.annotate(
                        f'STD {np.std(cell_data):0.3f}',
                        (0, 0),
                        xytext=(5, 5),
                        xycoords='axes fraction',
                        textcoords='offset points',
                        color='white',
                        backgroundcolor='k',
                        ha='left',
                        va='bottom',
                    )
            # Hide labels so they do not appear on every plot
            axs_cell.set_xlabel('')
            axs_cell.set_ylabel('')
            # Ensure each subplots has the same x limits
            x_ticks = np.linspace(0, total_steps, 5)
            axs_cell.set_xlim(min(x_ticks), max(x_ticks))
            # Only display x labels for the last active cell in each col
            total_terms = len(zernike_terms)
            if ((plot_row == n_rows - 1) or
                ((total_terms % n_cols > 0) and (plot_row == n_rows - 2) and
                 (plot_col >= (total_terms % n_cols)))):
                if not plot_psd:
                    # Set the x labels to time
                    labs = [f'{v:0.2f}' for v in np.linspace(0, total_time, 5)]
                    axs_cell.set_xticks(x_ticks, labs)
                label = 'Frequency' if plot_psd else 'Time [s]'
                axs_cell.set_xlabel(label)
            else:
                axs_cell.set_xticks([])
                axs_cell.set_xticklabels([])
            # Only display y labels for the first column
            if plot_col == 0:
                label = 'PSD\n[nm RMS/Hz]' if plot_psd else 'Coeffs\n[nm RMS]'
                axs_cell.set_ylabel(label)
            # Remove the padding on the left and right sides of the subplots
            axs_cell.set_xmargin(0)
            current_col += 1

    # Add the legend only if the labels were passed
    if legend_labels is not None:
        lines, labels = axs_cell.get_legend_handles_labels()
        fig.legend(lines, legend_labels, loc='lower right')

    fig.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.1)

    # =============
    # Save the plot
    # =============

    plt.savefig(plot_path)
