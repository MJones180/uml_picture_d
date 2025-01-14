import matplotlib.pyplot as plt
import numpy as np
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
    """

    # =============================
    # Ensure there are enough cells
    # =============================

    total_steps, col_count = zernike_coeffs.shape
    if n_rows * n_cols < col_count:
        terminate_with_message('Not enough rows and columns for the data.')

    # ==================
    # Setup the subplots
    # ==================

    subplot_args = {'figsize': (n_cols * 6, n_rows * 3)}
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        sharex=True,
        **subplot_args,
    )

    # ===============
    # Do the plotting
    # ===============

    plt.suptitle(title, size=30)
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
            # Choose whether to do a PSD or time series plot. The labels for
            # these do not matter, they will get overwritten at the end.
            if plot_psd:
                delta_time = total_time / (total_steps - 1)
                sampling_freq = 1 / delta_time
                axs_cell.psd(cell_data, Fs=sampling_freq, label='A')
                if extra_zernike_coeffs is not None:
                    axs_cell.psd(extra_cell_data, Fs=sampling_freq, label='B')
            else:
                axs_cell.plot(cell_data, label='A')
                if extra_zernike_coeffs is not None:
                    axs_cell.plot(extra_cell_data, label='B')
            # Hide labels so they do not appear on every plot
            axs_cell.set_xlabel('')
            axs_cell.set_ylabel('')
            # Only display x labels for the last row
            if plot_row == n_rows - 1:
                if not plot_psd:
                    # Set the x labels to time
                    pos = np.linspace(0, total_steps, 5)
                    labs = [f'{v:0.2f}' for v in np.linspace(0, total_time, 5)]
                    axs_cell.set_xticks(pos, labs)
                label = 'Frequency' if plot_psd else 'Time [s]'
                axs_cell.set_xlabel(label)
                axs_cell.xaxis.label.set_fontsize(20)
            # Only display y labels for the first column
            if plot_col == 0:
                label = 'PSD [nm RMS/Hz]' if plot_psd else 'Coeff [nm RMS]'
                axs_cell.set_ylabel(label)
                axs_cell.yaxis.label.set_fontsize(20)
            # Increase the font size for ticks
            for item in (axs_cell.get_xticklabels() +
                         axs_cell.get_yticklabels() + [
                             axs_cell.xaxis.get_offset_text(),
                             axs_cell.yaxis.get_offset_text()
                         ]):
                item.set_fontsize(15)
            current_col += 1

    # Add the legend only if the labels were passed
    if legend_labels is not None:
        lines, labels = axs_cell.get_legend_handles_labels()
        fig.legend(lines, legend_labels, loc='lower right', fontsize=25)

    # Fixes padding around subplots, `h_pad` decreases vertical spacing between
    # rows and `rect` makes room for the suptitle
    fig.tight_layout(h_pad=0, rect=[0, 0, 1, 0.99])

    # =============
    # Save the plot
    # =============

    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
