"""
Plot customized for the paper:
    Adaptive Optics Wavefront Capture and Stabilization Using Convolutional
    Neural Networks
Based on original plotting script:
    utils/plots/plot_comparison_scatter_grid.py
Requirements:
    - The data must all be in meters
    - Must be Zernikes 2-24
    - Every Zernike must share the same limits
"""

import matplotlib.pyplot as plt
from utils.constants import PLOT_STYLE_FILE

MODEL = 'RM'
MODEL = 'Capture CNN'
MODEL = 'Stabilization CNN'
N_ROWS = N_COLS = 2
# (row idx, col idx, zernike idx)
# For terms 2, 4, 11, 22
CELLS = [(0, 0, 0), (0, 1, 2), (1, 0, 9), (1, 1, 20)]
LIM = -2, 2


def paper_plot_model_scatters(
    model_data,
    truth_data,
    starting_zernike,
    plot_path=None,
):
    # Load in the style file
    plt.style.use(PLOT_STYLE_FILE)
    fig, axs = plt.subplots(
        N_ROWS,
        N_COLS,
        figsize=(N_COLS * 4, N_ROWS * 4),
        sharex=True,
        sharey=True,
    )
    plt.suptitle(f'Truth vs {MODEL}')
    for plot_row, plot_col, zernike_col in CELLS:
        # Grab the data for the current cell, put it in nm
        model_col = model_data[:, zernike_col] * 1e9
        truth_col = truth_data[:, zernike_col] * 1e9
        axs_cell = axs[plot_row, plot_col]
        # Add the Zernike number to the top left of the plot
        # https://stackoverflow.com/a/50091489
        axs_cell.annotate(
            f'Z{starting_zernike + zernike_col}',
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
        # Draw a 1-to-1 line, it should be based on truth values
        # https://stackoverflow.com/a/60950862
        axs_cell.plot(
            LIM,
            LIM,
            linestyle='-',
            color='#000000',
            scalex=False,
            scaley=False,
            zorder=-1,
        )
        # Plot the scatter of all the points
        axs_cell.scatter(truth_col, model_col, 1, alpha=.1)
        # Set the limits on each subplot
        axs_cell.set_xlim(*LIM)
        axs_cell.set_ylim(*LIM)
        # Only display x labels for the last row
        if plot_row == N_ROWS - 1:
            axs_cell.set_xlabel('Truth Outputs [nm]')
        # Only display y labels for the first column
        if plot_col == 0:
            axs_cell.set_ylabel('Model Outputs [nm]')
        axs_cell.locator_params(nbins=4)
    fig.tight_layout()
    plt.subplots_adjust(top=0.94, wspace=0.1, hspace=0.1)
    plt.savefig(plot_path)
