"""
Plot customized for the paper:
    Adaptive Optics Wavefront Capture and Stabilization Using Convolutional
    Neural Networks
Based on original plotting script:
    utils/plots/plot_zernike_response.py
Requirements:
    - The data must all be in meters
    - The wavefronts must contain single Zernikes at a time
"""

import matplotlib.pyplot as plt
import numpy as np
from utils.constants import PLOT_STYLE_FILE
from utils.idl_rainbow_cmap import idl_rainbow_colors

MODEL = 'RM'
MODEL = 'Capture CNN'
MODEL = 'Stabilization CNN'
LABEL_DECIMALS = 2


def paper_plot_zernike_response(
    zernike_terms,
    perturbation_grid,
    pred_groupings,
    plot_path,
):
    # Load in the style file
    plt.style.use(PLOT_STYLE_FILE)

    # Set the figure size and add the title + axes labels
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_title = f'Zernike Response ({MODEL})'
    ax.set_title(plot_title)
    ax.set_xlabel('Truth [nm RMS]')
    ax.set_ylabel('Output [nm RMS]')

    # Make the x and y axes have the same range and set a 1:1 aspect ratio
    min_val = np.min(perturbation_grid)
    max_val = np.max(perturbation_grid)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect(1)

    def _add_line(x_vals=ax.get_xlim(), y_vals=ax.get_xlim()):
        ax.plot(
            x_vals,
            y_vals,
            linestyle='--',
            color='#FF0000',
            scalex=False,
            scaley=False,
        )

    # Lines for 1-to-1, x, and y
    _add_line()
    _add_line(x_vals=(0, 0))
    _add_line(y_vals=(0, 0))

    # The colors that will be plotted for each line
    colors = idl_rainbow_colors(len(zernike_terms))

    # For this plot, we only care about elements along the main diagonal
    for term_idx, term in enumerate(zernike_terms):
        ax.plot(perturbation_grid,
                pred_groupings[:, term_idx, term_idx],
                label=fr'$Z_{{{term}}}$',
                color=colors[term_idx])

    # Set the labels
    tick_idxs = np.linspace(0, len(perturbation_grid) - 1, 7)
    tick_idxs = np.round(tick_idxs).astype(int)
    tick_pos = perturbation_grid[tick_idxs]
    # Need to put the positions into nm
    tick_labels = [f'{a:.{LABEL_DECIMALS}f}' for a in tick_pos * 1e9]
    ax.set_xticks(tick_pos, tick_labels)
    ax.set_yticks(tick_pos, tick_labels)

    # Display the legend to the right middle of the plot
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))

    # Save the plot
    plt.savefig(plot_path)
