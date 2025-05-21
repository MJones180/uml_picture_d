"""
Plot customized for the paper:
    Adaptive Optics Wavefront Stabilization Using a Convolutional Neural Network
Based on original plotting script:
    utils/plots/plot_zernike_total_cross_coupling.py
Requirements:
    - The wavefronts must contain single Zernikes at a time
"""

import matplotlib.pyplot as plt
import numpy as np
from utils.constants import PLOT_STYLE_FILE
from utils.idl_rainbow_cmap import idl_rainbow_colors
from utils.stats_and_error import rss


def paper_plot_total_crosstalk(
    zernike_terms,
    perturbation_grid,
    pred_groupings,
    title_append,
    plot_path=None,
):
    # Load in the style file
    plt.style.use(PLOT_STYLE_FILE)

    # Set the figure size and add the title + axes labels
    fig, ax = plt.subplots(figsize=(8, 6.5))
    title = f'Zernike Total Cross Coupling ({title_append})'
    ax.set_title(title, pad=10)
    ax.set_xlabel(r'Input Zernike Amplitude, $a$ [nm RMS]')
    ax.set_ylabel(r'RSS Total Cross Coupling, $\gamma$ [nm RMS]')

    # Set the limits on the x-axis
    ax.set_xlim(np.min(perturbation_grid), np.max(perturbation_grid))
    # Set the labels
    tick_idxs = np.linspace(0, len(perturbation_grid) - 1, 7)
    tick_idxs = np.round(tick_idxs).astype(int)
    tick_pos = perturbation_grid[tick_idxs]
    # Need to put the positions into nm
    tick_labels = [f'{a:.0f}' for a in tick_pos * 1e9]
    ax.set_xticks(tick_pos, tick_labels)

    # Zero out all entries along the main diagonal
    diag_idxs = np.arange(pred_groupings.shape[1])
    # Convert to nm and remove the main diagonal
    pred_groupings_no_diag = pred_groupings * 1e9
    pred_groupings_no_diag[:, diag_idxs, diag_idxs] = 0

    # The total crosstalk from all Zernikes
    crosstalk_total = rss(pred_groupings_no_diag, (1, 2))

    # This is the total cross coupling for each Zernike term. As an example,
    # if 10 nm is injected on Z2, then this is the RSS of crosstalk from all
    # the other Zernike terms.
    crosstalk_each = rss(pred_groupings_no_diag, 2)
    # The colors that will be plotted for each line
    colors = idl_rainbow_colors(len(zernike_terms))
    # Plot the cross coupling for each Zernike term
    for term_idx, term in enumerate(zernike_terms):
        gamma_subscript = '{' + f'{term},a' + '}'
        ax.plot(
            perturbation_grid,
            crosstalk_each[:, term_idx],
            label=fr'$\gamma_{gamma_subscript}$',
            color=colors[term_idx],
        )
    # Plot the total crosstalk from all Zernikes
    ax.plot(
        perturbation_grid,
        crosstalk_total,
        linestyle='--',
        label=r'$\gamma_{:,a}$',
        color='black',
    )
    # Display the legend to the right middle of the plot
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))

    # Save the plot
    plt.savefig(plot_path)
