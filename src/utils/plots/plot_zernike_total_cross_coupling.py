import matplotlib.pyplot as plt
import numpy as np
from utils.constants import PLOT_STYLE_FILE, ZERNIKE_NAME_LOOKUP
from utils.idl_rainbow_cmap import idl_rainbow_colors
from utils.stats_and_error import rss

# The old version of this plot only output a single line for all the terms.
# The new version outputs a line for each Zernike, along with a line for all
# the terms. If for some reason the old plot is needed, then this variable
# should be manually set to False.
LINE_PER_ZERNIKE = True


def plot_zernike_total_cross_coupling(
    zernike_terms,
    perturbation_grid,
    pred_groupings,
    title_append,
    identifier,
    plot_path=None,
    interactive_view=False,
):
    """
    Generates and saves a Zernike total cross coupling plot.

    Only one Zernike term should be perturbed at a time.

    Parameters
    ----------
    zernike_terms : list
        Noll Zernike terms.
    perturbation_grid : np.array
        Array for how much each group is perturbed by.
    pred_groupings : np.array
        The prediction data, 3D array (rms pert, zernike terms, zernike terms).
    title_append : str
        Value to add to the title.
    identifier : str
        Identifier for what predicted the data.
    plot_path : str
        Path to save the plot at.
    interactive_view : bool
        Display the plot in interactive mode instead of saving it.
    """

    # Load in the style file
    plt.style.use(PLOT_STYLE_FILE)

    # Set the figure size and add the title + axes labels
    fig, ax = plt.subplots(figsize=(8, 6.5))
    title = f'Zernike Total Cross Coupling ({title_append})\n{identifier}'
    ax.set_title(title)
    ax.set_xlabel('Input Zernike Amplitude [nm RMS]')
    ax.set_ylabel('RSS Total Cross Coupling [nm RMS]')

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

    if LINE_PER_ZERNIKE:
        # This is the total cross coupling for each Zernike term. As an example,
        # if 10 nm is injected on Z2, then this is the RSS of crosstalk from all
        # the other Zernike terms.
        crosstalk_each = rss(pred_groupings_no_diag, 2)
        # The colors that will be plotted for each line
        colors = idl_rainbow_colors(len(zernike_terms))
        # Plot the cross coupling for each Zernike term
        for term_idx, term in enumerate(zernike_terms):
            ax.plot(
                perturbation_grid,
                crosstalk_each[:, term_idx],
                label=f'Z{term} {ZERNIKE_NAME_LOOKUP[term]}',
                color=colors[term_idx],
            )
        # Plot the total crosstalk from all Zernikes
        ax.fill_between(
            perturbation_grid,
            crosstalk_total,
            color='#F3F3F3',
            edgecolor='black',
            linestyle='--',
            linewidth=1,
            label='All Terms',
        )
        # Display the legend to the right middle of the plot
        ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
    else:
        # Plot the total crosstalk from all Zernikes
        ax.plot(perturbation_grid, crosstalk_total, color='black')

    if interactive_view:
        plt.show()
    else:
        # Save the plot
        plt.savefig(plot_path)
