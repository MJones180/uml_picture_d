import matplotlib.pyplot as plt
import numpy as np
from utils.constants import PLOT_STYLE_FILE, ZERNIKE_NAME_LOOKUP
from utils.idl_rainbow_cmap import idl_rainbow_colors


def plot_zernike_response(
    zernike_terms,
    perturbation_grid,
    pred_groupings,
    title_append,
    identifier=None,
    plot_path=None,
    interactive_view=False,
):
    """
    Generates and saves a Zernike response plot.

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
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_title = f'Zernike Response ({title_append})'
    if identifier is not None:
        plot_title += f'\n{identifier}'
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
                label=f'Z{term} {ZERNIKE_NAME_LOOKUP[term]}',
                color=colors[term_idx])

    # Set the labels
    tick_idxs = np.linspace(0, len(perturbation_grid) - 1, 7)
    tick_idxs = np.round(tick_idxs).astype(int)
    tick_pos = perturbation_grid[tick_idxs]
    # Need to put the positions into nm
    tick_labels = [f'{a:.0f}' for a in tick_pos * 1e9]
    ax.set_xticks(tick_pos, tick_labels)
    ax.set_yticks(tick_pos, tick_labels)

    # Display the legend to the right middle of the plot
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))

    if interactive_view:
        plt.show()
    else:
        # Save the plot
        plt.savefig(plot_path)
