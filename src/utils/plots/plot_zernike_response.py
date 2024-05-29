import matplotlib.pyplot as plt
import numpy as np
from utils.constants import ZERNIKE_NAME_LOOKUP
from utils.idl_rainbow_cmap import idl_rainbow_cmap


def plot_zernike_response(
    zernike_terms,
    truth_groupings,
    pred_groupings,
    title_append,
    plot_path,
):
    """
    Generates and saves a Zernike response plot.

    Parameters
    ----------
    zernike_terms : list
        Noll Zernike terms.
    truth_groupings : np.array
        The truth data, see the Notes for more information.
    pred_groupings : np.array
        The prediction data, see the Notes for more information.
    title_append : str
        Value to add to the title.
    plot_path : str
        Path to save the plot at.

    Notes
    -----
    The `*_groupings` arguments must be 3D arrays and have the shape of
    (rms pert, zernike terms, zernike terms).

    It is assumed that the truth Zernike terms all have the same perturbation
    for each group and that there are only perturbations along the main
    diagonal. Therefore, each group (first dim of the truth array) should be
    equivalent to `perturbation * identity matrix`.
    """

    # The RMS perturbations should be the same for every Zernike polynomial
    truth_rms_pert = truth_groupings[:, 0, 0]

    # Set the figure size and add the title + axes labels
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    plt.title(f'Zernike Response ({title_append})')
    plt.xlabel('Truth [nm RMS]')
    plt.ylabel('Output [nm RMS]')

    # Make the x and y axes have the same range and set a 1:1 aspect ratio
    min_val = np.min(truth_rms_pert)
    max_val = np.max(truth_rms_pert)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect(1)

    def _add_line(x_vals=ax.get_xlim(), y_vals=ax.get_xlim()):
        ax.plot(
            x_vals,
            y_vals,
            linestyle='--',
            linewidth=1,
            color='#FF0000',
            scalex=False,
            scaley=False,
        )

    # Lines for 1-to-1, x, and y
    _add_line()
    _add_line(x_vals=(0, 0))
    _add_line(y_vals=(0, 0))

    # The colors that will be plotted for each line
    colors = idl_rainbow_cmap()(np.linspace(0, 1, len(zernike_terms)))

    # For this plot, we only care about elements along the main diagonal
    for term_idx, term in enumerate(zernike_terms):
        ax.plot(truth_rms_pert,
                pred_groupings[:, term_idx, term_idx],
                label=f'Z{term} {ZERNIKE_NAME_LOOKUP[term]}',
                color=colors[term_idx])

    # Set the labels
    tick_idxs = np.linspace(0, len(truth_rms_pert) - 1, 7)
    tick_idxs = np.round(tick_idxs).astype(int)
    tick_pos = truth_rms_pert[tick_idxs]
    # Need to put the positions into nm
    tick_labels = [f'{a:.0f}' for a in tick_pos * 1e9]
    ax.set_xticks(tick_pos, tick_labels)
    ax.set_yticks(tick_pos, tick_labels)

    # Display the legend to the right middle of the plot
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
    # Ensure the legend does not get cut off
    plt.tight_layout()

    # Save the plot
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
