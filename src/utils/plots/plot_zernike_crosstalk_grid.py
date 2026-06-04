import matplotlib.pyplot as plt
from utils.constants import PLOT_STYLE_FILE


def plot_zernike_crosstalk_grid(
    zernike_terms,
    perturbation_grid,
    pred_groupings,
    plot_path=None,
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
    plot_path : str
        Path to save the plot at.
    """

    # Load in the style file
    plt.style.use(PLOT_STYLE_FILE)

    # The min and max values for the y-axis
    y_min = perturbation_grid[0] * 1e9
    y_max = perturbation_grid[-1] * 1e9

    # Put the predictions into nm
    pred_groupings_nm = pred_groupings * 1e9

    # The number of rows and cols to use
    rows = cols = pred_groupings.shape[1]

    fig, axes = plt.subplots(
        nrows=rows,
        ncols=cols,
        sharex=True,
        sharey=True,
        figsize=(25, 25),
    )

    # Set the spacing between the subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for row_idx in range(rows):
        for col_idx in range(cols):
            ax = axes[row_idx, col_idx]
            ax.plot(
                pred_groupings_nm[:, col_idx, row_idx],
                color='tab:blue' if row_idx == col_idx else 'tab:red',
                linewidth=4,
            )
            ax.set_xmargin(0)
            if row_idx < (rows - 1):
                # Hide inner ticks
                ax.tick_params(axis='x', which='both', length=0)
            else:
                # Subplots on the bottom row
                ax.locator_params(axis='x', nbins=3)
                ax.set_xticklabels([])
                ax.set_xlabel(rf'$Z_{{{zernike_terms[col_idx]}}}$')
            ax.set_ylim(y_min, y_max)
            if col_idx > 0:
                # Hide inner ticks
                ax.tick_params(axis='y', which='both', length=0)
            else:
                # Subplots on the left-most col
                ax.set_yticks([y_min, 0, y_max])
                ax.set_yticklabels([])
                ax.set_ylabel(rf'$Z_{{{zernike_terms[row_idx]}}}$')

    fig.supxlabel('Injected Error Zernike', y=0.08)
    fig.supylabel('Zernike Error Response', x=0.09)
    fig.suptitle(
        f'Zernike Crosstalk (Subplot Bounds [{y_min}, {y_max}] nm)',
        y=0.9,
        fontweight='bold',
    )
    plt.savefig(plot_path)
