import matplotlib.pyplot as plt
import numpy as np
from utils.stats_and_error import rss


def plot_zernike_total_cross_coupling(
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

    # Set the figure size and add the title + axes labels
    fig, ax = plt.subplots()
    title = f'Zernike Total Cross Coupling ({title_append})\n{identifier}'
    ax.set_title(title)
    ax.set_xlabel('Input Zernike Amplitude [nm RMS]')
    ax.set_ylabel('Total RSS Cross Coupling [nm RMS]')

    # Zero out all entries along the main diagonal
    diag_idxs = np.arange(pred_groupings.shape[1])
    # Put the values in nm
    pred_groupings_nm = pred_groupings * 1e9
    # For 0 nm RMS, every row in the group will be the same. That means the
    # total calculated error will be much higher than it actually is. To fix
    # this, we can directly calculate the correct error and replace it.
    idx_for_0_nm = np.where(perturbation_grid == 0)[0]
    total_coupled_error_at_0 = rss(pred_groupings_nm[idx_for_0_nm, 0])
    # Copy and remove the main diagonal
    pred_groupings_no_diag = np.copy(pred_groupings_nm)
    pred_groupings_no_diag[:, diag_idxs, diag_idxs] = 0
    # Calculate the total coupled error at each perturbation amount
    total_coupled_error = rss(pred_groupings_no_diag, (1, 2))
    # Replace the error for at 0 nm
    total_coupled_error[idx_for_0_nm] = total_coupled_error_at_0

    # Set the limits on the x-axis
    ax.set_xlim(np.min(perturbation_grid), np.max(perturbation_grid))

    ax.plot(perturbation_grid, total_coupled_error)

    # Set the labels
    tick_idxs = np.linspace(0, len(perturbation_grid) - 1, 7)
    tick_idxs = np.round(tick_idxs).astype(int)
    tick_pos = perturbation_grid[tick_idxs]
    # Need to put the positions into nm
    tick_labels = [f'{a:.0f}' for a in tick_pos * 1e9]
    ax.set_xticks(tick_pos, tick_labels)

    if interactive_view:
        plt.show()
    else:
        # Save the plot
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
