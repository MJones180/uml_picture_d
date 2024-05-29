import matplotlib.pyplot as plt
import numpy as np
from utils.error import rss


def plot_zernike_total_cross_coupling(
    truth_groupings,
    pred_groupings,
    title_append,
    plot_path,
):
    """
    Generates and saves a Zernike total cross coupling plot.

    Parameters
    ----------
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
    plt.figure()
    ax = plt.subplot(111)
    plt.title(f'Zernike Total Cross Coupling ({title_append})')
    plt.xlabel('Input Zernike Amplitude [nm RMS]')
    plt.ylabel('Total Cross Coupling [nm RMS]')

    # Zero out all entries along the main diagonal
    diag_idxs = np.arange(pred_groupings.shape[1])
    # Copy and put in nm
    pred_groupings_no_diag = np.copy(pred_groupings) * 1e9
    pred_groupings_no_diag[:, diag_idxs, diag_idxs] = 0
    # Blank array to compare against
    zeros = np.zeros_like(pred_groupings_no_diag)
    # Calculate the total coupled error at each perturbation amount
    total_coupled_error = rss(pred_groupings_no_diag, zeros, (1, 2))

    # Set the limits on the x-axis
    ax.set_xlim(np.min(truth_rms_pert), np.max(truth_rms_pert))

    ax.plot(truth_rms_pert, total_coupled_error)

    # Set the labels
    tick_idxs = np.linspace(0, len(truth_rms_pert) - 1, 7)
    tick_idxs = np.round(tick_idxs).astype(int)
    tick_pos = truth_rms_pert[tick_idxs]
    # Need to put the positions into nm
    tick_labels = [f'{a:.0f}' for a in tick_pos * 1e9]
    ax.set_xticks(tick_pos, tick_labels)

    # Ensure the legend does not get cut off
    plt.tight_layout()

    # Save the plot
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
