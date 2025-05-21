from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from utils.constants import PLOT_STYLE_FILE
from utils.idl_rainbow_cmap import idl_rainbow_cmap


def plot_zernike_cross_coupling_mat_animation(
    zernike_terms,
    perturbation_grid,
    pred_groupings,
    title_append,
    identifier,
    animation_path,
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
    animation_path : str
        Path to save the animation at, must be `.gif`.
    """
    # Load in the style file
    plt.style.use(PLOT_STYLE_FILE)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f'Cross-Coupling Matrix ({title_append})\n{identifier}')
    ax.set_ylabel('Output Zernike')

    # Create the initial plot and colorbar that will be updated
    im = ax.imshow(np.zeros_like(pred_groupings[0]), cmap=idl_rainbow_cmap())
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical', label='nm RMS')

    # Invert the y-axis so that zero starts on the bottom
    ax.set_ylim(ax.get_ylim()[::-1])

    def update(frame_idx):
        frame_data = pred_groupings[frame_idx] * 1e9
        im.set_data(frame_data.T)
        # Need to update the limits to make the colorbar update as well
        im.set_clim(np.min(frame_data), np.max(frame_data))
        input_pert_amount = round(perturbation_grid[frame_idx] * 1e9)
        ax.set_xlabel(f'Input Zernike @ {input_pert_amount} nm RMS')

    # Generate the animation and save it
    FuncAnimation(
        fig=fig,
        func=update,
        frames=len(perturbation_grid),
    ).save(animation_path, writer=PillowWriter(fps=1))
