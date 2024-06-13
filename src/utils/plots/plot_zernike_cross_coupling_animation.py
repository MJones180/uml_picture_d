from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import numpy as np
from utils.constants import ZERNIKE_NAME_LOOKUP
from utils.idl_rainbow_cmap import idl_rainbow_cmap


def plot_zernike_cross_coupling_animation(
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

    # Set the figure size and add the axes labels
    fig, ax = plt.subplots(figsize=(8, 8))
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
            linewidth=1,
            color='#FF0000',
            scalex=False,
            scaley=False,
        )

    # Lines for 1-to-1, x, and y
    _add_line()
    _add_line(x_vals=(0, 0))
    _add_line(y_vals=(0, 0))

    zernike_count = len(zernike_terms)

    # The colors that will be plotted for each line
    colors = idl_rainbow_cmap()(np.linspace(0, 1, zernike_count))

    lines = [
        ax.plot(perturbation_grid,
                np.zeros_like(perturbation_grid),
                label=f'Z{term} {ZERNIKE_NAME_LOOKUP[term]}',
                color=colors[term_idx])[0]
        for term_idx, term in enumerate(zernike_terms)
    ]

    base_title = f'Cross-Coupling ({title_append})\n{identifier}\n'

    def update(frame_idx):
        for line_idx, line in enumerate(lines):
            line.set_ydata(pred_groupings[:, frame_idx, line_idx])
        term = zernike_terms[frame_idx]
        ax.set_title(f'{base_title}Z{term} {ZERNIKE_NAME_LOOKUP[term]}')

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
    # Ensure the legend does not get cut off
    plt.tight_layout()

    # Generate the animation and save it
    FuncAnimation(
        fig=fig,
        func=update,
        frames=zernike_count,
    ).save(animation_path, writer=PillowWriter(fps=1))
