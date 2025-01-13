import matplotlib.pyplot as plt
import numpy as np
from utils.constants import ZERNIKE_NAME_LOOKUP
from utils.idl_rainbow_cmap import idl_rainbow_cmap


def plot_control_loop_zernikes_psd(
    zernike_terms,
    zernike_time_steps,
    step_file,
    title_info,
    delta_time,
    plot_path,
):
    """
    Generates and saves a plot showing the PSD of the Zernike coefficients over
    time from a control loop run. This should probably just be merged with the
    `plot_control_loop_zernikes` plotting function, but for now it is separate.

    Parameters
    ----------
    zernike_terms : list
        Noll Zernike terms.
    zernike_time_steps : np.array
        The Zernike coefficients outputted from the model at each time step.
        Should be a 2D array (timesteps, model outputs). Values should be in nm.
    step_file : str
        Name of the step file being used.
    title_info : str
        Additional info on the run displayed as the second line of the title.
    delta_time : float
        Time between time steps.
    plot_path : str
        Path to save the plot at.
    """

    # Total number of time steps
    total_steps = len(zernike_time_steps)
    # Set the figure size and add the title + axes labels
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_title(f'Control Loop, Step File={step_file}, '
                 f'Timesteps={total_steps}\n{title_info}')
    ax.set_ylabel('PSD [nm RMS/Hz]')
    # The colors that will be plotted for each line
    colors = idl_rainbow_cmap()(np.linspace(0, 1, len(zernike_terms)))
    # Plot each Zernike terms coefficients over time
    for term_idx, term in enumerate(zernike_terms):
        term_data = zernike_time_steps[:, term_idx] * 1e9
        ax.psd(
            term_data,
            Fs=(1 / delta_time),
            label=f'Z{term} {ZERNIKE_NAME_LOOKUP[term]}',
            color=colors[term_idx],
            linewidth=0.8,
        )
    # Display the legend to the right middle of the plot
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
    # Ensure the legend does not get cut off
    plt.tight_layout()
    # Save the plot
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
