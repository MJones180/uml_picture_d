import matplotlib.pyplot as plt
import numpy as np
from utils.constants import PLOT_STYLE_FILE
from utils.stats_and_error import rss


def plot_control_loop_rss(
    zernike_coeffs_true,
    zernike_coeffs_meas,
    title,
    total_time,
    plot_path,
):
    """
    Plot the RSS of the coefficients for each timestep.

    Parameters
    ----------
    zernike_coeffs_true : np.array
        The true error Zernike coefficients from each time step in meters.
        Should be a 2D array (timesteps, true error).
    zernike_coeffs_meas : np.array
        The meas error Zernike coefficients from each time step in meters.
        Should be a 2D array (timesteps, model outputs).
    title : str
        The title to display.
    total_time : float
        Total time that the control loop ran over in seconds (assumes the time
        started at 0).
    plot_path : str
        Path to save the plot at.
    """

    # Load in the style file
    plt.style.use(PLOT_STYLE_FILE)

    # Set the figure size and add the title + axes labels
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(title, pad=20)

    # Plot the RSS
    ax.plot(
        rss(zernike_coeffs_meas * 1e9, 1),
        label='Measurement Error',
        color='red',
    )
    ax.plot(
        rss(zernike_coeffs_true * 1e9, 1),
        label='True Error',
        color='blue',
    )

    # Update the axis information
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('RSS [nm RMS]')
    # Set the x labels to time
    x_tick_pos = np.linspace(0, len(zernike_coeffs_true), 7)
    x_tick_labels = [f'{v:0.4f}' for v in np.linspace(0, total_time, 7)]
    ax.set_xticks(x_tick_pos, x_tick_labels)

    # Remove the margins on the x-axis
    ax.set_xmargin(0)

    # Add in the legend
    plt.legend(
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc='lower center',
        borderaxespad=0,
        ncol=2,
    )

    # Save the plot
    plt.savefig(plot_path, pad_inches=0.1)
