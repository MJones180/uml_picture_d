import matplotlib.pyplot as plt
import numpy as np
from utils.constants import PLOT_STYLE_FILE, ZERNIKE_NAME_LOOKUP
from utils.idl_rainbow_cmap import idl_rainbow_colors


def plot_control_loop_zernikes(
    zernike_terms,
    zernike_coeffs,
    title,
    info,
    total_time,
    plot_path,
    plot_psd=False,
):
    """
    Generates and saves a plot of all the Zernike coefficients plotted overtop
    of each other. The plot shows either the time series or PSD of the Zernike
    coefficients from a control loop run.

    Parameters
    ----------
    zernike_terms : list
        Noll Zernike terms.
    zernike_coeffs : np.array
        The Zernike coefficients from each time step in meters.
        Should be a 2D array (timesteps, model outputs).
    title : str
        The title to display.
    info : str
        The info text to display.
    total_time : float
        Total time that the control loop ran over in seconds (assumes the time
        started at 0).
    plot_path : str
        Path to save the plot at.
    plot_psd : bool
        Plot the PSD instead of the time series.
    """

    # Load in the style file
    plt.style.use(PLOT_STYLE_FILE)

    # Total number of time steps
    total_steps = len(zernike_coeffs)

    # Set the figure size and add the title + axes labels
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(title)

    # The colors that will be plotted for each line
    colors = idl_rainbow_colors(len(zernike_terms))

    # Plot each Zernike coefficient term
    for term_idx, term in enumerate(zernike_terms):
        # The data for the Zernike term in nm
        term_data = zernike_coeffs[:, term_idx] * 1e9
        label = f'Z{term} {ZERNIKE_NAME_LOOKUP[term]}'
        color = colors[term_idx]
        # Choose whether to do a PSD or time series plot
        if plot_psd:
            delta_time = total_time / (total_steps - 1)
            ax.psd(term_data, Fs=(1 / delta_time), label=label, color=color)
        else:
            ax.plot(term_data, label=label, color=color)

    # Update the axis information
    if plot_psd:
        ax.set_ylabel('PSD [nm RMS/Hz]')
    else:
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Coefficient [nm RMS]')
        # Set the x labels to time
        x_tick_pos = np.linspace(0, total_steps, 7)
        x_tick_labels = [f'{v:0.4f}' for v in np.linspace(0, total_time, 7)]
        ax.set_xticks(x_tick_pos, x_tick_labels)

    # Remove the margins on the x-axis
    ax.set_xmargin(0)

    # Display the legend to the right middle of the plot
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))

    # Add the info text box
    ax_pos = ax.get_position()
    plt.figtext(
        # Center of the plotting area
        ax_pos.x0 + (ax_pos.x1 - ax_pos.x0) / 2,
        0,
        info,
        ma='left',
        ha='center',
        va='top',
        bbox={},
    )

    # Save the plot
    plt.savefig(plot_path, pad_inches=0.1)
