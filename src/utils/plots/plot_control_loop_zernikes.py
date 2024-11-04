import matplotlib.pyplot as plt
import numpy as np
from utils.constants import ZERNIKE_NAME_LOOKUP
from utils.idl_rainbow_cmap import idl_rainbow_cmap


def plot_control_loop_zernikes(
    zernike_terms,
    zernike_time_steps,
    model_str,
    gain_values,
    total_time,
    plot_path,
):
    """
    Generates and saves a plot showing the Zernike coefficients over time from
    a control loop run.

    Parameters
    ----------
    zernike_terms : list
        Noll Zernike terms.
    zernike_time_steps : np.array
        The Zernike coefficients outputted from the model at each time step.
        Should be a 2D array (timesteps, model outputs).
    model_str : str
        Identifier of the model being used.
    gain_values : (float, float, float)
        The K_p, K_i, and K_d gain values.
    total_time : float
        Total time that the control loop ran over in seconds (assumes the time
        started at 0).
    plot_path : str
        Path to save the plot at.
    """

    # Total number of time steps
    total_steps = len(zernike_time_steps)

    # Set the figure size and add the title + axes labels
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(
        f'Control Loop\nTime Steps={total_steps}, Model={model_str}, '
        f'K_p={gain_values[0]}, K_i={gain_values[1]}, K_d={gain_values[2]}')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Coefficient [nm RMS]')

    # The colors that will be plotted for each line
    colors = idl_rainbow_cmap()(np.linspace(0, 1, len(zernike_terms)))
    # The x points for the scatter plots
    x_points = np.arange(total_steps)
    # Plot each Zernike terms coefficients over time
    for term_idx, term in enumerate(zernike_terms):
        term_data = zernike_time_steps[:, term_idx]
        # Plot the lines on the bottom layer
        ax.plot(
            term_data,
            label=f'Z{term} {ZERNIKE_NAME_LOOKUP[term]}',
            color=colors[term_idx],
            zorder=0,
        )
        # Next, plot the dots at each of the points
        ax.scatter(
            x_points,
            term_data,
            s=30,
            color=colors[term_idx],
            zorder=1,
        )
        # Lastly, plot the Zernike Noll index on top
        ax.scatter(
            x_points,
            term_data,
            s=20,
            color='white',
            zorder=2,
            marker=f"${term}$",
        )

    # Set the x labels to time
    x_tick_pos = np.linspace(0, total_steps, 7)
    x_tick_labels = [f'{val:0.4f}' for val in np.linspace(0, total_time, 7)]
    ax.set_xticks(x_tick_pos, x_tick_labels)

    # Display the legend to the right middle of the plot
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
    # Ensure the legend does not get cut off
    plt.tight_layout()

    # Save the plot
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
