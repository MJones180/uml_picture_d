import matplotlib.pyplot as plt
import numpy as np
from utils.constants import PLOT_STYLE_FILE
from utils.idl_rainbow_cmap import idl_rainbow_cmap

# The colormap used for log plots
log_cmap = idl_rainbow_cmap()


def plot_wavefront(
    data,
    colorbar_label,
    plot_sampling,
    title,
    plot_path,
    use_log=False,
    disable_plot_ticks=False,
):
    # Load in the style file
    plt.style.use(PLOT_STYLE_FILE)
    plot_points = data.shape[0]
    # Reset the plot
    plt.clf()
    plt.title(title, pad=20)
    if use_log:
        # Ignore divide by zero errors here if they occurr
        with np.errstate(divide='ignore'):
            data = np.log10(data)
        vmin = -14
        data[data == -np.inf] = vmin
        data = np.nan_to_num(data, nan=vmin)
        plt.imshow(data, vmin=vmin, vmax=0, cmap=log_cmap)
    else:
        plt.imshow(data, cmap='Greys_r')
    plt.xlabel('X [mm]')
    plt.ylabel('Y [mm]')
    tick_count = 3
    tick_locations = np.linspace(0, plot_points - 1, tick_count)
    # Half the width of the grid in mm (originally in meters)
    grid_rad_mm = 1e3 * plot_sampling * plot_points / 2
    tick_labels = np.linspace(-grid_rad_mm, grid_rad_mm, tick_count)
    # Sometimes the middle tick likes to be negative
    tick_labels[tick_count // 2] = 0
    # Round to two decimal places
    tick_labels = [f'{label:.2f}' for label in tick_labels]
    plt.xticks(tick_locations, tick_labels)
    # The y ticks get plotted from top to bottom, so flip them
    plt.yticks(tick_locations, tick_labels[::-1])
    if disable_plot_ticks:
        plt.axis('off')
    plt.colorbar(label=colorbar_label)
    plt.savefig(plot_path)


# ==============================================================================
# To avoid cluttering up the code with too many arguments that will rarely be
# used, I added the block of code below that can be uncommented when needed.
# It saves just the pixel data (no axes, title, colorbar, etc.).
# ==============================================================================
# def plot_wavefront(
#     data,
#     colorbar_label,
#     plot_sampling,
#     title,
#     plot_path,
#     use_log=False,
# ):
#     # Reset the plot
#     plt.clf()
#     if use_log:
#         # Ignore divide by zero errors here if they occurr
#         with np.errstate(divide='ignore'):
#             data = np.log10(data)
#         vmin = -8
#         data[data == -np.inf] = vmin
#         plt.imshow(data, vmin=vmin, vmax=0, cmap=log_cmap)
#     else:
#         plt.imshow(data, cmap='Greys_r')
#     plt.axis('off')
#     plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0)
# ==============================================================================
