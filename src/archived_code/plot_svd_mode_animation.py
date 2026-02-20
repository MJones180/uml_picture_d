"""
Animation that shows how SVD modes can be used to represent data.
"""

from h5py import File
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# ==============================================================================
# Datafile paths
# ==============================================================================

DATAFILE_PATH = '../data/raw/dh_both_hodms_efc_10_row_saved_surfaces/0_data.h5'
MODES_PATH = '../data/raw/dm1_modes_flat/0_data.h5'
OUT_ANIMATION_PATH = 'svd_animation.gif'
ANIMATION_FPS = 10
MODES_TO_PLOT = 500

# ==============================================================================
# Load in the DM1 command
# ==============================================================================

# Load in the data and use the first pattern for DM1
dm_cmd = File(DATAFILE_PATH)['dm1'][0]
number_pixels = dm_cmd.shape[0]
# The original DM command flattened
dm_cmd_flat = dm_cmd.reshape(-1)
# The active locations for the DM actuators when flattened
active_pixel_idxs = dm_cmd_flat != 0
# Take only the active pixels
dm_cmd_flat = dm_cmd_flat[active_pixel_idxs]

# ==============================================================================
# Load in the DM1 modes
# ==============================================================================

# Load in the DM1 modes
modes = File(MODES_PATH)['dm1_modes'][:].T
number_modes = modes.shape[0]
# Invert the SVD modes
modes_inv = np.linalg.pinv(modes)
# The DM command in terms of SVD mode coefficients
svd_mode_coeffs = dm_cmd_flat @ modes_inv

# ==============================================================================
# Create the animation
# ==============================================================================

# Create a figure with three subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# The plotting objects
ax_plots = []
for ax in axs:
    ax_plot = ax.imshow(
        np.zeros((number_pixels, number_pixels)),
        animated=True,
    )
    # Save the plotting object
    ax_plots.append(ax_plot)
    # Remove the tick and tick markers
    ax.axis('off')
    # Add the coloarbar
    cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05)
    fig.colorbar(ax_plot, cax=cax, orientation='vertical')


def set_subplot_data(ax_idx, data):
    ax_plots[ax_idx].set_data(data)
    # Need to update the limits to make the colorbar update as well
    ax_plots[ax_idx].set_clim(np.min(data), np.max(data))


def set_subplot_title(ax_idx, title):
    axs[ax_idx].set_title(title)


# The first subplot will not change
set_subplot_data(0, dm_cmd)
set_subplot_title(0, 'Original DM Command')
set_subplot_title(2, 'Reconstructed DM Command')

# The DM command reconstructed from each SVD mode
cumulative_dm_cmd_from_svd = np.zeros((number_pixels, number_pixels))


def update(mode_idx):
    print(f'On mode {mode_idx}')
    # Need to make the variable global to update it
    global cumulative_dm_cmd_from_svd
    # The SVD coeff associated with this mode
    mode_coeff = svd_mode_coeffs[mode_idx]
    # The 2D mode needed to reconstruct the orignal DM command
    mode_2d = np.zeros(number_pixels**2)
    mode_2d[active_pixel_idxs] = mode_coeff * modes[mode_idx]
    mode_2d = mode_2d.reshape((number_pixels, number_pixels))
    # Add this mode to the DM command reconstruction
    cumulative_dm_cmd_from_svd += mode_2d
    # Update the subplots
    set_subplot_data(1, mode_2d)
    set_subplot_data(2, cumulative_dm_cmd_from_svd)
    set_subplot_title(1, f'Mode {mode_idx + 1}, Coeff {mode_coeff:0.5f}')


# Remove extra padding
fig.tight_layout()
# Generate the animation and save it (note: Saving may take a long time)
FuncAnimation(
    fig=fig,
    func=update,
    frames=MODES_TO_PLOT,
).save(OUT_ANIMATION_PATH, writer=PillowWriter(fps=ANIMATION_FPS))
