import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from utils.constants import PLOT_STYLE_FILE


def plot_dh_contrast(data, vmin, vmax, title, plot_path):
    # Load in the style file
    plt.style.use(PLOT_STYLE_FILE)
    # Reset the plot
    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 8))
    max_contrast = np.log10(np.max(data[data != 0]))
    avg_contrast = np.log10(np.mean(data[data != 0]))
    title += f'\nContrasts: Max {max_contrast:.3f}, Avg {avg_contrast:.3f}'
    ax.set_title(title, pad=20)
    # Find the indices of the active rows/cols
    active_col_idxs = np.where((data != 0).any(axis=0))[0]
    active_row_idxs = np.where((data != 0).any(axis=1))[0]
    # Chop off the empty rows/cols
    data = data[active_row_idxs]
    data = data[:, active_col_idxs]
    im = ax.imshow(data, norm='log', vmin=vmin, vmax=vmax)
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(im, cax=cax)
    fig.savefig(plot_path)
