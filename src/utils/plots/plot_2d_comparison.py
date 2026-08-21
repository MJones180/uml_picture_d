import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from utils.constants import PLOT_STYLE_FILE


def plot_2d_comparison(
    data,
    row_identifiers,
    col_identifiers,
    fix_colorbars=False,
    plot_path=None,
):
    """
    Plot comparisons.

    Parameters
    ----------
    data : np.array(np.array, ...)
        A 4D array of the form: (rows, cols, img_x, img_y).
    row_identifiers : list(str, ...)
        The names of each row.
    col_identifiers : list(str, ...)
        The names of each col.
    fix_colorbars : bool
        Make the colorbars have the same scales in every column.
    plot_path : str
        The path to output the plot.
    """

    # Load in the style file
    plt.style.use(PLOT_STYLE_FILE)

    nrows, ncols, _, _ = data.shape

    fig = plt.figure(figsize=(ncols * 5, nrows * 4))
    ax = fig.subplots(nrows=nrows, ncols=ncols)

    # Accumulate the bounds for each column
    if fix_colorbars:
        col_mins = [None for _ in range(ncols)]
        col_maxs = [None for _ in range(ncols)]
        for row_idx, row in enumerate(data):
            for col_idx, col_cell in enumerate(row):
                cell_min = np.min(col_cell)
                cell_max = np.max(col_cell)
                if row_idx == 0 or cell_min < col_mins[col_idx]:
                    col_mins[col_idx] = cell_min
                if row_idx == 0 or cell_max > col_maxs[col_idx]:
                    col_maxs[col_idx] = cell_max

    for row_idx, row in enumerate(data):
        for col_idx, col_cell in enumerate(row):
            cell_ax = ax[row_idx, col_idx]
            vmin = np.min(col_cell)
            vmax = np.max(col_cell)
            if fix_colorbars:
                vmin = col_mins[col_idx]
                vmax = col_maxs[col_idx]
            plot_im = cell_ax.imshow(
                col_cell,
                vmin=vmin,
                vmax=vmax,
            )
            if row_idx == 0:
                cell_ax.set_title(
                    f'{col_identifiers[col_idx]}\n',
                    fontsize=14,
                    fontweight='bold',
                )
            if col_idx == 0:
                cell_ax.set_ylabel(
                    f'{row_identifiers[row_idx]}\n',
                    fontsize=14,
                    fontweight='bold',
                    rotation='vertical',
                )
            divider = make_axes_locatable(cell_ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(plot_im, cax=cax, orientation='vertical')

    plt.tight_layout()
    plt.savefig(plot_path)
