import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from utils.constants import PLOT_STYLE_FILE
from utils.stats_and_error import rss


def plot_dm_comparison(
    rows,
    row_identifiers,
    add_first_row_diff_comparison=False,
    fix_colorbars=False,
    round_colorbars=False,
    plot_path=None,
):
    """
    Plot rows of comparisons. Each row will contain the HODM command(s).

    Parameters
    ----------
    rows : list(dict, ...)
        A list of dictionaries containing the keys 'dm1' and 'dm2'.
    row_identifiers : list(str, ...)
        The names of each row.
    add_first_row_diff_comparison : bool
        Add comparisons to each subplot of their difference from the first row.
    fix_colorbars : bool
        Make the colorbars have the same scales in every column.
    round_colorbars : bool
        For every colorbar, ceil the vmax and floor the vmin.
    plot_path : str
        The path to output the plot.
    """

    # Load in the style file
    plt.style.use(PLOT_STYLE_FILE)

    nrows = len(rows)
    ncols = len(rows[0].keys())
    fig = plt.figure(figsize=(ncols * 5, nrows * 4))
    ax = fig.subplots(nrows=nrows, ncols=ncols)

    # Accumulate the bounds for each table
    if fix_colorbars:
        bounds = {}
        for key in rows[0].keys():
            all_vals = [row_data[key] for row_data in rows]
            bounds[key] = (np.min(all_vals), np.max(all_vals))

    for row_idx, (row_data, ident) in enumerate(zip(rows, row_identifiers)):
        col_info = [('DM1 CMD', 'dm1'), ('DM2 CMD', 'dm2')]
        for col_idx, (col_title, key) in enumerate(col_info):
            cell_ax = ax[row_idx, col_idx]
            cell_data = row_data[key]
            vmin = np.min(cell_data)
            vmax = np.max(cell_data)
            if fix_colorbars:
                vmin, vmax = bounds[key]
            if round_colorbars:
                vmin = np.floor(vmin)
                vmax = np.ceil(vmax)
            plot_im = cell_ax.imshow(
                cell_data,
                vmin=vmin,
                vmax=vmax,
            )
            if row_idx == 0:
                cell_ax.set_title(
                    f'{col_title}\n',
                    fontsize=14,
                    fontweight='bold',
                )
            elif add_first_row_diff_comparison:
                rss_val = rss(rows[0][key] - cell_data)
                cell_ax.set_title(
                    f'RSS from {row_identifiers[0]}: {rss_val:0.5f}',
                    fontsize=14,
                )
            if col_idx == 0:
                cell_ax.set_ylabel(
                    f'{ident}\n',
                    fontsize=14,
                    fontweight='bold',
                    rotation='vertical',
                )
            divider = make_axes_locatable(cell_ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(plot_im, cax=cax, orientation='vertical')

    plt.tight_layout()
    plt.savefig(plot_path)
