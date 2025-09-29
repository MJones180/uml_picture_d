import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from utils.constants import PLOT_STYLE_FILE
from utils.stats_and_error import rss


def plot_ef_and_dm_comparison(
    rows,
    row_identifiers,
    darkhole_mask,
    active_sci_cam_rows,
    active_sci_cam_cols,
    add_first_row_diff_comparison=False,
    fix_colorbars=False,
    plot_path=None,
):
    """
    Plot rows of comparisons. Each row will contain the real and imaginary
    parts of the EF, the intensity, and the HODM command(s).

    Parameters
    ----------
    rows : list(dict, ...)
        A list of dictionaries containing the keys 'sci_i', 'sci_r', and 'dm1';
        the key 'dm2' is optional but must be consistent.
    row_identifiers : list(str, ...)
        The names of each row.
    darkhole_mask : np.array
        The active pixels of the dark hole; should be of type bool.
    active_sci_cam_rows : np.array
        The active rows on the science camera.
    active_sci_cam_cols : np.array
        The active columns on the science camera.
    add_first_row_diff_comparison : bool
        Add comparisons to each subplot of their difference from the first row.
    fix_colorbars : bool
        Make the colorbars have the same scales in every column.
    plot_path : str
        The path to output the plot.
    """

    # Load in the style file
    plt.style.use(PLOT_STYLE_FILE)

    nrows = len(rows)
    # Add one to account for the intensity plot
    ncols = len(rows[0].keys()) + 1
    fig = plt.figure(figsize=(ncols * 5, nrows * 4))
    ax = fig.subplots(nrows=nrows, ncols=ncols)

    # Preprocess the data in each row
    for row_data in rows:
        for key in ('sci_r', 'sci_i'):
            sci = row_data[key]
            sci[~darkhole_mask] = 0
            sci = sci[:, active_sci_cam_cols]
            sci = sci[active_sci_cam_rows]
            row_data[key] = sci
        row_data['intensity'] = row_data['sci_r']**2 + row_data['sci_i']**2

    # Accumulate the bounds for each table
    if fix_colorbars:
        bounds = {}
        for key in rows[0].keys():
            all_vals = [row_data[key] for row_data in rows]
            bounds[key] = (np.min(all_vals), np.max(all_vals))

    for row_idx, (row_data, ident) in enumerate(zip(rows, row_identifiers)):
        col_info = [('Real(EF)', 'sci_r'), ('Imag(EF)', 'sci_i'),
                    ('Intensity', 'intensity'), ('DM1 CMD', 'dm1')]
        if ncols == 5:
            col_info.append(('DM2 CMD', 'dm2'))
        for col_idx, (col_title, key) in enumerate(col_info):
            cell_ax = ax[row_idx, col_idx]
            cell_data = row_data[key]
            vmin = np.min(cell_data)
            vmax = np.max(cell_data)
            if fix_colorbars:
                vmin, vmax = bounds[key]
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
