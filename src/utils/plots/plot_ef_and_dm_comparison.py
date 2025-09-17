import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.constants import PLOT_STYLE_FILE


def plot_ef_and_dm_comparison(
    rows,
    row_identifiers,
    darkhole_mask,
    active_sci_cam_rows,
    active_sci_cam_cols,
    plot_path=None,
):
    """
    Plot three rows of comparisons. Each row will contain the real and imaginary
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
    plot_path : str
        The path to output the plot.
    """

    # Load in the style file
    plt.style.use(PLOT_STYLE_FILE)

    nrows = len(rows)
    # Add one for the intensity plot
    ncols = len(rows[0].keys()) + 1
    fig = plt.figure(figsize=(ncols * 5, nrows * 4))
    ax = fig.subplots(nrows=nrows, ncols=ncols)

    def _add_row(row_idx, identifier, data_dict):

        def _preprocess_sci(key):
            sci = data_dict[key]
            sci[~darkhole_mask] = 0
            sci = sci[:, active_sci_cam_cols]
            sci = sci[active_sci_cam_rows]
            return sci

        sci_r = _preprocess_sci('sci_r')
        sci_i = _preprocess_sci('sci_i')
        intensity = sci_r**2 + sci_i**2

        def _plot_with_cb(col_idx, title, img):
            plot_im = ax[row_idx, col_idx].imshow(img)
            ax[row_idx, col_idx].set_title(f'{identifier}\n{title}',
                                           fontsize=14)
            divider = make_axes_locatable(ax[row_idx, col_idx])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(plot_im, cax=cax, orientation='vertical')

        _plot_with_cb(0, 'Real(EF)', sci_r)
        _plot_with_cb(1, 'Imag(EF)', sci_i)
        _plot_with_cb(2, 'Intensity', intensity)
        _plot_with_cb(3, 'DM1 CMD', data_dict['dm1'])
        if ncols == 5:
            _plot_with_cb(4, 'DM2 CMD', data_dict['dm2'])

    for idx, (row, identifier) in enumerate(zip(rows, row_identifiers)):
        _add_row(idx, identifier, row)

    plt.tight_layout()
    plt.savefig(plot_path)
