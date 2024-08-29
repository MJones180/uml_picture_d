"""
Go through each row in a raw dataset and see if any other rows in the dataset
are similar. Uses `np.isclose` for similarity checks, it defines tolerance as:
    absolute(a - b) <= (atol + rtol * absolute(b))

Commands to run this script:
    python3 main_stnp.py check_for_similar_rows \
        rows_with_gaussian_pert 1e-5 1e-4 80 --save-similar-plots
    python3 main_stnp.py check_for_similar_rows \
        random_10nm_large 1e-5 1e-4 80 \
        --save-similar-plots --max-row-compare 2000
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from prettytable import PrettyTable
from utils.constants import RANDOM_P
from utils.load_raw_sim_data_chunks import load_raw_sim_data_chunks
from utils.path import make_dir
from utils.printing_and_logging import dec_print_indent, step_ri, title


def check_for_similar_rows_parser(subparsers):
    subparser = subparsers.add_parser(
        'check_for_similar_rows',
        help='find similar rows in a dataset',
    )
    subparser.set_defaults(main=check_for_similar_rows)
    subparser.add_argument(
        'dataset',
        help='name of the raw dataset to find similar rows in',
    )
    subparser.add_argument(
        'rtol',
        type=float,
        help='rtol as defined by np.isclose',
    )
    subparser.add_argument(
        'atol',
        type=float,
        help='atol as defined by np.isclose',
    )
    subparser.add_argument(
        'percentage',
        type=int,
        help='percentage of pixels that must be within the tolerance',
    )
    subparser.add_argument(
        '--print-not-similar',
        action='store_true',
        help='print out the percentage of rows that are not similar',
    )
    subparser.add_argument(
        '--save-similar-plots',
        action='store_true',
        help='save plots of all the similar rows',
    )
    subparser.add_argument(
        '--max-row-compare',
        type=int,
        help='max number of rows to compare',
    )


def check_for_similar_rows(cli_args):
    title('Check for similar rows script')

    step_ri('Loading in CLI args')
    dataset = cli_args['dataset']
    rtol = cli_args['rtol']
    atol = cli_args['atol']
    percentage = cli_args['percentage']
    print_not_similar = cli_args['print_not_similar']
    save_similar_plots = cli_args['save_similar_plots']
    max_row_compare = cli_args['max_row_compare']

    step_ri('Loading in the target dataset')
    target_ds_data = load_raw_sim_data_chunks(dataset)
    wavefronts = target_ds_data[0][:]
    zernike_coeffs = target_ds_data[1][:]

    if max_row_compare:
        print(f'Only using first {max_row_compare} rows')
        wavefronts = wavefronts[:max_row_compare]
        zernike_coeffs = zernike_coeffs[:max_row_compare]

    step_ri('Looking for similar rows (will print indexes)')
    total_pixels = np.prod(wavefronts.shape[-2:])
    similar_idxs = []
    # Compare every row to every row. For two rows to be similar, `percentage`
    # of their pixels must be within the tolerance.
    for idx_o, wf_outer in enumerate(wavefronts):
        for idx_i, wf_inner in enumerate(wavefronts[idx_o + 1:]):
            similar_pixels = np.isclose(wf_outer, wf_inner, rtol, atol)
            # .sum() will add up all the similar pixels
            similarity_percentage = (similar_pixels.sum() / total_pixels) * 100
            similarity_percentage_str = f'{similarity_percentage:0.2f}'
            # Need to shift the inner index based on the outer loop
            idx_s = idx_i + idx_o + 1
            if similarity_percentage >= percentage:
                print(f'Similar: {idx_o} and {idx_s} '
                      f'({similarity_percentage_str}%)')
                similar_idxs.append((idx_o, idx_s, similarity_percentage_str))
            elif print_not_similar:
                print(f'NOT Similar: {idx_o} and {idx_s} '
                      f'({similarity_percentage_str}%)')

    step_ri(f'All indexes of similar rows (total {len(similar_idxs)})')
    dec_print_indent()
    table = PrettyTable(['idx1', 'idx2', '%'])
    table.add_rows(similar_idxs)
    print(table)

    if save_similar_plots:
        step_ri('Saving similar plots')
        plot_dir = f'{RANDOM_P}/{dataset}_similar_rows/'
        make_dir(plot_dir)
        for idx1, idx2, percentage in similar_idxs:
            fig, ax = plt.subplots(1, 3, figsize=(16, 8))
            coeffs = [[
                float(f'{(coeff * 1e9):0.2f}') for coeff in zernike_coeffs[idx]
            ] for idx in (idx1, idx2)]
            coeffs_table = PrettyTable(header=False, border=False)
            coeffs_table.align = 'l'
            coeffs_table.add_rows(coeffs)
            plt.suptitle(
                f'Percentage {percentage}%\n'
                f'Coeffs in nm (first row WF {idx1}, second row WF {idx2}):\n'
                f'{coeffs_table.get_string()}',
                fontname='monospace',
                wrap=True,
            )
            ax[0].imshow(wavefronts[idx1])
            ax[0].set_title(f'WF {idx1}')
            ax[1].imshow(wavefronts[idx2])
            ax[1].set_title(f'WF {idx2}')
            im = ax[2].imshow(wavefronts[idx2] - wavefronts[idx1])
            ax[2].set_title('Difference')
            divider = make_axes_locatable(ax[2])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
            fig.tight_layout()
            plot_path = f'{plot_dir}/{idx1}_{idx2}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
