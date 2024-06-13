"""
Rank all the model tests in the analysis directory.
"""

from glob import glob
import numpy as np
from prettytable import PrettyTable
from utils.constants import ANALYSIS_P, MAE, MSE, RESULTS_F
from utils.hdf_read_and_write import read_hdf
from utils.printing_and_logging import dec_print_indent, step_ri, title


def rank_analysis_dir_parser(subparsers):
    """
    Example commands:
        python3 main.py rank_analysis_dir fixed_50nm_range_processed \
            --ds-on-fixed-grid --r-min-filter 0.4
    """
    subparser = subparsers.add_parser(
        'rank_analysis_dir',
        help='rank all model analyses',
    )
    subparser.set_defaults(main=rank_analysis_dir)
    subparser.add_argument(
        'testing_ds',
        help='name of the testing dataset',
    )
    subparser.add_argument(
        '--filter',
        help='partial string to filter analysis directories by',
    )
    subparser.add_argument(
        '--use-mae',
        action='store_true',
        help='use MAE instead of MSE',
    )
    subparser.add_argument(
        '--ds-on-fixed-grid',
        action='store_true',
        help=('the data is generated along a fixed grid by using the '
              '`--fixed-amount-per-zernike-range` arg in `sim_data`'),
    )
    subparser.add_argument(
        '--r-min-filter',
        type=float,
        help='hide all rows with an r min value below this',
    )


def rank_analysis_dir(cli_args):
    title('Model test script')

    step_ri('Error function')
    error = 'mae' if cli_args['use_mae'] else 'mse'
    print(f'Using {error.upper()}')

    step_ri('Grabbing all potential analysis directories')
    testing_ds = cli_args['testing_ds']
    filter_str = cli_args['filter']
    if filter_str:
        print(f'Applying the filter `{filter_str}`')
    else:
        filter_str = ''
    dir_paths = glob(f'{ANALYSIS_P}/{testing_ds}/*{filter_str}*/{RESULTS_F}')
    print(f'Found a total of {len(dir_paths)}')

    ds_on_fixed_grid = cli_args['ds_on_fixed_grid']
    r_min_filter = cli_args['r_min_filter']

    step_ri('Looping through and grabbing HDF files')
    results = []
    for dir_path in dir_paths:
        data = read_hdf(dir_path)
        outputs_truth = data['outputs_truth'][:]
        # The predictions table could have two different names
        outputs_pred = data.get('outputs_model',
                                data.get('outputs_response_matrix'))
        # The number of Zernike terms
        zernike_count = outputs_truth.shape[1]
        # Produces a 2nx2n matrix where n is the number of Zernike terms.
        # Along the main diagonal in the first and third quadrants (they are
        # symmatrix) are how correlated the predictions are to the truth values
        # for each Zernike term.
        res = np.corrcoef(outputs_pred, outputs_truth, rowvar=False)
        correlations = np.diagonal(res[zernike_count:][:, :zernike_count])
        r_min = np.min(correlations)
        # Filter out rows that have too low of an r_min
        if r_min_filter is not None and r_min < r_min_filter:
            continue
        # Results that will be displayed
        res_to_append = {
            'identifier': dir_path.split('/')[-2],
            'mae': data[MAE][()],
            'mse': data[MSE][()],
            'r_min': r_min,
        }
        # Compute the correlation for each Zernike term by itself, this
        # represents the data from within the `zernike_response` plots
        if ds_on_fixed_grid:
            res_to_append['r_zr_min'] = np.min([
                np.corrcoef(
                    outputs_truth[idx::zernike_count, idx],
                    outputs_pred[idx::zernike_count, idx],
                )[0][1] for idx in range(zernike_count)
            ])
        results.append(res_to_append)

    step_ri('Sorting the rankings')
    # This will give us a sorted array consisting of tuples (error, idx)
    sorted_results = sorted(
        [(idx, values[error]) for idx, values in enumerate(results)],
        key=lambda pair: pair[1],
    )
    sorted_idxs = [idx for idx, val in sorted_results]

    step_ri('Rankings')
    dec_print_indent()
    col_names = ['Identifier', 'MAE', 'MSE', 'r min']
    if ds_on_fixed_grid:
        col_names.append('r_zr min')
    table = PrettyTable(col_names)
    table.align = 'l'
    for row_idx in sorted_idxs:
        row_data = results[row_idx]

        def _format_error(key):
            return f'{row_data[key]:0.3e}'

        def _format_r(key):
            return f'{row_data[key]:0.3f}'

        row_data_formatted = [
            row_data['identifier'],
            _format_error('mae'),
            _format_error('mse'),
            _format_r('r_min'),
        ]
        if ds_on_fixed_grid:
            row_data_formatted.append(_format_r('r_zr_min'))
        table.add_row(row_data_formatted)
    print(table)
