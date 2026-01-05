"""
This script plots a comparison between actual and predicted DM commands.
"""

import numpy as np
from utils.constants import (ANALYSIS_P, DM_ACTIVE_COL_IDXS, DM_ACTIVE_IDXS,
                             DM_ACTIVE_ROW_IDXS, DM_COMPARISONS_P,
                             EXTRA_VARS_F, PROC_DATA_P, RESULTS_F)
from utils.hdf_read_and_write import read_hdf
from utils.path import delete_dir, get_abs_path, make_dir
from utils.plots.plot_dm_comparison import plot_dm_comparison
from utils.printing_and_logging import step_ri, title
from utils.shared_argparser_args import shared_argparser_args
from utils.terminate_with_message import terminate_with_message


def dm_comparison_parser(subparsers):
    subparser = subparsers.add_parser(
        'dm_comparison',
        help='plot comparison between actual and predicted DM commands',
    )
    subparser.set_defaults(main=dm_comparison)
    shared_argparser_args(subparser, ['tag', 'epoch'])
    subparser.add_argument(
        'testing_ds',
        help='name of the testing dataset to compare the outputs from',
    )
    subparser.add_argument(
        'rows_to_plot',
        type=int,
        help='the number of rows to plot',
    )
    subparser.add_argument(
        'dm_size',
        type=int,
        help=('number of pixels per row on the DM; it is expected that both '
              'DMs are square and are the same size'),
    )
    subparser.add_argument(
        '--output-shape-correct',
        action='store_true',
        help=('the DM command output by the model is already in the correct '
              'shape: 2 channels of a 2D pixel grid'),
    )


def dm_comparison(cli_args):
    title('DM comparison script')

    tag = cli_args['tag']
    epoch = cli_args['epoch']
    model_str = f'{tag}_epoch_{epoch}'

    step_ri('Creating the output directory')
    testing_ds_tag = cli_args['testing_ds']
    output_path = f'{DM_COMPARISONS_P}/{testing_ds_tag}/{model_str}'
    output_path = get_abs_path(output_path)
    print(f'Output path: {output_path}')
    delete_dir(output_path, quiet=True)
    make_dir(output_path)

    step_ri('Loading the results data')
    testing_ds_tag = cli_args['testing_ds']
    results_path = f'{ANALYSIS_P}/{testing_ds_tag}/{model_str}/{RESULTS_F}'
    results_path = get_abs_path(results_path)
    results_data = read_hdf(results_path)
    outputs_model = results_data['outputs_model'][:]
    outputs_truth = results_data['outputs_truth'][:]
    print(f'Model output shape: {outputs_model.shape}')
    print(f'Truth output shape: {outputs_truth.shape}')

    step_ri('Taking only requested rows')
    rows_to_plot = cli_args['rows_to_plot']
    print(f'Using first {rows_to_plot} rows')
    outputs_model = outputs_model[:rows_to_plot]
    outputs_truth = outputs_truth[:rows_to_plot]
    print(f'Model output shape: {outputs_model.shape}')
    print(f'Truth output shape: {outputs_truth.shape}')

    step_ri('Loading in the extra variables data')
    extra_vars_path = f'{PROC_DATA_P}/{testing_ds_tag}/{EXTRA_VARS_F}'
    extra_vars_data = read_hdf(extra_vars_path)

    if cli_args['output_shape_correct']:
        step_ri('Grabbing active regions on the DM')

        def grab_dm_active_idxs(idx):
            return (extra_vars_data[DM_ACTIVE_ROW_IDXS(idx)][:],
                    extra_vars_data[DM_ACTIVE_COL_IDXS(idx)][:])

        def _prep_dm_cmd(row_data, dm_idx):
            row_idxs, col_idxs = grab_dm_active_idxs(dm_idx)
            values_2d = row_data[dm_idx]
            dm_cmd = values_2d[row_idxs]
            dm_cmd = dm_cmd[:, col_idxs]
            return dm_cmd

    else:
        step_ri('Grabbing active actuators on the DM')

        def grab_dm_active_idxs(idx):
            return extra_vars_data[DM_ACTIVE_ROW_IDXS(idx)][:]

        step_ri('Validating data')
        active_actuators = grab_dm_active_idxs(0).shape[0]
        if active_actuators != grab_dm_active_idxs(1).shape[0]:
            terminate_with_message('DMs must be the same size')
        elif 2 * active_actuators != outputs_model.shape[1]:
            terminate_with_message(
                'Model outputs have the wrong number of actuators')
        elif 2 * active_actuators != outputs_truth.shape[1]:
            terminate_with_message(
                'Truth outputs have the wrong number of actuators')

        def _prep_dm_cmd(row_data, dm_idx):
            if dm_idx == 0:
                values_1d = row_data[:active_actuators]
            else:
                values_1d = row_data[active_actuators:]
            dm_cmd = np.zeros(dm_size**2)
            dm_cmd[grab_dm_active_idxs(dm_idx)] = values_1d
            return dm_cmd.reshape(dm_size, dm_size)

    step_ri('DM specs')
    dm_size = cli_args['dm_size']
    print(f'Actuators per row/col: {dm_size}')
    print(f'Active actuators: {active_actuators}')

    step_ri('Generating comparison plots')

    def _plot_row(outputs):
        return {
            'dm1': _prep_dm_cmd(outputs, 0),
            'dm2': _prep_dm_cmd(outputs, 1),
        }

    for idx in range(rows_to_plot):
        plot_dm_comparison(
            [
                _plot_row(outputs_truth[idx]),
                _plot_row(outputs_model[idx]),
                _plot_row(outputs_truth[idx] - outputs_model[idx]),
            ],
            ['Truth', model_str, 'Difference'],
            fix_colorbars=True,
            plot_path=f'{output_path}/{idx}.png',
        )
