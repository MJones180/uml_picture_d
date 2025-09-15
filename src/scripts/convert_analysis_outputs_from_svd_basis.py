"""
Takes in an analysis file and converts the outputs from the SVD basis to
actuator heights. Updates the `outputs_model` and `outputs_truth` tables.
If there are two HODMs, it is expected that the HODM 2 outputs are stacked
on top of the HODM 1 outputs.
"""

import numpy as np
from utils.constants import ANALYSIS_P, RESULTS_F
from utils.hdf_read_and_write import HDFWriteModule, read_hdf
from utils.load_raw_sim_data import raw_sim_data_chunk_paths
from utils.model import Model
from utils.printing_and_logging import step_ri, title
from utils.shared_argparser_args import shared_argparser_args
from utils.terminate_with_message import terminate_with_message


def convert_analysis_outputs_from_svd_basis_parser(subparsers):
    subparser = subparsers.add_parser(
        'convert_analysis_outputs_from_svd_basis',
        help='convert analysis outputs from the SVD basis to actuator heights',
    )
    subparser.set_defaults(main=convert_analysis_outputs_from_svd_basis)
    shared_argparser_args(subparser, ['tag', 'epoch'])
    subparser.add_argument(
        'testing_ds',
        help='tag of the testing dataset',
    )
    subparser.add_argument(
        '--svd-modes-tags',
        nargs='*',
        help=('tags of the raw datasets containing the SVD modes; one tag '
              'represents HODM 1, two tags represent HODMs 1 and 2; must be '
              'a single chunk of data for each tag'),
    )
    subparser.add_argument(
        '--svd-modes-table-names',
        nargs='*',
        help=('name of each table that corresponds each tag listed in '
              'the `--svd-modes-tags` argument'),
    )
    subparser.add_argument(
        '--svd-modes-count',
        type=int,
        help=('the number of SVD modes to use from the beginning; '
              'applies to each HODM tag'),
    )


def convert_analysis_outputs_from_svd_basis(cli_args):
    title('Convert analysis outputs from svd basis script')

    step_ri('Grabbing the analysis data path')
    # Grab the model tag and epoch
    tag = cli_args['tag']
    epoch = cli_args['epoch']
    model = Model(tag, epoch)
    # This is needed when the epoch is called with 'last'
    epoch = model.epoch
    testing_ds_tag = cli_args['testing_ds']
    analysis_path = f'{ANALYSIS_P}/{testing_ds_tag}/{tag}_epoch_{epoch}'
    print(f'Using path {analysis_path}')

    step_ri('Loading the analysis data')
    analysis_data = read_hdf(f'{analysis_path}/{RESULTS_F}')

    def _load_and_print(table):
        table_data = analysis_data[table][:]
        print(f'{table}: {table_data.shape}')
        return table_data

    model_outputs = _load_and_print('outputs_model')
    truth_outputs = _load_and_print('outputs_truth')

    step_ri('Loading the SVD modes data')
    svd_modes_tags = cli_args['svd_modes_tags']
    svd_modes_table_names = cli_args['svd_modes_table_names']
    svd_modes_count = cli_args['svd_modes_count']

    if svd_modes_tags is None:
        terminate_with_message('The SVD modes tag(s) must be passed')
    if svd_modes_table_names is None:
        terminate_with_message('The SVD modes table name(s) must be passed')
    if svd_modes_count is None:
        terminate_with_message('The SVD modes count must be passed')

    svd_modes_tag_count = len(svd_modes_tags)
    if svd_modes_tag_count > 2:
        terminate_with_message('A max of two SVD modes tags can be passed')

    def _load_modes(idx):
        modes_tag = svd_modes_tags[idx]
        table_name = svd_modes_table_names[idx]
        print(f'Using {modes_tag} for HODM {idx + 1}')
        modes_path = raw_sim_data_chunk_paths(modes_tag)[0]
        print(f'Loading modes from {modes_path}')
        modes = read_hdf(modes_path)[table_name][:].astype(np.float32)
        # Flatten the modes
        modes = modes.reshape(modes.shape[0], -1)
        # Use the number of given modes
        modes = modes[:svd_modes_count]
        print(f'Using first {svd_modes_count} modes from {table_name}')
        # Remove the inactive pixels
        nonzero_actuators = (modes != 0).any(axis=0)
        active_idxs = np.where(nonzero_actuators)[0]
        modes = modes[:, active_idxs]
        print(f'Modes shape: {modes.shape}')
        return modes

    modes1 = _load_modes(0)
    if svd_modes_tag_count == 2:
        modes2 = _load_modes(1)

    step_ri('Converting from the SVD basis')

    def _convert_from_svd(outputs, var_str):
        print(f'[Old] {var_str}: {outputs.shape}')
        if svd_modes_tag_count == 2:
            outputs = np.concatenate(
                ((outputs[:, :svd_modes_count] @ modes1),
                 (outputs[:, svd_modes_count:] @ modes2)),
                axis=1,
            )
        else:
            outputs = outputs @ modes1
        print(f'[New] {var_str}: {outputs.shape}')
        return outputs

    model_outputs = _convert_from_svd(model_outputs, 'outputs_model')
    truth_outputs = _convert_from_svd(truth_outputs, 'outputs_truth')

    step_ri('Writing out the converted output data')
    out_file_path = f'{analysis_path}/actuator_heights_{RESULTS_F}'
    print(f'File location: {out_file_path}')
    HDFWriteModule(out_file_path).create_and_write_hdf_simple({
        'outputs_truth': truth_outputs,
        'outputs_model': model_outputs,
    })
