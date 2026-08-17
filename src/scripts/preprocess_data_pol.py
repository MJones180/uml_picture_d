"""
This script was adapted from the `preprocess_data_dark_hole` script.
This script is designed for the polarization NNs (input: focal plane PSFs
in both polarizations; output: EF being estimated).
"""

import numpy as np
from utils.cli_args import save_cli_args
from utils.constants import (DATA_F, EXTRA_VARS_F, INPUTS, MEAN, OUTPUTS,
                             PROC_DATA_P)
from utils.group_data_from_list import group_data_from_list
from utils.hdf_read_and_write import HDFWriteModule, read_hdf
from utils.load_raw_sim_data import raw_sim_data_chunk_paths
from utils.path import make_dir, path_exists
from utils.printing_and_logging import step_ri, title
from utils.stats_and_error import mse
from utils.terminate_with_message import terminate_with_message


def preprocess_data_pol_parser(subparsers):
    subparser = subparsers.add_parser(
        'preprocess_data_pol',
        help='preprocess pol data in to datasets',
    )
    subparser.set_defaults(main=preprocess_data_pol)
    subparser.add_argument(
        '--output-tags',
        nargs='+',
        help='tags of the datasets to create',
    )
    subparser.add_argument(
        '--output-tag-percentages',
        type=int,
        nargs='+',
        help=('percentage of the data each output tag will receive; must be '
              'in the same order as the `--output-tags` argument'),
    )
    subparser.add_argument(
        '--raw-data-tags',
        nargs='*',
        help='tags of the raw simulated data to preprocess and merge together',
    )
    subparser.add_argument(
        '--tables-to-load',
        nargs='+',
        help='tables to load in and preprocess',
    )
    subparser.add_argument(
        '--apply-mask',
        nargs='+',
        help=('apply a mask to the data; arguments expected: name of the raw '
              'datafile containing the mask, table in the datafile, *tables '
              'to apply the mask to'),
    )
    subparser.add_argument(
        '--merge-tables',
        nargs='*',
        help=('merge two tables into a single table; three arguments '
              'expected per group: table 1 name, table 2 name, merged name'),
    )
    subparser.add_argument(
        '--switch-basis',
        nargs='*',
        help=('switch to a different basis; four arguments expected per '
              'group: name of table to transform, raw data tag containing '
              'modes, name of table in modes file, number of modes to use'),
    )
    subparser.add_argument(
        '--input-tables',
        nargs='*',
        help='tables to join together at the end into the input',
    )
    subparser.add_argument(
        '--output-tables',
        nargs='*',
        help='tables to join together at the end into the output',
    )
    subparser.add_argument(
        '--disable-shuffle',
        action='store_true',
        help='do not shuffle the rows',
    )
    subparser.add_argument(
        '--extend-existing-preprocessed-data',
        action='store_true',
        help=('add more data chunks to existing preprocessed datasets; the '
              'data tags must match existing ones for this arg to be used; '
              'the extra variables from the first output tag will be used '
              'and will not be written out a second time'),
    )
    subparser.add_argument(
        '--fix-seed',
        type=int,
        help='fix the seed value for reproducible results',
    )


def preprocess_data_pol(cli_args):
    title('Preprocess data pol script')

    fix_seed = cli_args['fix_seed']
    if fix_seed:
        step_ri('Fixing the seed')
        print(f'Seed value: {fix_seed}')
        np.random.seed(fix_seed)

    # ==========================================================================

    step_ri('Getting output tags ready')
    output_tags = cli_args['output_tags']
    print(f'Output tags: {output_tags}')
    output_tag_paths = [f'{PROC_DATA_P}/{tag}' for tag in output_tags]
    extend_existing_data = cli_args['extend_existing_preprocessed_data']
    if extend_existing_data:
        print('Will extend existing data')
        # Verifying all the datasets exist
        for output_tag_path in output_tag_paths:
            if not path_exists(output_tag_path):
                terminate_with_message(f'{output_tag_path} must exist')
        # Loading in the extra variables so they can just be used
        extra_vars = read_hdf(f'{output_tag_paths[0]}/{EXTRA_VARS_F}')

        def _use_var(var, scalar=False):
            print(f'Using {var} from `extra_vars`')
            if scalar:
                return extra_vars[var][()]
            return extra_vars[var][:]
    else:
        print('Setting up dataset outputs')
        for output_tag_path in output_tag_paths:
            print(f'Making {output_tag_path}')
            make_dir(output_tag_path)
            # Write out the CLI args that this script was called with
            save_cli_args(output_tag_path, cli_args, 'preprocess_data_pol')
        # Extra variables that may be written out at the end
        extra_vars = {}

        def _save_var(arg, val):
            extra_vars[arg] = val
            print(f'Will save `{arg}` at the end')

    # ==========================================================================

    step_ri('Loading in data chunks')

    tables = cli_args['tables_to_load']
    print(f'Tables: {tables}')
    all_table_data = {table: [] for table in tables}
    for tag in cli_args['raw_data_tags']:
        for data_path in raw_sim_data_chunk_paths(tag):
            print(f'Loading in data from {data_path}')
            data = read_hdf(data_path)
            for table in tables:
                all_table_data[table].extend(data[table][:].astype(np.float32))

    # ==========================================================================

    step_ri('Converting loaded data to numpy arrays')
    for table, table_data in all_table_data.items():
        all_table_data[table] = np.asarray(table_data)
        print(f'{table} shape: {all_table_data[table].shape}')

    # ==========================================================================

    apply_mask = cli_args.get('apply_mask')
    if apply_mask is not None:
        step_ri('Applying mask to the data')
        mask_tag, mask_table, *tables_to_mask = apply_mask
        mask_path = raw_sim_data_chunk_paths(mask_tag)[0]
        print(f'Loading in the mask from {mask_path}')
        print(f'Mask table: {mask_table}')
        mask_data = read_hdf(mask_path)[mask_table][:]
        print(f'Mask shape: {mask_data.shape}')
        print(f'This mask has {mask_data.sum()} active pixels')
        print(f'Will apply mask to tables: {tables_to_mask}')
        for table in tables_to_mask:
            all_table_data[table][..., ~mask_data] = 0

    # ==========================================================================

    step_ri('Flattening data and removing all inactive pixels')
    for table, table_data in all_table_data.items():
        table_mask_name = f'{table}_mask'
        if extend_existing_data:
            table_mask = _use_var(table_mask_name)
        else:
            # Create a list of nonzero pixels; use only first row for efficiency
            table_mask = table_data[0] != 0
            _save_var(table_mask_name, table_mask)
        orig_shape = table_data.shape
        all_table_data[table] = table_data[..., table_mask]
        print(f'{table} shape: {orig_shape} -> {all_table_data[table].shape}')

    # ==========================================================================

    merge_tables = cli_args.get('merge_tables')
    if merge_tables is not None:
        step_ri('Merging table data together')
        for table1, table2, table_new in group_data_from_list(merge_tables, 3):
            print(f'Table 1: {table1}')
            print(f'Table 2: {table2}')
            print(f'New Merged Table: {table_new}')
            all_table_data[table_new] = np.concatenate(
                (all_table_data.pop(table1), all_table_data.pop(table2)),
                axis=-1,
            )
            print(f'{table_new} shape: {all_table_data[table_new].shape}')

    # ==========================================================================

    switch_basis = cli_args.get('switch_basis')
    if switch_basis is not None:
        step_ri('Switching basis')
        for group in group_data_from_list(switch_basis, 4):
            table, modes_tag, modes_table, number_modes = group
            print(f'Table to transform: {table}')
            print(f'Modes tag: {modes_tag}')
            print(f'Modes table name: {modes_table}')
            print(f'Max modes: {number_modes}')
            modes_data = read_hdf(raw_sim_data_chunk_paths(modes_tag)[0])
            modes = modes_data[modes_table][:].astype(np.float32)
            # Pick out the correct number of modes from the start
            modes = modes[:int(number_modes)]
            table_data = all_table_data[table]
            # Mean center the data if the table exists in the modes datafile
            if MEAN in list(modes_data):
                mean_name = f'{modes_tag}_{MEAN}'
                if extend_existing_data:
                    mean_values = _use_var(mean_name)
                else:
                    mean_values = modes_data[MEAN][:].astype(np.float32)
                    _save_var(mean_name, mean_values)
                table_data -= mean_values
            # Invert the modes
            modes_inv = np.linalg.pinv(modes)
            # The coefficients in the new basis
            new_basis_coeffs = table_data @ modes_inv
            # What the data look like in the new basis
            reconstructed_values = new_basis_coeffs @ modes
            # The error when switching to the new basis representation
            error = mse(table_data, reconstructed_values)
            print(f'{table} reconstruction MSE error of {error:0.3e}')
            all_table_data[table] = new_basis_coeffs
            print(f'{table} shape: {all_table_data[table].shape}')
            print('-----')

    # ==========================================================================

    def _create_merged_arrs(tables_to_merge):
        print(f'Tables that will be merged: {tables_to_merge}')
        merged_data = None
        for table_to_merge in tables_to_merge:
            table_data = all_table_data.pop(table_to_merge)
            if merged_data is None:
                merged_data = table_data
            else:
                merged_data = np.hstack((merged_data, table_data))
        return merged_data

    step_ri('Creating input array')
    input_data = _create_merged_arrs(cli_args['input_tables'])
    print(f'Input shape: {input_data.shape}')

    step_ri('Creating output array')
    output_data = _create_merged_arrs(cli_args['output_tables'])
    print(f'Output shape: {output_data.shape}')

    # ==========================================================================

    if not cli_args['disable_shuffle']:
        step_ri('Shuffling')
        random_shuffle_idxs = np.random.permutation(len(input_data))
        input_data = input_data[random_shuffle_idxs]
        output_data = output_data[random_shuffle_idxs]

    # ==========================================================================

    step_ri('Splitting and writing data')
    output_tag_percentages = cli_args['output_tag_percentages']
    print(f'Output percentages: {output_tag_percentages}')
    if np.sum(output_tag_percentages) != 100:
        terminate_with_message('Percentages must add up to 100%')
    rows_low = 0
    for idx, output_tag in enumerate(output_tags):
        percentage = output_tag_percentages[idx]
        rows = int(len(input_data) * percentage / 100)
        print(f'[{output_tag}] Rows: {rows} ({percentage}%)')
        # Grab the data that should be written to this tag
        input_chunk = input_data[rows_low:rows_low + rows]
        output_chunk = output_data[rows_low:rows_low + rows]
        rows_low += rows
        out_path = output_tag_paths[idx]
        datafile_path = f'{out_path}/{DATA_F}'
        print(f'Writing out to {datafile_path}')
        if extend_existing_data:
            print('Merging existing data in')
            # Add on to the existing datafiles
            with read_hdf(datafile_path) as existing_data:
                input_chunk = np.vstack(
                    (existing_data[INPUTS][:], input_chunk))
                output_chunk = np.vstack(
                    (existing_data[OUTPUTS][:], output_chunk))
        else:
            # Add a file with extra necessary variables
            file_path = f'{out_path}/{EXTRA_VARS_F}'
            HDFWriteModule(file_path).create_and_write_hdf_simple(extra_vars)
        # Write out the processed HDF file
        HDFWriteModule(datafile_path).create_and_write_hdf_simple({
            INPUTS: input_chunk,
            OUTPUTS: output_chunk,
        })
        print(f'Input shape: {input_chunk.shape}')
        print(f'Output shape: {output_chunk.shape}')
