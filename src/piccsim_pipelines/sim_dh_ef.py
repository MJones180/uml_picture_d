# =============================================================================
# Simulate the EF for a DH
# This script should be called from the root of `src/` as
#   python3 piccsim_pipelines/sim_dh_ef.py <tag> <epoch> <testing_ds> [other]
# Examples:
#   # Full DH
#   python3 piccsim_pipelines/sim_dh_ef.py dh_v14 210 \
#       test_dh_both_norm_xl --full-dh
#   # Full DH, 300 SVD modes
#   python3 piccsim_pipelines/sim_dh_ef.py dh_v13 95 \
#       test_dh_both_svd_300_norm --full-dh --svd-mode-count 300
#   # Half DH, 300 SVD modes
#   python3 piccsim_pipelines/sim_dh_ef.py dh_v7_4 91 \
#       test_dh_single_svd_300_norm_xl --svd-mode-count 300
# =============================================================================

# =============================================================================
# Make it so that the imports can be found.
# =============================================================================

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
# Make it so that the packages can be found
sys.path.append(str(Path('.').resolve()))

# =============================================================================
# The imports. Ignore all the E402 :(
# =============================================================================

import argparse  # noqa: E402
import numpy as np  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
from sys import argv  # noqa: E402
from scripts.convert_analysis_outputs_from_svd_basis import convert_analysis_outputs_from_svd_basis  # noqa: E402, E501
from scripts.convert_piccsim_fits_data import convert_piccsim_fits_data  # noqa: E402, E501
from utils.constants import ANALYSIS_P, EF_RECONSTRUCTIONS_P  # noqa: E402
from utils.hdf_read_and_write import read_hdf  # noqa: E402
from utils.load_raw_sim_data import raw_sim_data_chunk_paths  # noqa: E402
from utils.model import Model  # noqa: E402
from utils.path import make_dir  # noqa: E402
from utils.plots.plot_ef_and_dm_comparison import plot_ef_and_dm_comparison  # noqa: E402, E501
from utils.printing_and_logging import step_ri, title  # noqa: E402
from utils.shared_argparser_args import shared_argparser_args  # noqa: E402

if __name__ == '__main__':
    title('The Sim DH EF piccsim script')

    # ==========================================================================
    # The argparser.
    # ==========================================================================

    parser = argparse.ArgumentParser(
        prog='Sim_DH_EF',
        description='simulate a DH EF',
    )
    shared_argparser_args(parser, ['tag', 'epoch'])
    parser.add_argument(
        'testing_ds',
        help='tag of the testing dataset',
    )
    parser.add_argument(
        '--full-dh',
        action='store_true',
        help='use a full dh instead of a half dh; expects two HODMs',
    )
    parser.add_argument(
        '--num-rows',
        type=int,
        default=10,
        help='then number of rows to simulate',
    )
    parser.add_argument(
        '--svd-mode-count',
        type=int,
        help=('the number of SVD modes the outputs are in; if a value is not '
              'passed then the outputs are already in actuator heights'),
    )
    parser.add_argument(
        '--piccsim-path',
        default='/home/michael-jones/Documents/piccsim',
        help='the path to the `piccsim` repo',
    )

    # ==========================================================================
    # Setup the options.
    # ==========================================================================

    step_ri('Using CLI args')
    args_dict = vars(parser.parse_known_args(argv[1:])[0])

    def _grab_and_use_arg(arg):
        val = args_dict[arg]
        print(f'`{arg}`: {val}')
        return val

    model_tag = _grab_and_use_arg('tag')
    model_epoch = _grab_and_use_arg('epoch')
    testing_ds = _grab_and_use_arg('testing_ds')
    full_dh = _grab_and_use_arg('full_dh')
    num_rows = _grab_and_use_arg('num_rows')
    svd_mode_count = _grab_and_use_arg('svd_mode_count')
    piccsim_path = _grab_and_use_arg('piccsim_path')

    step_ri('Other args')
    tag_and_epoch = f'{model_tag}_epoch_{model_epoch}'
    cwd = os.getcwd()
    print(f'`tag_and_epoch`: {tag_and_epoch}')
    print(f'`cwd`: {cwd}')

    # ==========================================================================
    # If the outputs are in terms of SVD modes, then they need to be put in
    # terms of individual actuator heights.
    # ==========================================================================

    # The filename will change if the outputs are in terms of SVD modes
    analysis_filename = 'results'

    if svd_mode_count is not None:
        step_ri('Switching from SVD basis to actuator heights')
        svd_modes_tags = ['hodm1_756_modes']
        svd_modes_table_names = ['dm1_modes']
        if full_dh:
            svd_modes_tags.append('hodm2_756_modes')
            svd_modes_table_names.append('dm2_modes')
        convert_analysis_outputs_from_svd_basis({
            'tag': model_tag,
            'epoch': model_epoch,
            'testing_ds': testing_ds,
            'svd_modes_tags': svd_modes_tags,
            'svd_modes_table_names': svd_modes_table_names,
            'svd_modes_count': svd_mode_count,
        })
        analysis_filename = 'actuator_heights_results'

    # ==========================================================================
    # Export to CSV the rows that are being simulated.
    # ==========================================================================

    step_ri('Exporting rows to CSV')
    analysis_path = f'{ANALYSIS_P}/{testing_ds}/{tag_and_epoch}'
    data = read_hdf(f'{analysis_path}/{analysis_filename}.h5')
    truth_outputs = data['outputs_truth'][:][:num_rows]
    model_outputs = data['outputs_model'][:][:num_rows]
    outputs = np.concatenate((truth_outputs, model_outputs), axis=1)
    exported_csv_path = f'{piccsim_path}/exported_analysis_rows.csv'
    np.savetxt(exported_csv_path, outputs, delimiter=",")

    # ==========================================================================
    # Simulate the EF for each row in `piccsim`.
    # ==========================================================================

    step_ri('Calling `piccsim` for simulations')
    os.chdir(piccsim_path)
    if full_dh:
        idl_run_cmd = '.run batch.reconstruct_ef_full'
    else:
        idl_run_cmd = '.run batch.reconstruct_ef'
    idl_cmd = f'idl -e "{idl_run_cmd}"'
    process = subprocess.Popen(idl_cmd, shell=True, stdout=subprocess.PIPE)
    process.wait()

    # ==========================================================================
    # Convert the `piccsim` datafiles to HDF.
    # ==========================================================================

    step_ri('Converting `piccsim` data to HDF')
    os.chdir(cwd)
    # The list of files and tables that need to be converted to HDF
    prefixes = ['truth', 'cnn', 'matrix']
    file_globs = ['dm1_*', 'sci_*i', 'sci_*r']
    table_names = ['dm1', 'sci_i', 'sci_r']
    if full_dh:
        file_globs.append('dm2_*')
        table_names.append('dm2')
    converted_df_tag = f'dh_piccsim_sim_ef_{tag_and_epoch}'
    convert_piccsim_fits_data({
        'tag': converted_df_tag,
        'dir_path': f'{piccsim_path}/plots/reconstruct_ef',
        'fits_file_globs': [
            f'{prefix}_{file_glob}' for prefix in prefixes
            for file_glob in file_globs
        ],
        'fits_table_names': [
            f'{prefix}_{table_name}' for prefix in prefixes
            for table_name in table_names
        ],
    })

    # ==========================================================================
    # Do the final plots.
    # ==========================================================================

    step_ri('Plotting')
    # Easiest way to load in a lot of the needed data is from the model
    model_extra_vars = Model(model_tag, model_epoch).extra_vars
    darkhole_mask = model_extra_vars['dark_zone_mask'][:]
    active_sci_cam_rows = model_extra_vars['sci_cam_active_row_idxs'][:]
    active_sci_cam_cols = model_extra_vars['sci_cam_active_col_idxs'][:]
    # Load in the data that was just saved above
    data = read_hdf(raw_sim_data_chunk_paths(converted_df_tag)[0])

    # Grab the dict of data associated with each plot
    def _grab_data_dict(idx):
        return [{
            table_name: data[f'{prefix}_{table_name}'][:][idx]
            for table_name in table_names
        } for prefix in prefixes]

    # Create the output plot directory
    out_dir = f'{EF_RECONSTRUCTIONS_P}/{tag_and_epoch}/'
    make_dir(out_dir)
    # Create each plot
    for idx in range(num_rows):
        plot_ef_and_dm_comparison(
            _grab_data_dict(idx),
            ['Truth', f'CNN ({tag_and_epoch})', 'Matrix'],
            darkhole_mask,
            active_sci_cam_rows,
            active_sci_cam_cols,
            add_first_row_diff_comparison=True,
            fix_colorbars=True,
            plot_path=f'{out_dir}/{idx}.png',
        )
