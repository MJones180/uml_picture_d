from scripts.analyze_static_wavefront_convergence import analyze_static_wavefront_convergence_parser  # noqa: E501
from scripts.batch_model_test import batch_model_test_parser
from scripts.batch_model_train import batch_model_train_parser
from scripts.benchmark_model import benchmark_model_parser
from scripts.control_loop_run import control_loop_run_parser
from scripts.control_loop_static_wavefronts import control_loop_static_wf_parser
from scripts.convert_analysis_outputs_from_svd_basis import convert_analysis_outputs_from_svd_basis_parser  # noqa: E501
from scripts.convert_dh_rm import convert_dh_rm_parser
from scripts.convert_piccsim_fits_data import convert_piccsim_fits_data_parser
from scripts.convert_picd_instrument_data import convert_picd_instrument_data_parser  # noqa: E501
from scripts.create_response_matrix import create_response_matrix_parser
from scripts.dataset_info import dataset_info_parser
from scripts.dm_comparison import dm_comparison_parser
from scripts.export_model import export_model_parser
from scripts.export_zernike_inputs_to_binary import export_zernike_inputs_to_binary_parser  # noqa: E501
from scripts.gen_zernike_time_steps import gen_zernike_time_steps_parser
from scripts.hdf_file_ops import hdf_file_ops_parser
from scripts.interactive_model_test_plots import inter_model_test_plots_parser
from scripts.model_test import model_test_parser
from scripts.model_train import model_train_parser
from scripts.network_info import network_info_parser
from scripts.plot_model_loss import plot_model_loss_parser
from scripts.preprocess_data_bare import preprocess_data_bare_parser
from scripts.preprocess_data_complete import preprocess_data_complete_parser
from scripts.preprocess_data_dark_hole import preprocess_data_dark_hole_parser  # noqa: E501
from scripts.prune_tag_lookup import prune_tag_lookup_parser
from scripts.prune_trained_model import prune_trained_model_parser
from scripts.random_trim_raw_dataset import random_trim_raw_dataset_parser
from scripts.rank_analysis_dir import rank_analysis_dir_parser
from scripts.run_response_matrix import run_response_matrix_parser
from scripts.sim_data import sim_data_parser

script_parsers = [
    analyze_static_wavefront_convergence_parser,
    batch_model_test_parser,
    batch_model_train_parser,
    benchmark_model_parser,
    control_loop_run_parser,
    control_loop_static_wf_parser,
    convert_analysis_outputs_from_svd_basis_parser,
    convert_dh_rm_parser,
    convert_piccsim_fits_data_parser,
    convert_picd_instrument_data_parser,
    create_response_matrix_parser,
    dataset_info_parser,
    dm_comparison_parser,
    export_model_parser,
    export_zernike_inputs_to_binary_parser,
    gen_zernike_time_steps_parser,
    hdf_file_ops_parser,
    inter_model_test_plots_parser,
    model_test_parser,
    model_train_parser,
    network_info_parser,
    plot_model_loss_parser,
    preprocess_data_bare_parser,
    preprocess_data_complete_parser,
    preprocess_data_dark_hole_parser,
    prune_tag_lookup_parser,
    prune_trained_model_parser,
    random_trim_raw_dataset_parser,
    rank_analysis_dir_parser,
    run_response_matrix_parser,
    sim_data_parser,
]
