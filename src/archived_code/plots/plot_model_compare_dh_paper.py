"""
Plot output MAE between two models. For the SPIE DH NN paper.
"""

from utils.constants import ANALYSIS_P, RESULTS_F
from utils.hdf_read_and_write import read_hdf
from utils.model import Model
from utils.plots.plot_coeff_comparison import plot_coeff_comparison
from utils.stats_and_error import mae

SIMPLE = 'simple'
REALISTIC = 'realistic'

MODEL_NAMES = {
    SIMPLE: 'dh_v158_1',
    REALISTIC: 'dh_v158_2',
}

RESULT_FOLDERS = {
    SIMPLE: 'test_dh_both_hodms_efc_final_dh_all_iter_600_600_v2/dh_v158_1_epoch_1213',  # NOQA E501
    REALISTIC: 'test_dh_both_hodms_efc_final_dh_broadband_4iter_600_600_v1/dh_v158_2_epoch_1232',  # NOQA E501
}


def _load_model_and_results(model_key):
    model = Model(MODEL_NAMES[model_key], 'last')
    results = read_hdf(f'{ANALYSIS_P}/{RESULT_FOLDERS[model_key]}/{RESULTS_F}')
    return model, results


simple_nn, simple_results = _load_model_and_results(SIMPLE)
realistic_nn, realistic_results = _load_model_and_results(REALISTIC)


def _compute_mae(results_data):
    results_truth = results_data['outputs_truth'][:]
    results_model = results_data['outputs_model'][:]
    return mae(results_truth, results_model, axes=0)


simple_mae = _compute_mae(simple_results)
realistic_mae = _compute_mae(realistic_results)
plot_coeff_comparison(
    [len(simple_mae)],
    simple_mae,
    realistic_mae,
    'Simple NN',
    'Realistic NN',
    'model_mae_compare.png',
    title='Output Coefficient MAE',
)
