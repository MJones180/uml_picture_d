# ==============================================================================
# Path locations are all relative to the folders one below `src`

DATA_P = '../data'
PROC_DATA_P = f'{DATA_P}/processed'
RAW_SIMULATED_DATA_P = f'{DATA_P}/raw_simulated'

OUTPUT_P = '../output'
ANALYSIS_P = f'{OUTPUT_P}/analysis'
CONTROL_LOOP_STEPS_P = f'{OUTPUT_P}/control_loop_steps'
RANDOM_P = f'{OUTPUT_P}/random'
RESPONSE_MATRICES_P = f'{OUTPUT_P}/response_matrices'
TRAINED_MODELS_P = f'{OUTPUT_P}/trained_models'

PACKAGES_P = '../packages'

ABERRATIONS_F = 'aberrations.csv'
ARGS_F = 'args.json'
EPOCH_LOSS_F = 'epoch_loss.csv'
EXTRA_VARS_F = 'extra_variables.h5'
TAG_LOOKUP_F = 'tag_lookup.json'
DATA_F = 'data.h5'
RESULTS_F = 'results.h5'

# ==============================================================================
# `src` folder names

NETWORKS = 'networks'
SIM_OPTICAL_TRAINS = 'sim_optical_trains'

# ==============================================================================
# Dataset HDF tables (belongs to the `DATA_F` file)

INPUTS = 'inputs'
OUTPUTS = 'outputs'

# ==============================================================================
# Normalization values (belongs to the `EXTRA_VARS_F` file)

INPUT_MIN_X = 'input_min_x'
INPUT_MAX_MIN_DIFF = 'input_max_min_diff'
OUTPUT_MIN_X = 'output_min_x'
OUTPUT_MAX_MIN_DIFF = 'output_max_min_diff'
NORM_RANGE_ONES = 'norm_range_ones'

# ==============================================================================
# Random HDF table names

BASE_INT_FIELD = 'base_intensity_field'
# Both of the camera constants are still referred to by `ccd` for backwards
# compatability with older datafiles
CAMERA_INTENSITY = 'ccd_intensity'
CAMERA_SAMPLING = 'ccd_sampling'
FULL_INTENSITY = 'full_intensity'
FULL_SAMPLING = 'full_sampling'
MEAS_ERROR_HISTORY = 'MEAS_error_history'
PERTURBATION_AMOUNT = 'perturbation_amount'
RESPONSE_MATRIX_INV = 'response_matrix_inv'
TRUE_ERROR_HISTORY = 'true_error_history'
ZERNIKE_COEFFS = 'zernike_coeffs'
ZERNIKE_TERMS = 'zernike_terms'

# ==============================================================================
# Error

MAE = 'MAE'
MSE = 'MSE'
RMSE = 'RMSE'

# ==============================================================================
# Constants for the different available loss and optimizers functions.
# Each value should correspond to the function's name in PyTorch.

# nn.<loss_function>
LOSS_FUNCTIONS = {
    'mae': 'L1Loss',
    'mse': 'MSELoss',
}
# Optimizers are currently restricted to the learning rate (`lr`) parameter.
# torch.optim.<optimizer_function>
OPTIMIZERS = {
    'adagrad': 'Adagrad',
    'adam': 'Adam',
    'nadam': 'NAdam',
    'rmsprop': 'RMSprop',
    'sgd': 'SGD',
}

# ==============================================================================
# PROPER simulation code constants

# Vector Vortex Coronagraph charge
VVC_CHARGE = 6

# ==============================================================================
# Zernike polynomial names

ZERNIKE_NAME_LOOKUP = {
    1: 'Piston',
    2: 'Tilt X',
    3: 'Tilt Y',
    4: 'Power',
    5: 'Astig 1',
    6: 'Astig 2',
    7: 'Coma 1',
    8: 'Coma 2',
    9: 'Trefoil 1',
    10: 'Trefoil 2',
    11: 'Spherical',
    12: '2nd Astig 1',
    13: '2nd Astrig 2',
    14: 'Tetrafoil 1',
    15: 'Tetrafoil 2',
    16: '2nd Coma 1',
    17: '2nd Coma 2',
    18: '2nd Trefoil 1',
    19: '2nd Trefoil 2',
    20: 'Pentafoil 1',
    21: 'Pentafoil 2',
    22: '2nd Spherical',
    23: '3rd Astig 1',
    24: '3rd Astig 2',
}
