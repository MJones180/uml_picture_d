# ==============================================================================
# Path locations are all relative to the folders one below `src`

DATA_P = '../data'
PROC_DATA_P = f'{DATA_P}/processed'
RAW_DATA_P = f'{DATA_P}/raw'

OUTPUT_P = '../output'
ANALYSIS_P = f'{OUTPUT_P}/analysis'
CONTROL_LOOP_RESULTS_P = f'{OUTPUT_P}/control_loop_results'
CONTROL_LOOP_STEPS_P = f'{OUTPUT_P}/control_loop_steps'
EF_RECONSTRUCTIONS_P = f'{OUTPUT_P}/ef_reconstructions'
RANDOM_P = f'{OUTPUT_P}/random'
RESPONSE_MATRICES_P = f'{OUTPUT_P}/response_matrices'
TRAINED_MODELS_P = f'{OUTPUT_P}/trained_models'

PACKAGES_P = '../packages'

PLOT_STYLE_FILE = 'plot_styling.mplstyle'

ABERRATIONS_F = 'aberrations.csv'
ARGS_F = 'args.json'
BINARY_DATA_F = 'binary_data.bin'
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

INPUTS_ARCSINH = 'inputs_arcsinh'
INPUTS_SUM_TO_ONE = 'inputs_sum_to_one'
INPUT_MIN_X = 'input_min_x'
INPUT_MAX_MIN_DIFF = 'input_max_min_diff'
OUTPUT_MIN_X = 'output_min_x'
OUTPUT_MAX_MIN_DIFF = 'output_max_min_diff'
NORM_RANGE_ONES = 'norm_range_ones'
NORM_RANGE_ONES_INPUT = 'input_norm_range_ones'
NORM_RANGE_ONES_OUTPUT = 'output_norm_range_ones'

# ==============================================================================
# Random HDF table names

BASE_INT_FIELD = 'base_intensity_field'
# Two of the camera constants are still referred to by `ccd` for backwards
# compatability with older datafiles
CAMERA_EF = 'camera_efield'
CAMERA_INTENSITY = 'ccd_intensity'
CAMERA_SAMPLING = 'ccd_sampling'
DARK_ZONE_MASK = 'dark_zone_mask'
DM_ACTIVE_IDXS = lambda idx: f'dm_active_idxs_{idx}'  # noqa: E731
DM_SIZE = lambda idx: f'dm_size_{idx}'  # noqa: E731
EF_ACTIVE_IDXS = 'ef_active_idxs'
FULL_EF = 'full_efield'
FULL_INTENSITY = 'full_intensity'
FULL_SAMPLING = 'full_sampling'
MEAS_ERROR_HISTORY = 'meas_error_history'
PERTURBATION_AMOUNTS = 'perturbation_amounts'
RESPONSE_MATRIX_INV = 'response_matrix_inv'
SCI_CAM_ACTIVE_COL_IDXS = 'sci_cam_active_col_idxs'
SCI_CAM_ACTIVE_ROW_IDXS = 'sci_cam_active_row_idxs'
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

VVC_CHARGE = 6

# ==============================================================================
# Keys that appear in optical train files

OT_BEAM_RATIO = 'BEAM_RATIO'
OT_CAMERA_PIXELS = 'CAMERA_PIXELS'
OT_CAMERA_SAMPLING = 'CAMERA_SAMPLING'
OT_DM_LIST = 'DM_LIST'
OT_INIT_BEAM_D = 'INIT_BEAM_D'
OT_OPTICAL_TRAIN = 'OPTICAL_TRAIN'

# ==============================================================================
# Deformable mirrors

DM_ACTUATOR_HEIGHTS = lambda n: f'dm_actuator_heights_{n}'  # noqa: E731
DM_ACTUATOR_SPACING = 'dm_actuator_spacing'
DM_MASK = 'dm_mask'

# ==============================================================================
# Simulation plotting commands

PLOTTING_LINEAR_INT = 'linear_intensity'
PLOTTING_LINEAR_PHASE = 'linear_phase'
PLOTTING_LINEAR_PHASE_NON0_INT = 'linear_phase_nonzero_intensity'
PLOTTING_LOG_INT = 'log_intensity'
PLOTTING_PATH = 'path'

# ==============================================================================
# Devices

CPU = 'cpu'
CUDA = 'cuda'
MPS = 'mps'

# ==============================================================================
# Camera settings

CAM_BIAS = 'bias'
CAM_BITDEPTH = 'bitdepth'
CAM_DARK_RATE = 'dark_rate'
CAM_FULL_WELL = 'full_well'
CAM_GAIN = 'gain'
CAM_READ_NOISE = 'read_noise'

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
