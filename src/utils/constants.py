# Path locations are all relative to the folders one below `src`

DATA_P = '../data'
PROC_DATA_P = f'{DATA_P}/processed'
PROPER_SIM_DATA_P = f'{DATA_P}/proper_sim'
RAW_FITS_DATA_P = f'{DATA_P}/raw_fits'

OUTPUT_P = '../output'
ANALYSIS_P = f'{OUTPUT_P}/analysis'
RANDOM_P = f'{OUTPUT_P}/random'
TRAINED_MODELS_P = f'{OUTPUT_P}/trained_models'

PACKAGES_P = '../packages'

ARGS_F = 'args.json'
NORM_F = 'norm.json'
TAG_LOOKUP_F = 'tag_lookup.json'
DATA_F = 'data.h5'
RESULTS_F = 'results.h5'

# `src` folder names

NETWORKS = 'networks'
SIM_OPTICAL_TRAINS = 'sim_optical_trains'

# Dataset HDF tables (belongs to the `DATA_F` file)

INPUTS = 'inputs'
OUTPUTS = 'outputs'

# Normalization values (belongs to the `NORM_F` file)

INPUT_MIN_X = 'input_min_x'
INPUT_MAX_MIN_DIFF = 'input_max_min_diff'
OUTPUT_MIN_X = 'output_min_x'
OUTPUT_MAX_MIN_DIFF = 'output_max_min_diff'

# Error

MAE = 'MAE'
MSE = 'RMSE'
RMSE = 'RMSE'

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
    'rmsprop': 'RMSprop',
    'sgd': 'SGD',
}

# PROPER simulation code constants

# Vector Vortex Coronagraph charge
VVC_CHARGE = 6
