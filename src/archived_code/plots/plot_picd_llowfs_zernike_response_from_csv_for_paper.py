# ==============================================================================
# PICTURE-D hardware results which show the single Zernike response for the
# JATIS paper.
# To run this script, first move it to the `src/` directory.
# ==============================================================================

import numpy as np
from utils.plots.paper_plots.total_crosstalk import paper_plot_total_crosstalk
from utils.plots.plot_zernike_response import plot_zernike_response

BASE_PATH = '/Users/mjones180/Downloads'
TRUTH_COEFF_PATH = f'{BASE_PATH}/llowfs_20250801_181116_truth.csv'
CNN_MODEL_COEFF_PATH = f'{BASE_PATH}/llowfs_20250801_181116_cnn_model.csv'
RM_MODEL_COEFF_PATH = f'{BASE_PATH}/llowfs_20250801_181116_rm_model.csv'
# The zernike terms in the data
ZERNIKE_TERMS = np.arange(2, 24)
THRESHOLD = 50e-9


def _load_and_format_data(path):
    print(f'Loaded in from {path}')
    # Convert from nm surface error to meters wavefront error
    data = np.loadtxt(path, delimiter=',') * 1e-9 * 2
    rows_per_gr = data.shape[0] // data.shape[1]
    print(f'{rows_per_gr} rows per group')
    # Group together the data into the shape of (poke, zernikes, zernikes)
    return np.array([data[idx::rows_per_gr] for idx in range(rows_per_gr)])


truth_data = _load_and_format_data(TRUTH_COEFF_PATH)
cnn_data = _load_and_format_data(CNN_MODEL_COEFF_PATH)
rm_data = _load_and_format_data(RM_MODEL_COEFF_PATH)

# The perturbations for each row
poke_grid = truth_data[:, 0, 0]

# Use only values between the given threshold
mask = (poke_grid < THRESHOLD) & (poke_grid > -THRESHOLD)
poke_grid = poke_grid[mask]
truth_data = truth_data[mask]
cnn_data = cnn_data[mask]
rm_data = rm_data[mask]

paper_plot_total_crosstalk(
    ZERNIKE_TERMS,
    poke_grid,
    cnn_data,
    'Faster CNN',
    'cnn_paper_total_cross_coupling.png',
)
paper_plot_total_crosstalk(
    ZERNIKE_TERMS,
    poke_grid,
    rm_data,
    'RM',
    'rm_paper_total_cross_coupling.png',
)

plot_zernike_response(
    ZERNIKE_TERMS,
    poke_grid,
    cnn_data,
    'Faster CNN',
    plot_path='cnn_zernike_response.png',
)
plot_zernike_response(
    ZERNIKE_TERMS,
    poke_grid,
    rm_data,
    'RM',
    plot_path='rm_zernike_response.png',
)
