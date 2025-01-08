# =============================================================================
# The data should be downloaded from:
#     https://drive.google.com/file/d/11gegaQaB94ILQCRL9tj2uoi08YyHJQrN/view?usp=drive_link
# The datafile is named `picture_c_lowfe.csv`.
# The first column of data is the time and the others are Zernike terms 2-24.
# The coefficients are in nm RMS error.
# This script puts the data into the correct format to be run with the
# `control_loop_run` script.
# =============================================================================

import numpy as np

MAX_ROWS = 1000

data = np.loadtxt('picture_c_lowfe.csv', delimiter=',')
print(f'Total rows: {data.shape[0]}')
# Trim the data to the max number of rows.
data = data[:MAX_ROWS]
# The coefficient data is currently in nm, it should be in meters.
data[:, 1:] *= 1e-9
# Create a column with just the index of each row.
idx_col = np.arange(MAX_ROWS)
# Create a column that has the delta time.
delta_time_col = np.full(MAX_ROWS, data[1][0] - data[0][0])
# The columns of the final table
cols = (
    idx_col[:, None],
    data[:, :1],
    delta_time_col[:, None],
    data[:, 1:],
)
# Concat all of the columns together
output_data = np.concatenate(cols, axis=1)
# The column names
column_names = ('ROW_IDX, CUMULATIVE_TIME, DELTA_TIME, Z2, Z3, Z4, Z5, Z6, '
                'Z7, Z8, Z9, Z10, Z11, Z12, Z13, Z14, Z15, Z16, Z17, Z18, '
                'Z19, Z20, Z21, Z22, Z23, Z24')
# Write out the data
np.savetxt(
    f'picc_control_steps_{MAX_ROWS}.csv',
    output_data,
    delimiter=',',
    header=column_names,
    fmt='%.15f',
)
