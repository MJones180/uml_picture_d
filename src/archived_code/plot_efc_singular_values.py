"""
Plot the singular values from SVD done on the EFC matrix.
The file is located at `piccsim/output/svd_modes/dm1_w_matrix.fits`.
"""

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

FITS_PATH = '~/Downloads/dm1_w_matrix.fits'
# The number of modes to use
NUM_MODES = 500

with fits.open(FITS_PATH) as hdul:
    singular_values = hdul['PRIMARY'].data

singular_values = singular_values[:NUM_MODES]
# Normalize to have a max of 1
singular_values /= np.max(singular_values)

x_points = np.arange(NUM_MODES)

exp_scaling = 0.99**x_points
linear_scaling = np.linspace(1, 0.1, 500)

plt.plot(x_points, singular_values, label='singular values')
plt.plot(x_points, exp_scaling, label='exp (0.99) scaling')
plt.plot(x_points, linear_scaling, label='linear [1, 0.1] scaling')
plt.xlabel('Mode')
plt.ylabel('Value')
plt.legend()
plt.show()
