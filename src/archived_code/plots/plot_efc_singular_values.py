"""
Plot the singular values from SVD done on the EFC matrix.
"""

from h5py import File
import matplotlib.pyplot as plt
import numpy as np

# The path to the HDF datafile
HDF_PATH = '../data/raw/dm1_singular_values_flat/0_data.h5'
# The number of modes to use
NUM_MODES = 500

# Load in the style file
plt.style.use('plot_styling.mplstyle')

fig, ax = plt.subplots(2, 2, figsize=(12, 10), sharex=True)


def _plot(axs_obj, data, title, set_tick_labels=True):
    axs_obj.plot(data)
    axs_obj.axhline(y=data[0], linestyle='--')
    axs_obj.axhline(y=data[-1], linestyle='--')
    axs_obj.set_yticks([data[0], data[-1]])
    axs_obj.set_title(title)
    axs_obj.set_xlabel('Singular Value')
    axs_obj.set_ylabel('Magnitude', labelpad=-30)
    if set_tick_labels:
        axs_obj.set_yticklabels([f'{data[0]:0.3f}', f'{data[-1]:0.3f}'])


singular_values = File(HDF_PATH)['singular_values'][:NUM_MODES]
_plot(ax[0, 0], singular_values, '(A) Original Scale', False)

singular_values = (singular_values - singular_values[-1]) / (
    singular_values[0] - singular_values[-1])
_plot(ax[0, 1], singular_values, '(B) Min-Max Normalization')

singular_values = 0.1 + 0.9 * singular_values
_plot(ax[1, 0], singular_values, '(C) Lower-Bound Scaling')

singular_values /= np.mean(singular_values)
_plot(ax[1, 1], singular_values, '(D) Divide by Mean')

plt.savefig('singular_values_transformation.png')
