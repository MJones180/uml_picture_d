"""
Plot DH contrasts between the the NN and EFC matrix after one iteration.
For the SPIE DH NN paper.
"""

import matplotlib.pyplot as plt
import numpy as np

SIMPLE = 'Simple'
REALISTIC = 'Realistic'
NN_CONTRASTS = 'nn_contrasts'
EFC_CONTRASTS = 'efc_contrasts'

# In the contrast data: rows are realizations, columns are iterations
# yapf: disable; needed to allow long lines
data = {
    SIMPLE: {
        # dh_v158_1
        NN_CONTRASTS: np.array([
            [7.59e-06, 3.77e-09, 2.15e-10, 1.06e-10, 9.90e-11, 9.93e-11, 1.00e-10, 1.01e-10, 1.02e-10, 1.03e-10],
            [8.95e-06, 2.08e-09, 2.01e-10, 1.33e-10, 1.28e-10, 1.30e-10, 1.33e-10, 1.36e-10, 1.40e-10, 1.43e-10],
            [9.37e-06, 2.32e-09, 1.81e-10, 1.07e-10, 1.02e-10, 1.04e-10, 1.06e-10, 1.08e-10, 1.11e-10, 1.13e-10],
            [9.83e-06, 2.62e-09, 2.41e-10, 1.61e-10, 1.55e-10, 1.55e-10, 1.56e-10, 1.57e-10, 1.57e-10, 1.58e-10],
            [1.41e-05, 4.35e-09, 2.16e-10, 1.30e-10, 1.27e-10, 1.27e-10, 1.29e-10, 1.31e-10, 1.33e-10, 1.36e-10],
            [9.77e-06, 4.71e-09, 2.64e-10, 1.32e-10, 1.18e-10, 1.17e-10, 1.18e-10, 1.19e-10, 1.20e-10, 1.21e-10],
            [1.01e-05, 2.41e-09, 2.25e-10, 1.54e-10, 1.51e-10, 1.53e-10, 1.55e-10, 1.57e-10, 1.59e-10, 1.61e-10],
            [9.89e-06, 5.03e-09, 2.51e-10, 1.13e-10, 1.07e-10, 1.08e-10, 1.09e-10, 1.11e-10, 1.13e-10, 1.14e-10],
            [1.08e-05, 3.97e-09, 2.36e-10, 1.30e-10, 1.21e-10, 1.21e-10, 1.22e-10, 1.24e-10, 1.26e-10, 1.28e-10],
            [1.00e-05, 2.29e-09, 1.84e-10, 1.33e-10, 1.31e-10, 1.33e-10, 1.36e-10, 1.40e-10, 1.44e-10, 1.48e-10],
        ]),
        EFC_CONTRASTS: np.array([
            [7.59e-06, 8.95e-08, 2.19e-09, 7.93e-11, 1.53e-11, 1.09e-11, 9.41e-12, 8.37e-12, 7.59e-12, 6.99e-12],
            [8.95e-06, 7.34e-08, 1.27e-09, 6.17e-11, 1.86e-11, 1.38e-11, 1.16e-11, 1.02e-11, 9.15e-12, 8.37e-12],
            [9.37e-06, 1.03e-07, 1.89e-09, 5.44e-11, 1.13e-11, 8.62e-12, 7.47e-12, 6.75e-12, 6.24e-12, 5.86e-12],
            [9.83e-06, 9.76e-08, 1.82e-09, 7.87e-11, 2.30e-11, 1.70e-11, 1.41e-11, 1.20e-11, 1.05e-11, 9.42e-12],
            [1.41e-05, 1.20e-07, 1.90e-09, 6.49e-11, 1.79e-11, 1.31e-11, 1.11e-11, 9.81e-12, 8.85e-12, 8.12e-12],
            [9.77e-06, 7.65e-08, 1.25e-09, 5.16e-11, 1.70e-11, 1.31e-11, 1.14e-11, 1.02e-11, 9.30e-12, 8.62e-12],
            [1.01e-05, 1.04e-07, 1.56e-09, 4.96e-11, 1.78e-11, 1.39e-11, 1.19e-11, 1.05e-11, 9.49e-12, 8.72e-12],
            [9.89e-06, 7.24e-08, 1.33e-09, 5.92e-11, 1.53e-11, 1.11e-11, 9.34e-12, 8.19e-12, 7.34e-12, 6.70e-12],
            [1.08e-05, 1.03e-07, 1.51e-09, 5.98e-11, 1.89e-11, 1.38e-11, 1.13e-11, 9.74e-12, 8.62e-12, 7.81e-12],
            [1.00e-05, 9.57e-08, 1.34e-09, 5.15e-11, 1.68e-11, 1.33e-11, 1.17e-11, 1.05e-11, 9.60e-12, 8.89e-12],
        ]),
    },
    REALISTIC: {
        # dh_v158_2
        NN_CONTRASTS: np.array([
            [1.07e-04, 4.36e-07, 3.84e-08, 1.61e-08, 1.37e-08, 1.35e-08, 1.35e-08, 1.35e-08, 1.35e-08, 1.36e-08],
            [1.13e-04, 8.38e-07, 3.56e-08, 1.36e-08, 1.28e-08, 1.28e-08, 1.28e-08, 1.28e-08, 1.29e-08, 1.29e-08],
            [9.85e-05, 5.69e-07, 5.33e-08, 1.40e-08, 1.28e-08, 1.26e-08, 1.26e-08, 1.27e-08, 1.27e-08, 1.28e-08],
            [1.18e-04, 7.52e-07, 3.80e-08, 1.38e-08, 1.31e-08, 1.31e-08, 1.31e-08, 1.32e-08, 1.32e-08, 1.33e-08],
            [1.38e-04, 9.94e-07, 9.11e-08, 2.33e-08, 1.56e-08, 1.33e-08, 1.28e-08, 1.26e-08, 1.26e-08, 1.26e-08],
            [8.66e-05, 6.22e-07, 7.70e-08, 1.66e-08, 1.22e-08, 1.16e-08, 1.15e-08, 1.14e-08, 1.15e-08, 1.15e-08],
            [1.30e-04, 1.27e-06, 4.07e-07, 1.84e-07, 9.26e-08, 6.74e-08, 4.63e-08, 3.08e-08, 2.21e-08, 1.83e-08],
            [1.05e-04, 1.13e-06, 3.02e-07, 5.18e-08, 2.57e-08, 1.74e-08, 1.51e-08, 1.44e-08, 1.41e-08, 1.40e-08],
            [1.36e-04, 5.63e-07, 6.45e-08, 1.65e-08, 1.51e-08, 1.49e-08, 1.49e-08, 1.49e-08, 1.50e-08, 1.50e-08],
            [1.07e-04, 7.17e-07, 1.09e-07, 2.38e-08, 1.45e-08, 1.36e-08, 1.36e-08, 1.36e-08, 1.36e-08, 1.36e-08],
        ]),
        EFC_CONTRASTS: np.array([
            [1.07e-04, 2.26e-06, 9.87e-08, 2.07e-08, 1.42e-08, 1.35e-08, 1.34e-08, 1.34e-08, 1.35e-08, 1.35e-08],
            [1.13e-04, 3.25e-06, 1.05e-07, 1.82e-08, 1.32e-08, 1.29e-08, 1.29e-08, 1.29e-08, 1.30e-08, 1.30e-08],
            [9.85e-05, 2.02e-06, 1.31e-07, 2.34e-08, 1.36e-08, 1.23e-08, 1.21e-08, 1.21e-08, 1.21e-08, 1.21e-08],
            [1.18e-04, 4.47e-06, 2.43e-07, 2.89e-08, 1.39e-08, 1.26e-08, 1.26e-08, 1.26e-08, 1.26e-08, 1.26e-08],
            [1.38e-04, 5.70e-06, 4.21e-07, 7.29e-08, 2.68e-08, 1.60e-08, 1.32e-08, 1.23e-08, 1.20e-08, 1.19e-08],
            [8.66e-05, 6.53e-06, 7.19e-07, 1.06e-07, 2.59e-08, 1.38e-08, 1.19e-08, 1.13e-08, 1.11e-08, 1.10e-08],
            [1.30e-04, 2.41e-05, 6.79e-06, 2.86e-06, 1.40e-06, 7.53e-07, 4.35e-07, 2.49e-07, 1.54e-07, 9.71e-08],
            [1.05e-04, 1.26e-05, 2.67e-06, 7.07e-07, 2.05e-07, 7.64e-08, 3.43e-08, 2.07e-08, 1.67e-08, 1.51e-08],
            [1.36e-04, 4.72e-06, 2.16e-07, 2.62e-08, 1.60e-08, 1.51e-08, 1.50e-08, 1.51e-08, 1.52e-08, 1.52e-08],
            [1.07e-04, 1.27e-05, 1.80e-06, 2.83e-07, 5.67e-08, 2.10e-08, 1.46e-08, 1.30e-08, 1.25e-08, 1.24e-08],
        ]),
    },
}
# yapf: enable

plt.style.use('plot_styling.mplstyle')
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
for subplot_idx, subplot_key in enumerate((SIMPLE, REALISTIC)):
    data_dict = data[subplot_key]
    for contrast_idx, contrast_key in enumerate((NN_CONTRASTS, EFC_CONTRASTS)):
        color = 'red' if contrast_idx else 'blue'
        title = 'EFC' if contrast_idx else 'NN'
        contrasts = data_dict[contrast_key]
        n_rows, x_points = contrasts.shape
        x_idxs = np.arange(contrasts.shape[1])
        ax[subplot_idx].fill_between(
            x_idxs,
            np.min(contrasts, axis=0),
            np.max(contrasts, axis=0),
            color=color,
            alpha=0.2,
            linewidth=0,
            label=title,
        )
        # Compute the geometric mean; instead of calculating prod(x)^(1/N),
        # the following computational trick can be used
        mean_contrasts = 10**np.mean(np.log10(contrasts), axis=0)
        ax[subplot_idx].plot(x_idxs, mean_contrasts, color=color)
    ax[subplot_idx].set_title(subplot_key)
    ax[subplot_idx].legend()
    ax[subplot_idx].set_yscale('log')
    ax[subplot_idx].set_ylabel('Contrast')
    ax[subplot_idx].set_xlabel('Iteration')
    ax[subplot_idx].locator_params(axis='x', nbins=x_points)
plt.suptitle(f'DH Contrast ({n_rows} Realizations)', y=0.90)
plt.tight_layout()
plt.savefig('dh_contrasts_dh_paper.png')
