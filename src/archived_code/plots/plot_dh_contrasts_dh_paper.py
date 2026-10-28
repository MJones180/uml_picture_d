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
data = {
    SIMPLE: {
        # dh_v157_1 - first 600 input modes
        NN_CONTRASTS: np.array([
            [7.59e-06, 2.46e-09, 2.06e-10, 1.12e-10, 1.01e-10],
            [8.95e-06, 2.24e-09, 2.01e-10, 1.29e-10, 1.25e-10],
            [9.37e-06, 2.19e-09, 2.62e-10, 1.24e-10, 1.00e-10],
            [9.83e-06, 2.52e-09, 3.06e-10, 1.79e-10, 1.59e-10],
            [1.41e-05, 2.95e-09, 2.34e-10, 1.39e-10, 1.26e-10],
            [9.77e-06, 6.15e-09, 3.05e-10, 1.38e-10, 1.18e-10],
            [1.01e-05, 3.93e-09, 2.83e-10, 1.65e-10, 1.47e-10],
            [9.89e-06, 4.55e-09, 2.66e-10, 1.20e-10, 1.04e-10],
            [1.08e-05, 3.21e-09, 2.88e-10, 1.39e-10, 1.19e-10],
            [1.00e-05, 2.34e-09, 2.25e-10, 1.34e-10, 1.25e-10],
        ]),
        EFC_CONTRASTS: np.array([
            [7.59e-06, 8.95e-08, 2.19e-09, 7.93e-11, 1.53e-11],
            [8.95e-06, 7.34e-08, 1.27e-09, 6.17e-11, 1.86e-11],
            [9.37e-06, 1.03e-07, 1.89e-09, 5.44e-11, 1.13e-11],
            [9.83e-06, 9.76e-08, 1.82e-09, 7.87e-11, 2.30e-11],
            [1.41e-05, 1.20e-07, 1.90e-09, 6.49e-11, 1.79e-11],
            [9.77e-06, 7.65e-08, 1.25e-09, 5.16e-11, 1.70e-11],
            [1.01e-05, 1.04e-07, 1.56e-09, 4.96e-11, 1.78e-11],
            [9.89e-06, 7.24e-08, 1.33e-09, 5.92e-11, 1.53e-11],
            [1.08e-05, 1.03e-07, 1.51e-09, 5.98e-11, 1.89e-11],
            [1.00e-05, 9.57e-08, 1.34e-09, 5.15e-11, 1.68e-11],
        ]),
    },
    REALISTIC: {
        # dh_v156_1 - first 600 input modes
        NN_CONTRASTS: np.array([
            [1.07e-04, 4.69e-07, 3.44e-08, 1.59e-08, 1.39e-08],
            [1.13e-04, 6.95e-07, 3.32e-08, 1.35e-08, 1.30e-08],
            [9.85e-05, 6.83e-07, 6.96e-08, 1.63e-08, 1.36e-08],
            [1.18e-04, 4.92e-07, 5.62e-08, 1.53e-08, 1.33e-08],
            [1.38e-04, 7.28e-07, 1.21e-07, 3.45e-08, 1.98e-08],
            [8.66e-05, 6.35e-07, 7.56e-08, 1.73e-08, 1.24e-08],
            [1.30e-04, 1.25e-06, 4.32e-07, 2.43e-07, 1.19e-07],
            [1.05e-04, 1.28e-06, 1.79e-07, 3.83e-08, 2.00e-08],
            [1.36e-04, 9.27e-07, 5.48e-08, 1.63e-08, 1.55e-08],
            [1.07e-04, 8.84e-07, 1.87e-07, 3.20e-08, 1.61e-08],
        ]),
        EFC_CONTRASTS: np.array([
            [1.07e-04, 2.26e-06, 9.87e-08, 2.07e-08, 1.42e-08],
            [1.13e-04, 3.25e-06, 1.05e-07, 1.82e-08, 1.32e-08],
            [9.85e-05, 2.02e-06, 1.31e-07, 2.34e-08, 1.36e-08],
            [1.18e-04, 4.47e-06, 2.43e-07, 2.89e-08, 1.39e-08],
            [1.38e-04, 5.70e-06, 4.21e-07, 7.29e-08, 2.68e-08],
            [8.66e-05, 6.53e-06, 7.19e-07, 1.06e-07, 2.59e-08],
            [1.30e-04, 2.41e-05, 6.79e-06, 2.86e-06, 1.40e-06],
            [1.05e-04, 1.26e-05, 2.67e-06, 7.07e-07, 2.05e-07],
            [1.36e-04, 4.72e-06, 2.16e-07, 2.62e-08, 1.60e-08],
            [1.07e-04, 1.27e-05, 1.80e-06, 2.83e-07, 5.67e-08],
        ]),
    },
}

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
    ax[subplot_idx].set_xlabel('DH Realization')
    ax[subplot_idx].locator_params(axis='x', nbins=x_points)
plt.suptitle(f'DH Contrast ({n_rows} Realizations)', y=0.90)
plt.tight_layout()
plt.savefig('dh_contrasts_dh_paper.png')
