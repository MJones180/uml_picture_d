"""
Plot DH contrasts between the the NN and EFC matrix after one iteration.
For the Spring 2026 Graduate Committee Checkin.
"""

import matplotlib.pyplot as plt
import numpy as np

data = {
    'resp_mat': {
        'label': 'EFC',
        'color': '#43AA8B',
        'marker': '*',
        'data': [
            1.1e-7, 8.4e-8, 8.0e-8, 9.4e-8, 1.0e-7, 1.2e-7, 8.9e-8, 1.2e-7,
            1.1e-7, 9.8e-8
        ],
    },
    'exported_dh_v107_10_epoch642': {
        'label': 'NN',
        'color': '#00B7FF',
        'marker': '$+$',
        'data': [
            v * 1e-8
            for v in [0.95, 0.89, 0.67, 0.58, 0.76, 0.79, 0.57, 1, 0.59, 1.1]
        ],
    },
}

# Load in the style file
plt.style.use('plot_styling.mplstyle')

plt.figure(figsize=(10, 6))
for values in data.values():
    plt.scatter(
        np.arange(10),
        values['data'],
        s=400,
        label=values['label'],
        color=values['color'],
        marker=values['marker'],
    )
plt.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.12))
plt.xlabel('DH Realization')
plt.locator_params(axis='x', nbins=10)
plt.ylabel('Contrast')
plt.title('DH Contrast After First Iteration', pad=35)
plt.yscale('log')
plt.tight_layout()
plt.show()
