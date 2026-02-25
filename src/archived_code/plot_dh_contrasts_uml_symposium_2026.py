"""
Plot DH contrasts between the different NN models after one iteration
for the UML Symposium 2026 poster.
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
    'exported_dh_v84_1_epoch122': {
        'label': 'CNN 1',
        'color': '#2079C7',
        'marker': '$-$',
        'data': [
            v * 1e-8 for v in [4.5, 5.7, 4, 4.7, 6.1, 5, 5.5, 6, 4.6, 7.3]
        ],
    },
    'exported_dh_v85_1_epoch184': {
        'label': 'CNN 2',
        'color': '#1D4E89',
        'marker': '$×$',
        'data': [
            v * 1e-8 for v in [4.5, 5.9, 4.7, 5.1, 5.9, 5.3, 4.7, 6.2, 4, 6.7]
        ],
    },
    'exported_dh_v86_1_epoch346': {
        'label': 'NN',
        'color': '#00B7FF',
        'marker': '$+$',
        'data': [
            v * 1e-8 for v in [2.4, 4, 4.4, 4.1, 4.8, 4.8, 5.9, 4.2, 4, 4.5]
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
plt.tight_layout()
plt.savefig('DH_Contrast_Plot_UML_Symposium_2026.png', dpi=600)
