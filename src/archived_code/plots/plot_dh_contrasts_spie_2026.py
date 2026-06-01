"""
Plot DH contrasts between the the NN and EFC matrix after one iteration.
For the SPIE Astronomical Telescopes + Instrumentation 2026 conference.
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
    'dh_v122_25': {
        'label': 'NN',
        'color': '#00B7FF',
        'marker': '$+$',
        'data': [v * 1e-9 for v in [10, 13, 8, 8, 7.5, 14, 10, 10, 10, 19]],
    },
}

# Load in the style file
plt.style.use('plot_styling.mplstyle')

plt.figure(figsize=(10, 4))
for values in data.values():
    plt.scatter(
        np.arange(10),
        values['data'],
        s=350,
        label=values['label'],
        color=values['color'],
        marker=values['marker'],
    )
plt.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.175))
plt.yscale('log')
plt.ylabel('Contrast', labelpad=-15)
plt.xlabel('DH Realization')
plt.locator_params(axis='x', nbins=10)
plt.title('DH Contrast After First Iteration', pad=40)
plt.savefig('dh_contrasts_spie_2026.png')
