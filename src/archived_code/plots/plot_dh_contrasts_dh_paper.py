"""
Plot DH contrasts between the the NN and EFC matrix after one iteration.
For the SPIE DH NN paper.
"""

import matplotlib.pyplot as plt
import numpy as np

data = {
    'resp_mat': {
        'label': 'EFC',
        'color': '#43AA8B',
        'marker': '*',
        'data': [
            v * 1e-8 for v in [11, 8.4, 8, 9.4, 10, 12, 8.9, 12, 11, 9.8]
        ],
    },
    'dh_v146_2': {
        'label': 'NN',
        'color': '#00B7FF',
        'marker': '$+$',
        'data': [
            v * 1e-9 for v in [3.9, 3, 2.9, 2.7, 2.8, 3.9, 2.7, 3.5, 2.5, 6.8]
        ],
    },
}

# Load in the style file
plt.style.use('plot_styling.mplstyle')


def math_notation(value):
    coeff, exp = f'{value:.3e}'.split('e')
    return rf'${coeff} \times 10^{{{int(exp)}}}$'


fig, ax = plt.subplots(figsize=(10, 4))
for values in data.values():
    # Compute the geometric mean; instead of calculating prod(x)^(1/N), the
    # following computational trick can be used
    mean_value = 10**np.mean(np.log10(values['data']))
    ax.scatter(
        np.arange(10),
        values['data'],
        s=350,
        label=values['label'] + f' (Mean {math_notation(mean_value)})',
        color=values['color'],
        marker=values['marker'],
    )
    ax.axhline(y=mean_value, linestyle='--', color=values['color'])

ax.set_title('DH Contrast After First Iteration', pad=40)
ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.175))
ax.set_yscale('log')
ax.set_ylabel('Contrast')
ax.set_xlabel('DH Realization')
ax.locator_params(axis='x', nbins=10)

plt.savefig('dh_contrasts_dh_paper.png')
