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
            v * 1e-8 for v in [11, 8.4, 8, 9.4, 10, 12, 8.9, 12, 11, 9.8]
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


def math_notation(value):
    coeff, exp = f'{value:.3e}'.split('e')
    return rf'${coeff} \times 10^{{{int(exp)}}}$'


fig, ax1 = plt.subplots(figsize=(10, 4))
mean_values = []
for values in data.values():
    # Compute the geometric mean; instead of calculating prod(x)^(1/N), the
    # following computational trick can be used
    mean_value = 10**np.mean(np.log10(values['data']))
    mean_values.append(mean_value)
    ax1.scatter(
        np.arange(10),
        values['data'],
        s=350,
        label=values['label'],
        color=values['color'],
        marker=values['marker'],
    )
    ax1.axhline(y=mean_value, linestyle='--', color=values['color'])

ax1.set_title('DH Contrast After First Iteration', pad=40)
ax1.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.175))
ax1.set_yscale('log')
ax1.set_ylabel('Contrast', labelpad=-18)
ax1.set_xlabel('DH Realization')
ax1.locator_params(axis='x', nbins=10)

ax2 = ax1.twinx()
ax2.set_ylim(ax1.get_ylim())
ax2.set_yscale('log')
ax2.set_ylabel('Mean\nContrast', labelpad=-40)
ax2.set_yticks(mean_values)
ax2.set_yticklabels([math_notation(val) for val in mean_values])
ax2.minorticks_off()

plt.savefig('dh_contrasts_spie_2026.png')
