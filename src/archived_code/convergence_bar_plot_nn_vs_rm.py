"""
Plot the NN vs RM for cases converged as outputted by the
`analyze_static_wavefront_convergence` script.
"""

import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# CONFIG VALUES
# ==============================================================================

ITERATIONS = 20
WAVEFRONTS_PER_BAR = 5000
GAIN_VALUES = (-0.5, 0, 0)
RANGES = (500, 20, 10)
SCALE_FACTORS = [1, 0.5, 0.25, 0.125, 0.0625]
# Scaling 1: (500, 20, 10)
# Scaling 0.5: (250, 10, 5)
# Scaling 0.25: (125, 5, 2.5)
# Scaling 0.125: (62.5, 2.5, 1.25)
# Scaling 0.0625: (31.25, 1.25, 0.625)
Y_NN = [4950, 4995, 4999, 5000, 5000]  # Dud values need to be updated
Y_RM = [1, 10, 500, 1000, 5000]  # Dud values need to be updated

# ==============================================================================

# Create the subplot
fig, ax = plt.subplots(1, 1, figsize=(10, 5))


def _display_bars(y_values, offset, color, label):
    bar = ax.bar(
        np.arange(len(y_values)) + offset,
        y_values,
        width=0.4,
        align='center',
        label=label,
        color=color,
        edgecolor='#00364D',
        linewidth=2,
    )
    # Add the percentage above every bar
    # https://stackoverflow.com/a/40491960
    for rect in bar:
        height = rect.get_height()
        percentage = height / WAVEFRONTS_PER_BAR * 100
        plt.text(
            rect.get_x() + rect.get_width() / 2.0,
            height,
            f'{percentage:.02f}%',
            ha='center',
            va='bottom',
        )


# Display the bars for both the NN and RM
_display_bars(Y_NN, -0.2, '#43AA8B', 'Neural Network')
_display_bars(Y_RM, 0.2, '#00B7FF', 'Respone Matrix')

# Set the limits on the y axis along with the correct ticks
ax.set_ylim([0, WAVEFRONTS_PER_BAR * 1.06])
ax.set_yticks(np.linspace(0, WAVEFRONTS_PER_BAR, 5))

# The scale factor values should be the tick labels
ax.set_xticklabels([None, *SCALE_FACTORS])

# Set the titles and axis labels
plt.title(
    f'Convergence for {ITERATIONS} Iteration Control Loops '
    'on Static Aberration Wavefronts\nGain Values $K_{P,I,D}$ = ' +
    str(GAIN_VALUES),
    pad=30,
)
ax.set_xlabel(
    'Scaling Factor ($k$)\nAberration Ranges Are Uniformly Random Between: '
    r'$k\cdot\;[-x_Z, x_Z]$ nm RMS error where $x_{2-3}=$' + str(RANGES[0]) +
    '$, x_{4-8}=$' + str(RANGES[1]) + '$, x_{9-24}=$' + str(RANGES[2]))
ax.set_ylabel('Static Wavefronts Converged')

# https://stackoverflow.com/a/43439132
plt.legend(
    bbox_to_anchor=(0, 1.02, 1, 0.2),
    loc='lower center',
    borderaxespad=0,
    ncol=2,
)

plt.tight_layout()
plt.show()
