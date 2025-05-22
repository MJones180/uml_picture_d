"""
Plot the NN vs RM for cases converged as outputted by the
`analyze_static_wavefront_convergence` script.
Plot for the paper:
    MS Thesis (older version of script)
    Adaptive Optics Wavefront Stabilization Using a Convolutional Neural Network
"""

import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# CONFIG VALUES
# ==============================================================================

ITERATIONS = 20
WAVEFRONTS_PER_BAR = 5000
RANGES = (500, 20, 10)
THRESHOLD = 1e-9
GAIN_VALUES = (-0.5, 0, 0)
# NN: weighted_aberration_ranges_local_v4_last
# RM: fixed_40nm_positive_and_negative
# SCALING   Z2-3  Z4-8  Z9-24 |   NN    RM
#  4        2000    80     40 |  395     2
#  3.5      1750    70     35 | 1014     4
#  3        1500    60     30 | 2076    14
#  2.5      1250    50     25 | 3626    19
#  2        1000    40     20 | 4886    66
#  1.5       750    30     15 | 5000   124
#  1.25      625    25   12.5 | 5000   183
#  1         500    20     10 | 5000   291
#  0.5       250    10      5 | 5000  1192
#  0.25      125     5    2.5 | 5000  4448
#  0.125    62.5   2.5   1.25 | 5000  5000
#  0.0625  31.25  1.25  0.625 | 5000  5000
SCALE_FACTORS = [4, 3, 2, 1, 0.5, 0.25, 0.125]
Y_NN = [395, 2076, 4886, 5000, 5000, 5000, 5000]
Y_RM = [2, 14, 66, 291, 1192, 4448, 5000]

# ==============================================================================

# Load in the style file
plt.style.use('plot_styling.mplstyle')

# Create the subplot
fig, ax = plt.subplots(figsize=(20, 8))


def _display_bars(y_values, offset, color, label):
    bars = ax.bar(
        np.arange(len(y_values)) + offset,
        y_values,
        width=0.4,
        align='center',
        label=label,
        color=color,
        edgecolor='#00364D',
    )
    # Add the percentage above every bar
    # https://stackoverflow.com/a/40491960
    for bar in bars:
        bar_height = bar.get_height()
        percentage = bar_height / WAVEFRONTS_PER_BAR * 100
        # Don't display the decimal places for 100%
        if percentage == 100:
            percentage_str = f'{percentage:.0f}'
        else:
            percentage_str = f'{percentage:.02f}'
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar_height + (-650 if percentage > 20 else 100),
            f'{percentage_str}%\n({bar_height})',
            ha='center',
            va='bottom',
        )


# Display the bars for both the NN and RM
_display_bars(Y_NN, -0.2, '#FFAD99', 'Convolutional Neural Network')
_display_bars(Y_RM, 0.2, '#99C2FF', 'Response Matrix')

# Set the limits on the y axis along with the correct ticks
ax.set_ylim([0, WAVEFRONTS_PER_BAR * 1.035])
ax.set_yticks(np.linspace(0, WAVEFRONTS_PER_BAR, 5))

# The margin inside the actual plotting area needs to be smaller
ax.margins(0.01)

# The scale factor values should be the tick labels
ax.set_xticklabels([None, *SCALE_FACTORS])

# Set the titles and axis labels
plt.title(
    f'{ITERATIONS} Iteration Control Loops on Static Aberration '
    'Input Signals\nGain Factors $K_{P,I,D}$ = ' + str(GAIN_VALUES),
    pad=40,
)
ax.set_xlabel('Input Aberration Scaling Factor, $k$\n'
              'Aberrations Uniformly Random Between: '
              r'$k\;[-x_Z, x_Z]$ nm RMS error, where $x_{2-3}=$' +
              str(RANGES[0]) + '$, x_{4-8}=$' + str(RANGES[1]) +
              '$, x_{9-24}=$' + str(RANGES[2]))
ax.set_ylabel('Wavefronts Flattened\nTrue Error Threshold: '
              f'[-{THRESHOLD}, {THRESHOLD}]')

# https://stackoverflow.com/a/43439132
plt.legend(
    bbox_to_anchor=(0, 1.02, 1, 0.2),
    loc='lower center',
    borderaxespad=0,
    ncol=2,
)

plt.tight_layout()
plt.savefig('convergence_plot.png')
