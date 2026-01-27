"""
Plot the Faster CNN vs Preferred CNN vs RM for cases converged as outputted by
the `analyze_static_wavefront_convergence` script.
Plot for the paper: "Adaptive Optics Wavefront Capture and Stabilization Using
    Convolutional Neural Networks"
"""

import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# CONFIG VALUES
# ==============================================================================

ITERATIONS = 20
WAVEFRONTS_PER_BAR = 5000
THRESHOLD = 1e-9
GAIN_VALUES = (-0.6, -0.2, 0)
BAR_WIDTH = 0.2

# Capture CNN: [V71a] `wavefront_capture_v5`
# Stabilization CNN: [V71b] `wavefront_stabilization_v5`
# RM: `fixed_40nm_pm`
# SCALING   Z2-3  Z4-8  Z9-24 | Capture CNN  Stabilization CNN    RM
#  4        2000    80     40 |         738                  3     2
#  3.5      1750    70     35 |        1493                 12    12
#  3        1500    60     30 |        2648                 22    20
#  2.5      1250    50     25 |        4189                 29    31
#  2        1000    40     20 |        4982                 84    76
#  1.5       750    30     15 |        5000                151   134
#  1.25      625    25   12.5 |        5000                215   174
#  1         500    20     10 |        5000                337   282
#  0.5       250    10      5 |        5000               1415  1156
#  0.25      125     5    2.5 |        5000               4752  4323
#  0.125    62.5   2.5   1.25 |        5000               5000  5000
#  0.0625  31.25  1.25  0.625 |        5000               5000  5000
RANGES = (500, 20, 10)
SCALE_FACTORS = [0.125, 0.25, 0.5, 1, 2, 3, 4]
Y_CAPTURE_NN = [5000, 5000, 5000, 5000, 4982, 2648, 738]
Y_STABIL_CNN = [5000, 4752, 1415, 337, 84, 22, 3]
Y_RM = [5000, 4323, 1156, 282, 76, 20, 2]

# ==============================================================================

# Load in the style file
plt.style.use('plot_styling.mplstyle')

# Create the subplot
fig, ax = plt.subplots(figsize=(16, 8))


def _display_bars(y_values, offset, color, label):
    ax.bar(
        np.arange(len(y_values)) + offset,
        y_values,
        width=BAR_WIDTH,
        align='center',
        label=label,
        color=color,
        edgecolor='black',
    )


# Display the bars for both the NNs and RM
_display_bars(Y_RM, -BAR_WIDTH, '#FFB3B3', 'RM')
_display_bars(Y_CAPTURE_NN, 0, '#B3E6B3', 'Capture CNN')
_display_bars(Y_STABIL_CNN, BAR_WIDTH, '#99C2FF', 'Stabilization CNN')

# Set the limits on the y axis along with the correct ticks
# ax.set_ylim([0, WAVEFRONTS_PER_BAR * 1.02])
ax.set_ylim([0, WAVEFRONTS_PER_BAR * 1.015])
ax.set_yticks(np.linspace(0, WAVEFRONTS_PER_BAR, 6))

# The margin inside the actual plotting area needs to be smaller
ax.margins(0.005)

# The scale factor values should be the tick labels
ax.set_xticklabels([None, *SCALE_FACTORS])

# Set the titles and axis labels
plt.title(
    f'{ITERATIONS} Iteration Control Loops on Static Aberration '
    'Input Signals\nGain Factors $K_{P,I,D}$ = ' + str(GAIN_VALUES),
    pad=45,
)
ax.set_xlabel('Input Aberration Scaling Factor, $k$\n'
              'Aberrations Uniformly Random Between: '
              r'$k\;[-x_Z, x_Z]$ nm RMS Error, $x_{2-3}=$' + str(RANGES[0]) +
              '$, x_{4-8}=$' + str(RANGES[1]) + '$, x_{9-24}=$' +
              str(RANGES[2]))
ax.set_ylabel('Wavefronts Captured\nTrue Error Threshold: '
              f'[-{int(THRESHOLD * 1e9)}, {int(THRESHOLD * 1e9)}] nm')

# https://stackoverflow.com/a/43439132
plt.legend(
    bbox_to_anchor=(0, 1.02, 1, 0.2),
    loc='lower center',
    borderaxespad=0,
    ncol=3,
)

plt.grid(axis='y', color='black', linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig('convergence_plot.png')
