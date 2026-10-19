from h5py import File
import numpy as np
import matplotlib.pyplot as plt
from utils.constants import PLOT_STYLE_FILE
from utils.norm import z_score_normalize


def _load_data(dir_name):
    data = File(f'../data/processed/{dir_name}/data.h5')
    return data['inputs'][:]


mono = _load_data('coeff_comparison_simplified')
broadband = _load_data('coeff_comparison_broadband')

LOWER_PERCENTILE = 5
UPPER_PERCENTILE = 95
TOTAL_ITERS = 4
Z_SCORE_NORM = True

plt.style.use(PLOT_STYLE_FILE)


def _plot_coeffs(coeffs, iteration, name, ymin, ymax):
    rows, coeff_count = coeffs.shape
    indices = np.arange(coeff_count)
    plt.figure(figsize=(8, 5))
    plt.title(f'{name} EFC Iteration {iteration} (N={rows})')
    plt.xlabel('Coefficient Index')
    plt.ylabel('Coefficient Value')
    plt.fill_between(
        indices,
        np.percentile(coeffs, LOWER_PERCENTILE, axis=0),
        np.percentile(coeffs, UPPER_PERCENTILE, axis=0),
        alpha=0.5,
        label=f'{LOWER_PERCENTILE}-{UPPER_PERCENTILE} Percentile Range',
    )
    plt.plot(
        indices,
        np.percentile(coeffs, 50, axis=0),
        linewidth=1.5,
        label='Median',
    )
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.savefig(f'{name}_coeff_ranges_{iteration}.png')


if Z_SCORE_NORM:
    mono = z_score_normalize(
        mono,
        np.mean(mono[0::TOTAL_ITERS], axis=0),
        np.std(mono[0::TOTAL_ITERS], axis=0),
    )
    broadband = z_score_normalize(
        broadband,
        np.mean(broadband[0::TOTAL_ITERS], axis=0),
        np.std(broadband[0::TOTAL_ITERS], axis=0),
    )

mono_min = np.percentile(mono[0::TOTAL_ITERS, 0], LOWER_PERCENTILE)
mono_max = np.percentile(mono[0::TOTAL_ITERS, 0], UPPER_PERCENTILE)
broadband_min = np.percentile(broadband[0::TOTAL_ITERS, 0], LOWER_PERCENTILE)
broadband_max = np.percentile(broadband[0::TOTAL_ITERS, 0], UPPER_PERCENTILE)

for idx in range(TOTAL_ITERS):
    _plot_coeffs(mono[idx::TOTAL_ITERS], idx + 1, 'Mono', mono_min, mono_max)
    _plot_coeffs(broadband[idx::TOTAL_ITERS], idx + 1, 'Broadband',
                 broadband_min, broadband_max)
