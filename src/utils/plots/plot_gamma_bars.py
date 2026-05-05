import matplotlib.pyplot as plt
import numpy as np
from utils.constants import PLOT_STYLE_FILE


def plot_gamma_bars(gamma_magnitudes, plot_path):
    # Load in the style file
    plt.style.use(PLOT_STYLE_FILE)
    # Reset the plot
    plt.clf()
    plt.figure(figsize=(14, 6))
    plt.bar(np.arange(len(gamma_magnitudes)), gamma_magnitudes, linewidth=0.5)
    plt.title('Gamma Magnitudes per Layer')
    plt.xlabel('Layer')
    plt.ylabel('Mean Gamma Magnitude')
    plt.xticks(rotation=90)
    plt.savefig(plot_path)
