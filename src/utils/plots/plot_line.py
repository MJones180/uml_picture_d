import matplotlib.pyplot as plt
import numpy as np
from utils.constants import PLOT_STYLE_FILE


def plot_line(
    data,
    title,
    xlabel,
    ylabel,
    plot_path,
    labels=None,
    show_grid=False,
    hlines=None,
):
    # Load in the style file
    plt.style.use(PLOT_STYLE_FILE)
    plt.clf()
    plt.figure(figsize=(14, 6))
    # Treat all data as 2D
    if len(np.asarray(data).shape) == 1:
        data = [data]
    for idx, line in enumerate(data):
        label = '' if labels is None else labels[idx]
        plt.plot(line, label=label)
    if labels is not None:
        plt.legend()
    if hlines is not None:
        for hline in hlines:
            plt.axhline(hline, color='black', linestyle='--')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if show_grid:
        plt.grid()
    plt.savefig(plot_path)
