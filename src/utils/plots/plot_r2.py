import matplotlib.pyplot as plt
import numpy as np
from utils.constants import PLOT_STYLE_FILE
from utils.stats_and_error import sum_of_squares


def plot_r2(truth_output, pred_output, plot_path, title_append=''):
    # Error from the response matrix predictions
    model_error = sum_of_squares(truth_output - pred_output, 0)
    # Error from guessing the mean
    mean_error = sum_of_squares(truth_output - truth_output.mean(axis=0), 0)
    # R^2 describes much better the model predictions are
    # than just guessing the mean of the data
    r2_per_output = 1 - model_error / mean_error
    cumulative_r2 = 1 - np.cumsum(model_error) / np.cumsum(mean_error)

    plt.style.use(PLOT_STYLE_FILE)
    plt.clf()
    fig, ax1 = plt.subplots(figsize=(14, 6))
    color_ax1 = 'tab:blue'
    color_ax2 = 'tab:red'
    ax1.set_xlabel('Output Index')
    ax1.set_ylabel(r'$R^2$', color=color_ax1)
    ax1.plot(r2_per_output, color=color_ax1)
    ax1.tick_params(axis='y', labelcolor=color_ax1)
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'Cumulative $R^2$', color=color_ax2)
    ax2.plot(cumulative_r2, color=color_ax2)
    ax2.tick_params(axis='y', labelcolor=color_ax2)
    plt.title(rf'$R^2$ Per Output{title_append}')
    plt.savefig(plot_path)
