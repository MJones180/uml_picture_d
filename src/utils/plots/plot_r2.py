import numpy as np
from utils.plots.plot_line import plot_line
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
    plot_line(
        [r2_per_output, cumulative_r2],
        rf'$R^2$ Per Output{title_append}',
        'Output Index',
        'Value',
        plot_path,
        labels=[r'$R^2$', r'Cumulative $R^2$'],
        hlines=[0],
    )
