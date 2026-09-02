import numpy as np
from utils.plots.plot_line import plot_line
from utils.stats_and_error import r2


def plot_r2(truth_output, pred_output, plot_path, title_append=''):
    r2_per_output, model_error, mean_error = r2(
        truth_output,
        pred_output,
        return_errors=True,
    )
    cumulative_r2 = 1 - np.cumsum(model_error) / np.cumsum(mean_error)
    plot_line(
        [r2_per_output, cumulative_r2],
        rf'$R^2$ Per Output{title_append}',
        'Output Index',
        'Value',
        plot_path,
        labels=[
            rf'$R^2$ (Avg {np.mean(r2_per_output):0.4f})',
            r'Cumulative $R^2$',
        ],
        hlines=[0],
    )
