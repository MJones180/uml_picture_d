"""
Adapted from `utils/plots/plot_control_loop_zernikes_subplots.py`.
Plots to compare the error between the CNN and RM.
"""

import matplotlib.pyplot as plt
import numpy as np

# True: true error | False: measurement error
TRUE_DATA = True
STARTING_ZERNIKE = 2

NN_CLR_TAG = 'picc_control_steps_1000_NN_weighted_aberration_ranges_local_v4_last_-0.6_-0.2_0.0'
RM_CLR_TAG = 'picc_control_steps_1000_RM_fixed_pm_40nm_-0.6_-0.2_0.0'

NN_CLR_TAG = 'picc_control_steps_5000_NN_weighted_aberration_ranges_local_v4_last_-0.6_-0.2_0.0'
RM_CLR_TAG = 'picc_control_steps_5000_RM_fixed_pm_40nm_-0.6_-0.2_0.0'


def plot_nn_rm_control_loop_zernikes_subplots(
    zernike_terms,
    nn_coeffs,
    rm_coeffs,
    title,
    n_rows,
    n_cols,
    plot_path,
    legend_labels=None,
):
    # Load in the style file
    plt.style.use('plot_styling.mplstyle')

    # ==================
    # Setup the subplots
    # ==================

    total_steps, col_count = nn_coeffs.shape
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        sharex=True,
        figsize=(n_cols * 4, n_rows * 2),
    )

    # ===============
    # Do the plotting
    # ===============

    plt.suptitle(title)
    current_col = 0
    for plot_row in range(n_rows):
        for plot_col in range(n_cols):
            # Remove axes for unused cells
            if current_col >= col_count:
                fig.delaxes(axs[plot_row, plot_col])
                continue
            axs_cell = axs[plot_row, plot_col]
            # Add the Zernike number to the top left of the plot
            # https://stackoverflow.com/a/50091489
            axs_cell.annotate(
                f'Z{zernike_terms[current_col]}',
                (0, 1),
                xytext=(5, -5),
                xycoords='axes fraction',
                textcoords='offset points',
                fontweight='bold',
                color='white',
                backgroundcolor='k',
                ha='left',
                va='top',
            )

            # A line along zero
            axs_cell.plot(
                np.zeros_like(nn_coeffs[:, 0]),
                color='Black',
            )

            current_label = 0

            def _add_line(data, color):
                nonlocal current_label
                axs_cell.plot(
                    data[:, current_col] * 1e9,
                    label=str(current_label),
                    color=color,
                )
                current_label += 1

            _add_line(nn_coeffs, 'Blue')
            _add_line(rm_coeffs, 'Red')

            # Hide labels so they do not appear on every plot
            axs_cell.set_xlabel('')
            axs_cell.set_ylabel('')
            # Only display x labels for the last row
            if plot_row == n_rows - 1:
                pos = np.linspace(0, total_steps, 5)
                labs = [int(v) for v in np.linspace(0, total_steps, 5)]
                axs_cell.set_xticks(pos, labs)
                label = 'Timesteps'
                axs_cell.set_xlabel(label)
            # Only display y labels for the first column
            if plot_col == 0:
                label = 'Coeffs\n[nm RMS]'
                axs_cell.set_ylabel(label)
            axs_cell.set_xmargin(0)
            current_col += 1

    # Add the legend only if the labels were passed
    if legend_labels is not None:
        lines, labels = axs_cell.get_legend_handles_labels()
        fig.legend(lines, legend_labels, loc='lower right')

    fig.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.1)

    # =============
    # Save the plot
    # =============

    plt.savefig(plot_path)


CONTROL_LOOP_PATH = '../output/control_loop_results'
MEAS_PATH = '/history_data/meas_error.csv'
TRUE_PATH = '/history_data/true_error.csv'


def _load_data(tag, error):
    path = f'{CONTROL_LOOP_PATH}/{tag}/{error}'
    return np.loadtxt(path, delimiter=',')


if TRUE_DATA:
    nn_data = _load_data(NN_CLR_TAG, TRUE_PATH)
    rm_data = _load_data(RM_CLR_TAG, TRUE_PATH)
    legend_labels = ['NN True', 'RM True']
else:
    nn_data = _load_data(NN_CLR_TAG, MEAS_PATH)
    rm_data = _load_data(RM_CLR_TAG, MEAS_PATH)
    legend_labels = ['NN Meas', 'RM Meas']

plot_nn_rm_control_loop_zernikes_subplots(
    np.arange(nn_data.shape[1]) + STARTING_ZERNIKE,
    nn_data,
    rm_data,
    f'{NN_CLR_TAG}\n{RM_CLR_TAG}',
    5,
    5,
    './nn_vs_rm_control_loop_zernike_scatters.png',
    legend_labels=legend_labels,
)
