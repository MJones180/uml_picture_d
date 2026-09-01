import numpy as np
from utils.constants import DATA_F, INPUTS, OUTPUTS, PROC_DATA_P, RANDOM_P
from utils.hdf_read_and_write import read_hdf
from utils.path import make_dir
from utils.plots.plot_r2 import plot_r2
from utils.printing_and_logging import step_ri, title


def linear_observability_analysis_parser(subparsers):
    subparser = subparsers.add_parser(
        'linear_observability_analysis',
        help='determine the observability of features in a processed dataset',
    )
    subparser.set_defaults(main=linear_observability_analysis)
    subparser.add_argument(
        'train_tag',
        help='tag of the processed training data',
    )
    subparser.add_argument(
        'val_tag',
        help='tag of the processed validation data',
    )
    subparser.add_argument(
        '--alpha',
        type=float,
        help='alpha to use in the ridge regression',
    )


def linear_observability_analysis(cli_args):
    title('Linear observability analysis script')

    step_ri('Loading in the processed datasets')
    train_tag = cli_args['train_tag']
    print(f'Train tag: {train_tag}')
    val_tag = cli_args['val_tag']
    print(f'Val tag: {val_tag}')

    def _load_data(tag):
        data_obj = read_hdf(f'{PROC_DATA_P}/{tag}/{DATA_F}')
        input_data = data_obj[INPUTS][:]
        output_data = data_obj[OUTPUTS][:]
        print(f'Input data shape: {input_data.shape}')
        print(f'Output data shape: {output_data.shape}')
        return input_data, output_data

    train_input, train_output = _load_data(train_tag)
    val_input, val_output = _load_data(val_tag)

    step_ri('Computing response matrix')
    alpha = cli_args.get('alpha') or 0
    print(f'alpha: {alpha}')
    # The penalty due to the Ridge regression
    diagonal_penalty = alpha * np.eye(train_input.shape[1])
    # Solving a system of equations using least squares with a ridge regression
    # Equation: input @ response_matrix = output
    # Shape of response_matrix: (inputs, outputs)
    response_matrix = np.linalg.solve(
        train_input.T @ train_input + diagonal_penalty,
        train_input.T @ train_output,
    )
    print(f'Response matrix shape: {response_matrix.shape}')

    step_ri('Predicting output using response matrix')
    pred_output = val_input @ response_matrix
    print(f'Output prediction shape: {pred_output.shape}')

    step_ri('Creating the output dir')
    out_plot_path = f'{RANDOM_P}/linear_observability_{train_tag}_{val_tag}/'
    print(f'Path: {out_plot_path}')
    make_dir(out_plot_path)

    step_ri('Plotting the R^2')
    plot_r2(
        val_output,
        pred_output,
        f'{out_plot_path}/r2_alpha_{alpha}.png',
        f' (alpha: {alpha})',
    )
