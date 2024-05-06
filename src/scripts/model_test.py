"""
Will output results to `/output/analysis/<tag>_epoch_<number>`.
Any prior results under this directory will be deleted.

The plotting within this script should be put in its own module.

A check should be added in the future to ensure that the `n_rows` and
`n_cols` have enough cells for the outputs.

The `epoch` argument should accept the 'last' value to automatically select
the most trained epoch for a model.
"""

from h5py import File
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils.hdf_loader import HDFLoader
from utils.load_model import LoadModel
from utils.norm import min_max_denorm
from utils.path import delete_dir, make_dir
from utils.printing_and_logging import step_ri, title


def model_test_parser(subparsers):
    """
    Example commands:
    python3 main.py model_test v1a 110 testing_03_05_global 5 5
    python3 main.py model_test v1a 110 testing_03_05_ind 5 5
    """
    subparser = subparsers.add_parser(
        'model_test',
        help='test a trained model',
    )
    subparser.set_defaults(main=model_test)
    subparser.add_argument(
        'tag',
        help='tag of the model, will look in `/output/trained_models`',
    )
    subparser.add_argument(
        'epoch',
        help='epoch of the trained model to test (just the number part)',
    )
    subparser.add_argument(
        'testing_dataset_name',
        help=('name of the testing dataset, will look in `/data/processed/`, '
              'will use the norm values from the trained model - NOT from the '
              'testing dataset directly'),
    )
    subparser.add_argument(
        'n_rows',
        type=int,
        help='number of rows in the plot for the output comparison',
    )
    subparser.add_argument(
        'n_cols',
        type=int,
        help='number of cols in the plot for the output comparison',
    )


def model_test(cli_args):
    title('Model test script')

    tag = cli_args['tag']
    epoch = cli_args['epoch']

    loaded_model = LoadModel(tag, epoch, eval_mode=True)
    norm_values = loaded_model.get_norm_values()
    model = loaded_model.get_model()

    step_ri('Creating the analysis directory')
    analysis_path = f'../output/analysis/{tag}_epoch_{epoch}'
    delete_dir(analysis_path, quiet=True)
    make_dir(analysis_path)

    step_ri('Loading in the testing dataset')
    testing_name = cli_args['testing_dataset_name']
    testing_dataset = HDFLoader(f'../data/processed/{testing_name}/data.h5')

    step_ri('Calling the model and obtaining its outputs')
    inputs = testing_dataset.get_all_inputs_as_torch()
    with torch.no_grad():
        outputs_model = model(inputs).numpy()

    print('Denormalizing the outputs')
    # Denormalize the outputs
    outputs_model_denormed = min_max_denorm(
        outputs_model,
        norm_values['output_max_min_diff'],
        norm_values['output_min_x'],
    )
    # Testing output data should already be unnormalized
    outputs_truth = testing_dataset.get_all_outputs()

    print('Computing the MAE and MSE')

    def _compute_loss(loss_func):
        return loss_func(reduction='none')(
            torch.from_numpy(outputs_truth),
            torch.from_numpy(outputs_model_denormed)).numpy()

    mae = _compute_loss(torch.nn.L1Loss)
    mse = _compute_loss(torch.nn.MSELoss)
    print(f'MAE: {np.mean(mae)}')
    print(f'MSE: {np.mean(mse)}')

    step_ri('Writing results to HDF')
    out_file_path = f'{analysis_path}/results.h5'
    print(f'File location: {out_file_path}')
    with File(out_file_path, 'w') as out_file:
        out_file['outputs_truth'] = outputs_truth
        out_file['outputs_model_denormed'] = outputs_model_denormed
        out_file['mae'] = mae
        out_file['mse'] = mse

    step_ri('Generating plots')
    n_rows = cli_args['n_rows']
    n_cols = cli_args['n_cols']
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    current_col = 0
    for plot_row in range(n_rows):
        for plot_col in range(n_cols):
            if current_col == outputs_truth.shape[1]:
                break
            model_col = outputs_model_denormed[:, current_col]
            truth_col = outputs_truth[:, current_col]
            axs_cell = axs[plot_row, plot_col]
            axs_cell.set_title(current_col)
            # Take the lowest and greatest values from both sets of data
            lower = min(np.amin(model_col), np.amin(truth_col))
            upper = max(np.amax(model_col), np.amax(truth_col))
            # Fix the bounds on both axes so they are 1-to-1
            axs_cell.set_xlim(lower, upper)
            axs_cell.set_ylim(lower, upper)
            # Draw a 1-to-1 line for the scatters
            # https://stackoverflow.com/a/60950862
            xpoints = ypoints = axs_cell.get_xlim()
            axs_cell.plot(
                xpoints,
                ypoints,
                linestyle='-',
                linewidth=2,
                color='#FFB200',
                scalex=False,
                scaley=False,
            )
            # Plot the scatter of all the points
            axs_cell.scatter(model_col, truth_col, 0.25)
            current_col += 1
    for ax in axs.flat:
        ax.set(xlabel='Model Outputs', ylabel='Truth Outputs')
    fig.tight_layout()
    plt.savefig(
        f'{analysis_path}/comparisons.png',
        dpi=300,
        bbox_inches='tight',
    )
