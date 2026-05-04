import matplotlib.pyplot as plt
import numpy as np
from utils.constants import RANDOM_P
from utils.printing_and_logging import step_ri, title
from utils.torch_hdf_ds_loader import DSLoaderHDF


def plot_output_coeff_ranges_parser(subparsers):
    subparser = subparsers.add_parser(
        'plot_output_coeff_ranges',
        help='plot output coefficient ranges',
    )
    subparser.set_defaults(main=plot_output_coeff_ranges)
    subparser.add_argument(
        'testing_dataset_tag',
        help=('name of the testing dataset to use the output coeffs from; '
              'must use a testing dataset as the outputs are not normalized'),
    )
    subparser.add_argument(
        '--coeff-group-idxs',
        type=int,
        nargs='+',
        help='upper index for each coeff group',
    )
    subparser.add_argument(
        '--lower-percentile',
        type=float,
        default=5,
        help='lower percentile to plot',
    )
    subparser.add_argument(
        '--upper-percentile',
        type=float,
        default=95,
        help='upper percentile to plot',
    )


def plot_output_coeff_ranges(cli_args):
    title('Plot output coeff ranges script')

    step_ri('Loading in the testing dataset')
    tag = cli_args['testing_dataset_tag']
    print(f'Tag: {tag}')
    coeffs = DSLoaderHDF(tag).get_outputs()
    print(f'Shape: {coeffs.shape}')

    step_ri('Calculating percentiles')
    lower_percentile = cli_args.get('lower_percentile')
    upper_percentile = cli_args.get('upper_percentile')
    print(f'Lower percentile: {lower_percentile}')
    print(f'Upper percentile: {upper_percentile}')
    lower_percentile_vals = np.percentile(coeffs, lower_percentile, axis=0)
    upper_percentile_vals = np.percentile(coeffs, upper_percentile, axis=0)
    median_percentile_vals = np.percentile(coeffs, 50, axis=0)

    step_ri('Setting coeff groups')
    coeff_group_idxs = cli_args.get('coeff_group_idxs')
    print(f'Upper group idxs: {coeff_group_idxs}')
    numb_groups = len(coeff_group_idxs)
    print(f'Number of groups: {numb_groups}')
    # Add the first coeff index at zero
    coeff_group_idxs = [0, *coeff_group_idxs]

    step_ri('Plotting')
    fig, axs = plt.subplots(numb_groups, 1, figsize=(12, 10))
    for idx in range(numb_groups):
        lower_bound = coeff_group_idxs[idx]
        upper_bound = coeff_group_idxs[idx + 1]
        indices = np.arange(lower_bound, upper_bound)
        axs[idx].set_title(f'Coefficient Group {idx + 1} '
                           f'({lower_bound} - {upper_bound})')
        axs[idx].set_xlabel('Coefficient Index')
        axs[idx].set_ylabel('Coefficient Value')
        axs[idx].plot(
            indices,
            median_percentile_vals[lower_bound:upper_bound],
            linewidth=1.5,
            label='Median',
        )
        axs[idx].fill_between(
            indices,
            lower_percentile_vals[lower_bound:upper_bound],
            upper_percentile_vals[lower_bound:upper_bound],
            alpha=0.3,
            label=f'{lower_percentile}-{upper_percentile} Percentile Range',
        )
        axs[idx].legend()
        axs[idx].grid(True, alpha=0.3)
    path = f'{RANDOM_P}/coeff_ranges_{tag}.png'
    print(f'Saving plot to {path}')
    plt.savefig(path, bbox_inches='tight')
