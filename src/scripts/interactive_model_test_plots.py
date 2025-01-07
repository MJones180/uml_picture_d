"""
This script allows plots to be run in interactive mode from a previously run
model test. The animations are not run in interactive mode.

A lot of this code is taken from the `model_test` script and should be
modularized, but that is a task for another day.
"""

import numpy as np
from utils.constants import (ANALYSIS_P, RESULTS_F)
from utils.hdf_read_and_write import read_hdf
from utils.plots.plot_comparison_scatter_grid import plot_comparison_scatter_grid  # noqa
from utils.plots.plot_zernike_response import plot_zernike_response
from utils.plots.plot_zernike_total_cross_coupling import plot_zernike_total_cross_coupling  # noqa
from utils.printing_and_logging import step_ri, title
from utils.shared_argparser_args import shared_argparser_args


def inter_model_test_plots_parser(subparsers):
    subparser = subparsers.add_parser(
        'interactive_model_test_plots',
        help='run the plots in interactive mode from a previous model test',
    )
    subparser.set_defaults(main=interactive_model_test_plots)
    shared_argparser_args(subparser, ['tag', 'epoch'])
    subparser.add_argument(
        'testing_ds',
        help=('name of the testing dataset used'),
    )
    subparser.add_argument(
        '--scatter-plot',
        nargs=5,
        metavar=('[n_rows]', '[n_cols]', '[starting_zernike]',
                 '[filter_value]', '[plot_density]'),
        help=('generate a scatter plot; takes the args: number of rows, '
              'number of cols, first Zernike the model outputs, filter value '
              'range, points per pixel to use for the density plot'),
    )
    subparser.add_argument(
        '--zernike-plots',
        nargs=2,
        metavar=('[zernike_low]', '[zernike_high]'),
        type=int,
        help=('generate the Zernike plots, the range of the Zernike terms '
              'being used must be passed in as this data is not stored with '
              'the outputted datafile and the dataset may not be local'),
    )


def interactive_model_test_plots(cli_args):
    title('Interactive model test plots script')

    tag = cli_args['tag']
    epoch = cli_args['epoch']
    testing_ds_tag = cli_args['testing_ds']

    data = read_hdf(f'{ANALYSIS_P}/{testing_ds_tag}/{tag}_epoch_{epoch}'
                    f'/{RESULTS_F}')
    outputs_truth = data['outputs_truth']
    outputs_model = data['outputs_model']

    plot_title = 'Neural Network'
    plot_identifier = f'{tag}, epoch {epoch}'

    scatter_plot = cli_args.get('scatter_plot')
    if scatter_plot is not None:
        step_ri('Generating scatter plot and density scatter plot')
        filter_value = float(scatter_plot.pop(3))
        (n_rows, n_cols, starting_zernike,
         plot_density) = [int(arg) for arg in scatter_plot]
        print(f'Using {n_rows} rows and {n_cols} cols.')
        print(f'Starting Zernike: {starting_zernike}.')
        print(f'Filtering between: [-{filter_value},{filter_value}].')
        print(f'Point per pixel for density plot: {plot_density}.')
        step_ri('Displaying scatter plot')
        plot_comparison_scatter_grid(
            outputs_model,
            outputs_truth,
            n_rows,
            n_cols,
            plot_title,
            plot_identifier,
            starting_zernike,
            interactive_view=True,
            filter_value=filter_value,
        )
        step_ri('Displaying density scatter plot')
        plot_comparison_scatter_grid(
            outputs_model,
            outputs_truth,
            n_rows,
            n_cols,
            plot_title,
            plot_identifier,
            starting_zernike,
            interactive_view=True,
            plot_density=plot_density,
            filter_value=filter_value,
        )

    zernike_plots = cli_args.get('zernike_plots')
    if zernike_plots:
        nrows = outputs_truth.shape[0]
        zernike_terms = np.arange(zernike_plots[0], zernike_plots[1] + 1)

        def _split(data):
            # Split the data so that each group (first dim) consists of all
            # the Zernike terms perturbed by a given amount
            return np.array(np.split(data, nrows // len(zernike_terms)))

        # Groups will have the shape (rms pert, zernike terms, zernike terms)
        outputs_truth_gr = _split(outputs_truth)
        outputs_model_gr = _split(outputs_model)

        # It is assumed that the truth terms all have the same perturbation
        # for each group and that there are only perturbations along the main
        # diagonal. Therefore, each group (first dim) should be equivalent to
        # `perturbation * identity matrix`. Due to this, we can simply obtain
        # the list of all RMS perturbations.
        perturbation_grid = outputs_truth_gr[:, 0, 0]

        step_ri('Displaying a Zernike response plot')
        plot_zernike_response(
            zernike_terms,
            perturbation_grid,
            outputs_model_gr,
            plot_title,
            plot_identifier,
            interactive_view=True,
        )

        step_ri('Displaying a Zernike total cross coupling plot')
        plot_zernike_total_cross_coupling(
            perturbation_grid,
            outputs_model_gr,
            plot_title,
            plot_identifier,
            interactive_view=True,
        )
