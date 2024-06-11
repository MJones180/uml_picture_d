"""
This script outputs information on a preprocessed dataset.
"""

import matplotlib.pyplot as plt
import torchvision
from utils.constants import RANDOM_P
from utils.load_network import load_network
from utils.path import make_dir
from utils.printing_and_logging import step_ri, title
from utils.torch_hdf_ds_loader import DSLoaderHDF


def dataset_info_parser(subparsers):
    """
    Example command:
        python3 main.py dataset_info \
            train_fixed_10nm_gl \
            --verify-network-compatability test \
            --plot-example-images --plot-outputs-hist
    """
    subparser = subparsers.add_parser(
        'dataset_info',
        help='display info on a dataset',
    )
    subparser.set_defaults(main=dataset_info)
    subparser.add_argument(
        'dataset_name',
        help='name of the dataset',
    )
    subparser.add_argument(
        '--verify-network-compatability',
        help=('name of the python script containing the network (without the '
              '`.py`)'),
    )
    subparser.add_argument(
        '--plot-example-images',
        action='store_true',
        help='plot 5 example images',
    )
    subparser.add_argument(
        '--plot-outputs-hist',
        action='store_true',
        help='plot a histogram for each of the outputs in the dataset',
    )


def dataset_info(cli_args):
    title('Dataset info script')

    dataset_name = cli_args['dataset_name']
    data = DSLoaderHDF(dataset_name)
    inputs = data.get_inputs_torch()
    outputs = data.get_outputs()
    step_ri(dataset_name)
    print('Number of rows: ', len(data))
    print('Input shape: ', inputs.shape[1:])
    print('Output shape: ', outputs.shape[1:])

    # Verify that the dimensions of the dataset work with the given network
    verify_network = cli_args['verify_network_compatability']
    if verify_network:
        step_ri(f'Verifying that `{verify_network}` works with this dataset')
        # We are feeding in one row, so we need to add back in the batch dim
        input_test = inputs[0][None, :]
        print(f'Feeding in an input of shape: {input_test.shape}')
        output_network_shape = load_network(verify_network)()(input_test).shape
        print('Network outputs the shape: ', output_network_shape)
        # We need to add back in the batch dimension
        expected_output_shape = outputs[0][None, :].shape
        if output_network_shape == expected_output_shape:
            print('The network outputs the correct shape!')
        else:
            print('The network DOES NOT output the correct shape')

    plot_example_images = cli_args['plot_example_images']
    plot_outputs_hist = cli_args['plot_outputs_hist']
    if plot_example_images or plot_outputs_hist:
        base_plot_path = f'{RANDOM_P}/ds_info_{dataset_name}/'
        make_dir(base_plot_path)

    if plot_example_images:
        step_ri('Plotting 5 example images from this dataset')
        images = inputs[:5]
        img_grid = torchvision.utils.make_grid(images, nrow=5).mean(dim=0)
        plt.imshow(img_grid.numpy(), cmap='Greys_r')
        path = f'{base_plot_path}/example_images.png'
        print(f'Saving plot: {path}')
        plt.savefig(path, bbox_inches='tight')

    if plot_outputs_hist:
        step_ri('Plotting a histogram for each output')
        for idx in range(outputs.shape[1]):
            plt.clf()
            plt.title(f'Output {idx}')
            plt.hist(outputs[:, idx])
            path = f'{base_plot_path}/output_hist_{idx}.png'
            print(f'Saving plot: {path}')
            plt.savefig(path, bbox_inches='tight')
