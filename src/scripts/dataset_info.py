"""
This script outputs information on a preprocessed dataset.
"""

import matplotlib.pyplot as plt
import torch
import torchvision
from utils.hdf_loader import HDFLoader
from utils.load_network import load_network
from utils.printing_and_logging import step_ri, title


def dataset_info_parser(subparsers):
    """
    Example command:
    python3 main.py dataset_info \
        training_03_05_global \
        --verify-network-compatability test \
        --display-example-images
    """
    subparser = subparsers.add_parser(
        'dataset_info',
        help='display info on a dataset',
    )
    subparser.set_defaults(main=dataset_info)
    subparser.add_argument(
        'dataset_name',
        help='name of the dataset, will look in `/data/processed/`',
    )
    subparser.add_argument(
        '--verify-network-compatability',
        help=('name of the python script containing the network (without the '
              '`.py`), must be located in the `/src/networks` folder'),
    )
    subparser.add_argument(
        '--display-example-images',
        action='store_true',
        help='display 5 example images',
    )


def dataset_info(cli_args):
    title('Dataset info script')

    dataset_name = cli_args['dataset_name']
    data = HDFLoader(f'../data/processed/{dataset_name}/data.h5')
    inputs = data.get_all_inputs()
    outputs = data.get_all_outputs()
    step_ri(dataset_name)
    print('Number of rows: ', len(data))
    print('Input shape: ', inputs.shape[1:])
    print('Output shape: ', outputs.shape[1:])

    # Verify that the dimensions of the dataset work with the given network
    verify_network = cli_args['verify_network_compatability']
    if verify_network:
        step_ri(f'Verifying that `{verify_network}` works with this dataset')
        # We are feeding in one row, so we need to add back in the batch dim
        input_test = torch.from_numpy(inputs[0][None, :])
        print(f'Feeding in an input of shape: {input_test.shape}')
        output_network_shape = load_network(verify_network)()(input_test).shape
        print('Network outputs the shape: ', output_network_shape)
        # We need to add back in the batch dimension
        expected_output_shape = outputs[0][None, :].shape
        if output_network_shape == expected_output_shape:
            print('The network outputs the correct shape!')
        else:
            print('The network DOES NOT output the correct shape :(')

    example_images = cli_args['display_example_images']
    if example_images:
        step_ri('Displaying 5 example images from this dataset')
        images = torch.from_numpy(inputs[:5])
        img_grid = torchvision.utils.make_grid(images, nrow=5).mean(dim=0)
        plt.imshow(img_grid.numpy(), cmap='Greys_r')
        plt.show()
