"""
This script outputs information on a given network (untrained model structure).
"""

import torch
from utils.benchmark_nn import benchmark_nn
from utils.load_network import load_network
from utils.printing_and_logging import (dec_print_indent, inc_print_indent,
                                        step_ri, title)
from utils.shared_argparser_args import shared_argparser_args
from utils.torch_grab_device import torch_grab_device


def network_info_parser(subparsers):
    subparser = subparsers.add_parser(
        'network_info',
        help='display info on a network',
    )
    subparser.set_defaults(main=network_info)
    shared_argparser_args(subparser, ['network_name', 'force_cpu'])
    subparser.add_argument(
        '--benchmark',
        type=int,
        help='benchmark network by running n rows',
    )


def network_info(cli_args):
    title('Network info script')

    step_ri('Loading in the network')
    network_name = cli_args['network_name']
    print(f'Network: {network_name}')
    network = load_network(network_name)
    input_data = network.example_input()
    device = torch_grab_device(cli_args['force_cpu'])
    network_inst = network().to(device)

    step_ri('Trainable parameters')
    total = 0
    for name, parameter in network_inst.named_parameters():
        if parameter.requires_grad:
            params = parameter.numel()
            print(name, params)
            total += params
    dec_print_indent()
    print(f'Total: {total}')

    step_ri('Layers')
    print(network_inst)

    input_data_dev = input_data.to(device)

    step_ri('Sample call')
    output_data = network_inst(input_data_dev)
    print('Output data: ', output_data)
    print('Output shape: ', output_data.shape)

    step_ri('Input and output shapes from each layer')

    # https://discuss.pytorch.org/t/how-to-get-the-input-and-output-feature-maps
    # -of-a-cnn-model/152259/2
    def hook(module, input_data, output_data):
        print(module)
        inc_print_indent()
        print('Input: ', input_data[0].shape)
        print('Output: ', output_data.shape)
        dec_print_indent()

    for module in network_inst.children():
        module.register_forward_hook(hook)
    network_inst(input_data_dev)

    if cli_args['benchmark'] is not None:
        # Redefining the instance because the previous one has an added hook
        model = network().to(device)

        def call_wrapper():
            with torch.no_grad():
                model(input_data_dev).cpu().numpy()

        benchmark_nn(cli_args['benchmark'], call_wrapper)
