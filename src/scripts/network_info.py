"""
This script outputs information on a given network (untrained model structure).
"""

from utils.load_network import load_network
from utils.printing_and_logging import (dec_print_indent, inc_print_indent,
                                        step_ri, title)
from utils.shared_argparser_args import shared_argparser_args


def network_info_parser(subparsers):
    """
    Example command:
    python3 main.py network_info test
    """
    subparser = subparsers.add_parser(
        'network_info',
        help='display info on a network',
    )
    subparser.set_defaults(main=network_info)
    shared_argparser_args(subparser, ['network_name'])


def network_info(cli_args):
    title('Network info script')

    step_ri('Loading in the network')
    network_name = cli_args['network_name']
    print(f'Network: {network_name}')
    network = load_network(network_name)
    input_data = network.example_input()
    network_inst = network()
    print('Layers:')
    inc_print_indent()
    total = 0
    for name, parameter in network_inst.named_parameters():
        if parameter.requires_grad:
            params = parameter.numel()
            print(name, params)
            total += params
    dec_print_indent()
    print(f'Total Trainable Params: {total}')

    step_ri('Sample call')
    output_data = network_inst(input_data)
    print('Output data: ', output_data)
    print('Output shape: ', output_data.shape)
