from utils.load_network import load_network


def network_info_parser(subparsers):
    """
    Example command:
    python3 main.py network_info basic
    """
    subparser = subparsers.add_parser(
        'network_info',
        help='display info on a network',
    )
    subparser.set_defaults(main=network_info)
    subparser.add_argument(
        'network_name',
        help=('name of the python script containing the network (without the '
              '`.py`), must be located in the `/src/networks` folder'),
    )


def network_info(cli_args):
    network_name = cli_args['network_name']
    network = load_network(network_name)
    input_data = network.example_input()
    network_inst = network()
    print(f'Network: {network_name}')
    print('Layers:')
    total = 0
    for name, parameter in network_inst.named_parameters():
        if parameter.requires_grad:
            params = parameter.numel()
            print('    ', name, params)
            total += params
    print(f'Total Trainable Params: {total}')
    print('Sample call...')
    print(network_inst(input_data))
