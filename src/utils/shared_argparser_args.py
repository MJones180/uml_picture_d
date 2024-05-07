from utils.terminate_with_message import terminate_with_message


def shared_argparser_args(sp, args_to_add):
    """
    Many scripts will share the same arguments, so they are added here.
    The order that the arguments are passed in the list do matter as that is
    the order they will appear on the CLI.

    This might not be the most pythonic file and may go against proper
    `argparse` etiquette, but it works and is easy.
    """
    # All shared arguments are functions so that they are not instantly
    # evaluated, this prevents them all from getting added
    shared_args = {
        'epoch': lambda: sp.add_argument(
            'epoch',
            help=('epoch of the trained model (just the number part), '
                  'alternatively the value of `last` can be passed which '
                  'will use the highest epoch available (note, this does '
                  'not necessarily mean it is the "best" epoch)'),
        ),
        'network_name': lambda: sp.add_argument(
            'network_name',
            help=('name of the python script containing the network '
                  '(without the `.py`)'),
        ),
        'tag': lambda: sp.add_argument(
            'tag',
            help='tag of the model',
        ),
    }
    # Add only the requested args
    for arg_to_add in args_to_add:
        if arg_to_add not in shared_args:
            terminate_with_message(f'Unknown shared arg: {arg_to_add}')
        shared_args[arg_to_add]()
