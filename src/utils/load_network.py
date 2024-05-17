from importlib import import_module
from utils.constants import NETWORKS
from utils.terminate_with_message import terminate_with_message


def load_network(name):
    try:
        module = import_module(f'{NETWORKS}.{name}')
    except ModuleNotFoundError:
        terminate_with_message(f'No network by the name of `{name}` found '
                               f'within the `{NETWORKS}` folder.')
    return module.Network
