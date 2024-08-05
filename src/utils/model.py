from glob import glob
import torch
from utils.constants import ARGS_F, EXTRA_VARS_F, TRAINED_MODELS_P
from utils.hdf_read_and_write import read_hdf
from utils.json import json_load
from utils.load_network import load_network
from utils.path import path_exists
from utils.printing_and_logging import dec_print_indent, step
from utils.terminate_with_message import terminate_with_message
from utils.torch_grab_device import torch_grab_device


class Model():

    def __init__(self, tag, epoch, suppress_logs=False, force_cpu=False):

        def _step(text):
            if suppress_logs:
                return
            step(text)

        def _print(text):
            if suppress_logs:
                return
            print(text)

        _step('Loading in the trained model')

        dir_path = f'{TRAINED_MODELS_P}/{tag}'
        _print(f'Model directory path: {dir_path}')
        if not path_exists(dir_path):
            terminate_with_message(f'Directory not found: {dir_path}')

        if epoch.lower() == 'last':
            _step('Epoch set to `last` mode, so finding last epoch')
            # The base path of the model to find the epochs within
            epoch_path_part = f'{dir_path}/epoch_'
            epoch_path_part_len = len(epoch_path_part)
            # Grab the highest epoch found
            epoch = max([
                # Chop off all the path except for the number and make it an int
                int(path[epoch_path_part_len:])
                # Get a glob of all epochs found
                for path in glob(f'{epoch_path_part}[0-9]*')
            ])
            _print(f'Using epoch {epoch}')
            dec_print_indent()
            print()

        # Set the instance variables
        self.tag = tag
        self.epoch = epoch

        model_path = f'{dir_path}/epoch_{epoch}'
        _print(f'Model directory path with epoch: {model_path}')
        if not path_exists(model_path):
            terminate_with_message(f'Model not found at {model_path}')

        _print('Loading in the training args')
        self.training_args = json_load(f'{dir_path}/{ARGS_F}')

        _print('Loading in the extra variables')
        self.extra_vars = read_hdf(f'{dir_path}/{EXTRA_VARS_F}')

        self.network_name = self.training_args['network_name']
        _print(f'Loading in the network (`{self.network_name}`) '
               'and setting weights')
        self.device = torch_grab_device(force_cpu)
        # Need to first load in the network
        self.network = load_network(self.network_name)
        self.model = self.network().to(self.device)
        # Now, the weights can be set
        self.model.load_state_dict(torch.load(model_path))
        # Set to evaluation mode
        self.model.eval()
        dec_print_indent()

    def call_model(self, data):
        with torch.no_grad():
            data_dev = data.to(self.device)
            model_outputs = self.model(data_dev).cpu().numpy()
        return model_outputs

    def __call__(self, *args, **kwargs):
        return self.call_model(*args, **kwargs)
