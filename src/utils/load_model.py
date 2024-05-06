import torch
from utils.json import json_load
from utils.load_network import load_network
from utils.path import path_exists
from utils.printing_and_logging import dec_print_indent, step
from utils.terminate_with_message import terminate_with_message


class LoadModel():

    def __init__(self, tag, epoch, eval_mode=False):
        self.tag = tag
        self.epoch = epoch
        self.eval_mode = eval_mode

        step('Loading in the trained model')

        self.dir_path = f'../output/trained_models/{self.tag}'
        print(f'Model directory path: {self.dir_path}')

        self.model_path = f'{self.dir_path}/epoch_{self.epoch}'
        if not path_exists(self.model_path):
            terminate_with_message(f'Model not found at {self.model_path}')

        print('Loading in the norm values')
        self.norm_values = json_load(f'{self.dir_path}/norm.json')

        print('Loading in the training args')
        self.training_args = json_load(f'{self.dir_path}/args.json')

        self.network_name = self.training_args['network_name']
        print(f'Network used: {self.network_name}')
        print('Loading in the network and setting weights')
        # Need to first load in the network
        self.network = load_network(self.network_name)
        self.model = self.network()
        # Now, the weights can be set
        self.model.load_state_dict(torch.load(self.model_path))
        if self.eval_mode:
            # Set to evaluation mode
            self.model.eval()

        dec_print_indent()

    def get_args(self):
        return self.training_args

    def get_model(self):
        return self.model

    def get_network(self):
        return self.network

    def get_norm_values(self):
        return self.norm_values
