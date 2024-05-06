import torch
from utils.json import json_load
from utils.load_network import load_network


class LoadModel():

    def __init__(self, tag, epoch, eval_mode=False):
        self.tag = tag
        self.epoch = epoch
        self.eval_mode = eval_mode

        self.dir_path = f'../output/trained_models/{self.tag}'
        print(f'Model directory path: {self.dir_path}')

        self._load_norm()
        self._load_args()
        self._load_model()

    def _load_norm(self):
        self.norm_values = json_load(f'{self.dir_path}/norm.json')

    def _load_args(self):
        self.training_args = json_load(f'{self.dir_path}/args.json')

    def _load_model(self):
        self.network_name = self.training_args['network_name']
        print(f'Network used: {self.network_name}')
        # Need to first load in the network
        self.network = load_network(self.network_name)
        self.model = self.network()
        model_epoch_path = f'{self.dir_path}/epoch_{self.epoch}'
        # Now, the weights can be set
        self.model.load_state_dict(torch.load(model_epoch_path))
        if self.eval_mode:
            # Set to evaluation mode
            self.model.eval()

    def get_args(self):
        return self.training_args

    def get_model(self):
        return self.model

    def get_network(self):
        return self.network

    def get_norm_values(self):
        return self.norm_values
