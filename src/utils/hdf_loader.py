from h5py import File
import torch
from utils.path import path_exists
from utils.terminate_with_message import terminate_with_message


class HDFLoader(torch.utils.data.Dataset):

    def __init__(self, path):
        self.path = path
        self.inputs = None
        self.outputs = None
        if not path_exists(path):
            terminate_with_message(f'Dataset not found at {path}')

    def get_path(self):
        return self.path

    def get_all_inputs(self):
        self._load_dataset()
        return self.inputs

    def get_all_outputs(self):
        self._load_dataset()
        return self.outputs

    def _load_dataset(self):
        if self.inputs is None:
            dataset = File(self.path, 'r')
            # Inputs must be float32
            self.inputs = dataset['inputs'][...].astype('float32')
            self.outputs = dataset['outputs']

    def __getitem__(self, index):
        self._load_dataset()
        return self.inputs[index], self.outputs[index]

    def __len__(self):
        self._load_dataset()
        return len(self.inputs)
