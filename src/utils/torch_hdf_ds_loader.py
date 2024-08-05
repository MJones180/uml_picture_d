import torch
from utils.constants import DATA_F, INPUTS, OUTPUTS, PROC_DATA_P
from utils.hdf_read_and_write import read_hdf
from utils.path import path_exists
from utils.terminate_with_message import terminate_with_message


class DSLoaderHDF(torch.utils.data.Dataset):
    """
    HDF dataset loader using H5py for Torch.
    Dataset H5 files must have 'inputs' and 'outputs' tables.
    """

    def __init__(self, dataset_name=None, path=None):
        self.inputs = None
        self.outputs = None
        if dataset_name is not None:
            path = f'{PROC_DATA_P}/{dataset_name}/{DATA_F}'
        self.path = path
        if not path_exists(path):
            terminate_with_message(f'Dataset not found at {path}')

    def get_path(self):
        return self.path

    def get_inputs(self):
        self._load_dataset()
        return self.inputs

    def get_outputs(self):
        self._load_dataset()
        return self.outputs

    def get_inputs_torch(self):
        return torch.from_numpy(self.get_inputs())

    def get_outputs_torch(self):
        return torch.from_numpy(self.get_outputs())

    def _load_dataset(self):
        if self.inputs is None:
            dataset = read_hdf(self.path)
            # Make both inputs and outputs float32 instead of float64
            self.inputs = dataset[INPUTS][...].astype('float32')
            self.outputs = dataset[OUTPUTS][...].astype('float32')

    def __getitem__(self, index):
        self._load_dataset()
        return self.inputs[index], self.outputs[index]

    def __len__(self):
        self._load_dataset()
        return len(self.inputs)
