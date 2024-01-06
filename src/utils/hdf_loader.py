from h5py import File
import torch


class H5Dataset(torch.utils.data.Dataset):

    def __init__(self, path):
        self.path = path
        self.inputs = None
        self.outputs = None

    def _load_dataset(self):
        if self.inputs is None:
            dataset = File(self.path, 'r')
            self.inputs = dataset['inputs']
            self.outputs = dataset['outputs']

    def __getitem__(self, index):
        self._load_dataset()
        return self.inputs[index], self.outputs[index]

    def __len__(self):
        self._load_dataset()
        return len(self.inputs)
