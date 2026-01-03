import torch
from utils.constants import CPU, CUDA, MPS


def torch_grab_device(force_cpu=False, suppress_logs=False):
    if torch.cuda.is_available():
        device = CUDA
    elif torch.backends.mps.is_available():
        device = MPS
    else:
        device = CPU
    if force_cpu is True:
        device = CPU
    if not suppress_logs:
        print(f'Device: {device}')
    return device
