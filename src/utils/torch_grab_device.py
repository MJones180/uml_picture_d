import torch


def torch_grab_device(force_cpu=False):
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    if force_cpu is True:
        device = 'cpu'
    print(f'Device: {device}')
    return device
