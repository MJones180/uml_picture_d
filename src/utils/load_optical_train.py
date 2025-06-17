from importlib import import_module
from utils.constants import SIM_OPTICAL_TRAINS
from utils.terminate_with_message import terminate_with_message


def load_optical_train(name):
    try:
        module = import_module(f'{SIM_OPTICAL_TRAINS}.{name}')
    except ModuleNotFoundError:
        terminate_with_message(f'No optical train by the name of `{name}` '
                               f'found within the `{SIM_OPTICAL_TRAINS}` '
                               'folder.')
    # The module itself is returned so extra variables can be grabbed if needed
    return (module.INIT_BEAM_D, module.BEAM_RATIO, module.OPTICAL_TRAIN,
            module.CAMERA_PIXELS, module.CAMERA_SAMPLING, module)
