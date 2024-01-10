import json
import numpy as np


# https://stackoverflow.com/a/47626762
# Removes the "JSON serializable" error that often arises with numpy arrays
class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def json_load(path):
    """Read in data from JSON.

    Parameters
    ----------
    path : str
        The path to open.
    """

    with open(path, 'r') as f:
        data = json.load(f)
    return data


def json_write(output_path, data, mode='w'):
    """Write out data to JSON.

    Parameters
    ----------
    output_path : str
        The path and file to write out to.
    data : dict
        The data to write out to JSON.
    mode : str, optional
        The output mode for the JSON file.
    """

    with open(output_path, mode) as output_file:
        json.dump(
            data,
            output_file,
            ensure_ascii=False,
            indent=4,
            cls=NumpyEncoder,
        )
