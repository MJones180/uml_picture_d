import numpy as np


def min_max_norm(data, max_min_diff, min_x):
    """Min max normalize values given the max_min_diff and min_x.

    Parameters
    ----------
    data : np.array
        The data to normalize.
    max_min_diff : float or np.array
        Difference between the min and max values.
    min_x : float or np.array
        The minimum value.

    Returns
    -----
    np.array
        The normalized data.
    """

    # (data - min_x) / (max_x - min_x)
    # -> (data - min_x) / max_min_diff
    return (data - min_x) / max_min_diff


def min_max_denorm(data, max_min_diff, min_x):
    """Min max denormalize values given the max_min_diff and min_x.

    Parameters
    ----------
    data : np.array
        The data to normalize.
    max_min_diff : float or np.array
        Difference between the min and max values.
    min_x : float or np.array
        The minimum value.

    Returns
    -----
    np.array
        The denormalized data.
    """

    return (data * max_min_diff) + min_x


def find_min_max_norm(data, globally=False):
    """Find the max_min_diff and min_x and then normalize the data.

    Parameters
    ----------
    data : np.array
        The data to normalize.
    globally : bool, optional
        Whether to normalize across columns.

    Returns
    -----
    tuple
        (normalized data, max_min_diff, min_x)
    """

    # All elements are equal, set everything to zero for the constant case
    if np.all(data == data[0]):
        data = np.zeros(data.shape)
        return data, data[0], data[0]
    if globally:
        min_x = data.min()
        max_min_diff = data.max() - min_x
    else:
        min_x = data.min(axis=0)
        max_min_diff = data.max(axis=0) - min_x
    normalized = min_max_norm(data, max_min_diff, min_x)
    return normalized, max_min_diff, min_x
