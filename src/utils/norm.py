import numpy as np


def min_max_norm(data, max_min_diff, min_x, ones_range=False):
    """Min max normalize values given the max_min_diff and min_x.

    Parameters
    ----------
    data : np.array
        The data to normalize.
    max_min_diff : float or np.array
        Difference between the min and max values.
    min_x : float or np.array
        The minimum value.
    ones_range : bool
        If true, will normalize between -1 and 1 instead of 0 to 1.

    Returns
    -----
    np.array
        The normalized data.
    """

    # (data - min_x) / (max_x - min_x)
    # -> (data - min_x) / max_min_diff
    norm = (data - min_x) / max_min_diff
    if ones_range:
        # -> 2 * [ (data - min_x) / max_min_diff ] - 1
        return 2 * norm - 1
    return norm


def min_max_denorm(data, max_min_diff, min_x, ones_range=False):
    """Min max denormalize values given the max_min_diff and min_x.

    Parameters
    ----------
    data : np.array
        The data to denormalize.
    max_min_diff : float or np.array
        Difference between the min and max values.
    min_x : float or np.array
        The minimum value.
    ones_range : bool
        If true, data was normalized between -1 and 1 instead of 0 to 1.

    Returns
    -----
    np.array
        The denormalized data.
    """

    if ones_range:
        return (((data + 1) / 2) * max_min_diff) + min_x
    # norm = (original - min_x) / max_min_diff
    # -> original = (norm * max_min_diff) + min_x
    return (data * max_min_diff) + min_x


def find_min_max_norm(data, globally=False, ones_range=False):
    """Find the max_min_diff and min_x and then normalize the data.

    Parameters
    ----------
    data : np.array
        The data to normalize.
    globally : bool, optional
        Whether to normalize across columns.
    ones_range : bool
        If true, will normalize between -1 and 1 instead of 0 to 1.

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
    normalized = min_max_norm(data, max_min_diff, min_x, ones_range)
    return normalized, max_min_diff, min_x
