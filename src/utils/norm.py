import numpy as np


def sum_to_one(data, dims=None):
    """Have the values sum to one.

    Parameters
    ----------
    data : np.array
        The data to have sum to one.
    dims : int or list, optional
        The dimensions to sum along, by default will sum along every dim.

    Returns
    -----
    np.array
        The data that sums to one.
    """

    return data / np.sum(data, axis=dims, keepdims=True)


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

    Notes
    -----
    A value of 1e-10 is added to the denominator for numerical stability.
    """

    # (data - min_x) / (max_x - min_x)
    # -> (data - min_x) / max_min_diff
    norm = (data - min_x) / (max_min_diff + 1e-10)
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


def find_min_max_norm(
    data,
    globally=False,
    ones_range=False,
    round_values=None,
):
    """Find the max_min_diff and min_x and then normalize the data.

    Parameters
    ----------
    data : np.array
        The data to normalize.
    globally : bool, optional
        Whether to normalize across columns.
    ones_range : bool
        If true, will normalize between -1 and 1 instead of 0 to 1.
    round_values : int, optional
        Round the `max_min_diff` and `min_x` to n decimal places as given by
        this argument.

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
    if round_values is not None:
        print(f'Original min_x: {min_x}')
        print(f'Original max_min_diff: {max_min_diff}')
        print(f'Rounding values to {round_values} decimal places')
        min_x = np.round(min_x, round_values)
        max_min_diff = np.round(max_min_diff, round_values)
        print(f'Rounded min_x: {min_x}')
        print(f'Rounded max_min_diff: {max_min_diff}')
    normalized = min_max_norm(data, max_min_diff, min_x, ones_range)
    return normalized, max_min_diff, min_x
