import numpy as np


def element_wise_abs_error(x1, x2):
    return np.abs(x1 - x2)


def element_wise_square_error(x1, x2):
    return (x1 - x2)**2


def _mean_wrapper(x1, x2, sum_axes, element_wise_func):
    diff = element_wise_func(x1, x2)
    # None equates to all axes, so can either pass in 'all' or None
    if sum_axes == 'all':
        sum_axes = None
    return np.mean(diff, sum_axes)


def mae(x1, x2, sum_axes):
    return _mean_wrapper(x1, x2, sum_axes, element_wise_abs_error)


def mse(x1, x2, sum_axes):
    return _mean_wrapper(x1, x2, sum_axes, element_wise_square_error)


def rmse(x1, x2, sum_axes):
    return mse(x1, x2, sum_axes)**0.5
