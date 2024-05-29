import numpy as np


def _agg_wrapper(x1, x2, sum_axes, element_wise_func, method='mean'):
    diff = element_wise_func(x1, x2)
    # None equates to all axes, so can either pass in 'all' or None
    if sum_axes == 'all':
        sum_axes = None
    method_func = np.mean if method == 'mean' else np.sum
    return method_func(diff, sum_axes)


def element_wise_abs_error(x1, x2):
    return np.abs(x1 - x2)


def element_wise_square_error(x1, x2):
    return (x1 - x2)**2


def mae(x1, x2, sum_axes):
    return _agg_wrapper(x1, x2, sum_axes, element_wise_abs_error, 'mean')


def mse(x1, x2, sum_axes):
    return _agg_wrapper(x1, x2, sum_axes, element_wise_square_error, 'mean')


def rmse(x1, x2, sum_axes):
    return mse(x1, x2, sum_axes)**0.5


# Root sum of squares
def rss(x1, x2, sum_axes):
    return _agg_wrapper(x1, x2, sum_axes, element_wise_square_error,
                        'sum')**0.5
