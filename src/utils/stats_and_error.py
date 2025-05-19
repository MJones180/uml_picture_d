"""
The following statistical functions are implemented (some might not be useful
or standard, but they are all there just in case):
    Function Name         Acronym  Equation
    -------------         -------  --------
    sum_of_abs                     Σ_i abs(x_i)
    mean_of_abs                    Σ_i abs(x_i) / N
    root_sum_of_abs                sqrt( Σ_i abs(x_i) )
    root_mean_of_abs               sqrt( Σ_i abs(x_i) / N )
    sum_of_squares                 Σ_i (x_i)**2
    mean_of_squares                Σ_i (x_i)**2 / N
    root_sum_of_squares   RSS      sqrt( Σ_i (x_i)**2 )
    root_mean_of_squares  RMS      sqrt( Σ_i (x_i)**2 / N )

The following error metrics are implemented (takes two arrays):
    Function Name           Acronym   Equation
    -------------           -------   --------
    mean_absolute_error     MAE       Σ_i abs(x1_i - x2_i) / N
    mean_square_error       MSE       Σ_i (x1_i - x2_i)**2 / N
    root_mean_square_error  RMSE      sqrt( Σ_i (x1_i - x2_i)**2 / N )
    percent_error                     abs((x2 - x1)/x1); x1 is truth

If a function has an acronym, it can be access via that instead.

All functions take the `axes` argument, it is the axes that should be summed or
averaged over. The value must be either None (all axes), a single integer, or a
tuple of integers.
"""

import numpy as np

# ==============================================================================
# Statistical functions


def sum_of_abs(x, axes=None):
    return np.sum(np.abs(x), axes)


def mean_of_abs(x, axes=None):
    return np.mean(np.abs(x), axes)


def root_sum_of_abs(x, axes=None):
    return sum_of_abs(x, axes)**0.5


def root_mean_of_abs(x, axes=None):
    return mean_of_abs(x, axes)**0.5


def sum_of_squares(x, axes=None):
    return np.sum(x**2, axes)


def mean_of_squares(x, axes=None):
    return np.mean(x**2, axes)


def root_sum_of_squares(x, axes=None):
    return sum_of_squares(x, axes)**0.5


def root_mean_of_squares(x, axes=None):
    return mean_of_squares(x, axes)**0.5


# ==============================================================================
# Error functions


def mean_absolute_error(x1, x2, axes=None):
    return mean_of_abs(x1 - x2, axes)


def mean_square_error(x1, x2, axes=None):
    return mean_of_squares(x1 - x2, axes)


def root_mean_square_error(x1, x2, axes=None):
    return root_mean_of_squares(x1 - x2, axes)


def percent_error(truth, x):
    return np.abs((x - truth) / truth)


# ==============================================================================
# Shorthand functions


def rss(*args, **kwargs):
    return root_sum_of_squares(*args, **kwargs)


def rms(*args, **kwargs):
    return root_mean_of_squares(*args, **kwargs)


def mae(*args, **kwargs):
    return mean_absolute_error(*args, **kwargs)


def mse(*args, **kwargs):
    return mean_square_error(*args, **kwargs)


def rmse(*args, **kwargs):
    return root_mean_square_error(*args, **kwargs)
