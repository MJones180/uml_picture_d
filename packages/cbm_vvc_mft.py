"""
IDL code converted to Python by Michael Jones.
Michael_Jones6@student.uml.edu

Beware: while this code may have been converted, it is a terrifying mess and
contains only the original comments. This script works, but it has not been
neatened and does not have added comments for clarity.
"""

import numpy as np
import proper


def draw_ellipse(n, xrad, yrad, xcenter, ycenter, dark=False):
    t = np.arange(1000) / 999 * 2 * np.pi
    xel = xrad * np.cos(t)
    yel = yrad * np.sin(t)
    xdiff = xel[1:] - xel[:999]
    ydiff = yel[1:] - yel[:999]
    dt = t[1] / np.max(np.sqrt(xdiff**2 + ydiff**2))
    dt = dt / 100
    nt = int(2 * np.pi / dt)
    t = np.arange(nt) / (nt - 1) * 2 * np.pi
    xel = xrad * np.cos(t) + xcenter
    yel = yrad * np.sin(t) + ycenter
    w = np.where(xel - np.fix(xel) == 0.5)[0]
    if len(w) != 0:
        xel[w] = xel[w] - 0.00001
    w = np.where(xel - np.fix(xel) == -0.5)[0]
    if len(w) != 0:
        xel[w] = xel[w] + 0.00001
    w = np.where(yel - np.fix(yel) == 0.5)[0]
    if len(w) != 0:
        yel[w] = yel[w] - 0.00001
    w = np.where(yel - np.fix(yel) == -0.5)[0]
    if len(w) != 0:
        yel[w] = yel[w] + 0.00001
    xel = np.round(xel).astype(int)
    yel = np.round(yel).astype(int)
    mask = np.zeros((n, n))
    mask[xel, yel] = 1
    xcenter = np.round(xcenter)
    ycenter = np.round(ycenter)
    for j in range(np.max((np.min(yel) + 1, 0)), np.min((np.max(yel), n))):
        w = np.where(mask[:xcenter + 1, j] != 0)[0]
        len_w = len(w)
        if len_w != 0 and w[len_w - 1] != xcenter:
            mask[w[len_w - 1] + 1:xcenter + 1, j] = 1
        w = np.where(mask[xcenter + 1:, j] != 0)[0]
        if len(w) != 0 and w[0] != 0:
            mask[xcenter + 1:xcenter + w[0] + 1, j] = 1
    if dark is True:
        mask = 1.0 - mask
    return np.minimum(np.maximum(mask, 0), 1)


def fftsw(data, direction=-1):
    nx, ny = data.shape
    a = np.roll(data, (-nx // 2, -ny // 2), (0, 1))
    direction_str = 'FFTW_FORWARD' if direction == -1 else 'FFTW_BACKWARD'
    proper.prop_fftw(a, direction_str)
    a = np.roll(a, (nx // 2, ny // 2), (0, 1))
    return a


def cos_window(n, rad_pix, outer_fraction):
    x = np.repeat(np.arange(n)[None, :], n, axis=0) - n // 2
    r = np.sqrt(x**2 + (x.T)**2) / rad_pix
    inner_fraction = 1 - outer_fraction
    cos_term = np.cos(np.pi * (r - inner_fraction) / outer_fraction)
    m = (cos_term + 1) * (r <= 1) / 2
    m[r < inner_fraction] = 1.0
    return m


def trim(image, new_n=512):
    n = image.shape[0]
    if n == new_n:
        return image
    n2 = n // 2
    new_n2 = new_n // 2
    if new_n < n:
        # Trimming image
        return image[n2 - new_n2:n2 + new_n2, n2 - new_n2:n2 + new_n2]
    else:
        # Embedding image into larger one
        new_image = np.empty((new_n, new_n), dtype=image.dtype)
        new_image[new_n2 - n2:new_n2 + n2, new_n2 - n2:new_n2 + n2] = image
        return new_image


def mft2(in_arr, dout, D, nout, direction, xoffset=0, yoffset=0, xc=0, yc=0):
    # -----------------------------------------------------------------------
    # Compute a matrix fourier transform.  Based on Soummer et al. 2007.
    # Written by Dimitri Mawet (JPL)
    # March 2010
    # Copyright 2012 California Institute of Technology
    # ------------------------------------------------------------------------
    nin = in_arr.shape[0]
    b1 = np.arange(nin) - nin / 2
    x = b1 - xc
    y = b1 - yc
    nout = np.fix(nout)
    b2 = np.arange(nout) - nout / 2
    u = (b2 - xoffset / dout) * dout / D
    v = (b2 - yoffset / dout) * dout / D
    two_pi_i = 2 * np.pi * 1j
    if direction == -1:
        two_pi_i *= -1
    exp_xu = np.exp(two_pi_i * np.outer(x, u))
    exp_vy = np.exp(two_pi_i * np.outer(v, y))
    chain = exp_vy @ (in_arr @ exp_xu)
    return dout / D * chain


# Save the VVC between function calls so that it does not need to be computed
# each time. It will be re-calculated if its parameters change.
vvc = None
vvc_args_gl = ()


def obtain_vvc(nvvc, charge, offset, ramp_sign):
    global vvc
    global vvc_args_gl
    # Reset the VVC and set the args that will be used to create it if this is
    # the first run or if the args have changed
    vvc_args = (nvvc, charge, offset, ramp_sign)
    if vvc_args_gl != vvc_args:
        vvc_args_gl = vvc_args
        vvc = None
    # Compute the VVC if it is either the first run or it has been changed
    if vvc is None:
        # Need to add in correct dimensions, (nvvc,1)x(1, nvvc) -> (nvvc, nvvc)
        y = np.outer(np.arange(nvvc) - nvvc / 2, np.ones(nvvc))
        # A divide by zero occurs from the `x` array, so ignore the warning
        with np.errstate(all='ignore'):
            # x = y.T
            theta = np.arctan(y / y.T)
        vvc = np.exp((offset + ramp_sign * charge * theta) * 1j)
        # Middle pixel will be NaN, so set to 0
        vvc[nvvc // 2, nvvc // 2] = 0
    return vvc


# This is an approximation of the complete `cbm_vvc_mft`. Instead of doing the
# full MFT, this function just multiplies the wavefront by the VVC mask. This
# function will be much faster at the cost of a minor loss in accuracy.
# The `center_spot_scaling` parameter needs to be tailored for each set of VVC
# args so that it best aligns with the results obtained by calling the complete
# `cbm_vvc_mft` function.
def cbm_vvc_approx(wavefront, charge, offset, ramp_sign, center_spot_scaling):
    n = proper.prop_get_gridsize(wavefront)
    sampling = proper.prop_get_sampling(wavefront)
    vvc = obtain_vvc(n, charge, offset, ramp_sign)
    rad = center_spot_scaling * sampling
    center_dot = proper.prop_ellipse(wavefront, rad, rad, 0, 0, DARK=True)
    # Multiply input wavefront by vvc with the center dot
    proper.prop_multiply(wavefront, vvc * center_dot)


def cbm_vvc_mft(
    wavefront,
    charge,
    offset,
    ramp_sign,
    spot_rad,
    beam_ratio,
    d_occulter_lyotcoll,
    fl_lyotcoll,
    d_lyotcoll_lyotstop,
):
    # Get parameters
    n = proper.prop_get_gridsize(wavefront)
    sampling = proper.prop_get_sampling(wavefront)
    pupil_diam_pix = n * beam_ratio
    # Note: the only way i've found to improve contrast is to increase the
    # simulation gridsize (n). Each factor of 2 in gridsize lowers the
    # contrast floor by roughly an order of magnitude.
    nvvc = n * 4

    # =======================
    # Phase ramp construction
    # =======================

    vvc = obtain_vvc(nvvc, charge, offset, ramp_sign)

    # ============================================================
    # Resample the psf at two different resolutions (inner,outer).
    # This bit uses MFT.
    # ============================================================

    # Propagate a copy of the wavefront to the lyot stop pupil
    proper.prop_propagate(wavefront, d_occulter_lyotcoll)
    proper.prop_lens(wavefront, fl_lyotcoll)
    proper.prop_propagate(wavefront, d_lyotcoll_lyotstop)
    pupil = proper.prop_get_wavefront(wavefront)

    # Define the inner/outer boundary in l/d
    # Magnified region lambda/d radius
    window_rad_lod = np.max((1.1 * (spot_rad / sampling) * beam_ratio, 2))
    window_rolloff = 0.5

    # Calculate the outer field, multiple by window and vvc
    window_rad_pix = window_rad_lod * nvvc / (beam_ratio * n)

    cwindow = 1 - cos_window(nvvc, window_rad_pix, window_rolloff)
    # To focus using FFTW
    outer = fftsw(trim(pupil, nvvc), -1) * cwindow * trim(vvc, nvvc)
    # To pupil using FFTW
    pupil_outer = trim(fftsw(outer, 1), n)

    # Calculate the inner field, multiple by window, spot and vvc.
    # MFT the original wavefront to the fpm at high sampling, apply the reverse
    # tapered window to block out the region outside of "inner_rad", and apply
    # the vortex and occulting spot. Using mft to create the high-res psf
    # because in order to use fft, we would need pad the native pupil image to
    # an array ~40x larger to get the right beam ratio.

    # -2 ensures a perimeter of zeros around the window function
    window_rad_pix = n / 2 - 2
    inner_dx_lod = window_rad_lod / window_rad_pix
    cwindow = cos_window(n, window_rad_pix, window_rolloff)
    spot = 1
    if spot_rad > 0:
        spot_rad_pix = (spot_rad / sampling) * (beam_ratio / inner_dx_lod)
        spot = draw_ellipse(
            n,
            spot_rad_pix,
            spot_rad_pix,
            n // 2,
            n // 2,
            dark=True,
        )
    inner = mft2(pupil, inner_dx_lod, pupil_diam_pix, n,
                 -1) * cwindow * spot * trim(vvc, n)
    pupil_inner = mft2(inner, inner_dx_lod, pupil_diam_pix, n, 1)

    # Combine fields
    pupil = pupil_inner + pupil_outer

    # Prop back
    wavefront.wfarr = proper.prop_shift_center(pupil)
    proper.prop_propagate(wavefront, -d_lyotcoll_lyotstop)
    proper.prop_lens(wavefront, -fl_lyotcoll)
    proper.prop_propagate(wavefront, -d_occulter_lyotcoll)
