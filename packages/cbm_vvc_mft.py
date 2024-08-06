"""
IDL code converted to Python by Michael Jones.
Michael_Jones6@student.uml.edu

Beware: while this code may have been converted, it is a terrifying mess and
contains only the original comments. This script works, but it has not been
neatned or optimized, so hopefully it never needs to be touched again.

When I went back to the code to do some benchmarking, I realized that PyTorch
performs a lot of the operations faster than NumPy does. I am not sure why
this would be, it could be very system dependent. Due to this, I created a copy
of the code that uses PyTorch instead for those systems where it ends up being
faster. I know I should have created a single variable that references the
correct package, but I was not sure if the functions would be a one-to-one match
between NumPy and PyTorch (they end up not being exactly the same). That is why
I ended up just duplicating the code a second time and swapped out references.
"""

import numpy as np
import proper
import torch

# ==============================================================================
# TORCH VERSION OF THE CODE However, some portions still use NumPy as they were
# already fast and it makes converting way easier.
# ==============================================================================


def draw_ellipse_torch(n, xrad, yrad, xcenter, ycenter, dark=False):
    t = torch.arange(1000) / 999 * 2 * torch.pi
    xel = xrad * torch.cos(t)
    yel = yrad * torch.sin(t)
    xdiff = xel[1:] - xel[:999]
    ydiff = yel[1:] - yel[:999]
    dt = t[1] / torch.max(torch.sqrt(xdiff**2 + ydiff**2))
    dt = dt / 100
    nt = int(2 * torch.pi / dt)
    t = torch.arange(nt) / (nt - 1) * 2 * torch.pi
    xel = xrad * torch.cos(t) + xcenter
    yel = yrad * torch.sin(t) + ycenter
    w = torch.where(xel - torch.fix(xel) == 0.5)[0]
    if len(w) != 0:
        xel[w] = xel[w] - 0.00001
    w = torch.where(xel - torch.fix(xel) == -0.5)[0]
    if len(w) != 0:
        xel[w] = xel[w] + 0.00001
    w = torch.where(yel - torch.fix(yel) == 0.5)[0]
    if len(w) != 0:
        yel[w] = yel[w] - 0.00001
    w = torch.where(yel - torch.fix(yel) == -0.5)[0]
    if len(w) != 0:
        yel[w] = yel[w] + 0.00001
    xel = torch.round(xel).int().cpu().numpy()
    yel = torch.round(yel).int().cpu().numpy()
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


def fftsw_torch(data, direction=-1):
    nx, ny = data.shape
    a = torch.roll(data, (-nx // 2, -ny // 2), (0, 1))
    direction_str = 'FFTW_FORWARD' if direction == -1 else 'FFTW_BACKWARD'
    a = a.cpu().numpy()
    proper.prop_fftw(a, direction_str)
    a = torch.from_numpy(a)
    a = torch.roll(a, (nx // 2, ny // 2), (0, 1))
    return a


def cos_window_torch(n, rad_pix, outer_fraction):
    x = torch.repeat_interleave(torch.arange(n)[None, :], n, dim=0) - n // 2
    r = torch.sqrt(x**2 + (x.T)**2) / rad_pix
    inner_fraction = 1 - outer_fraction
    cos_term = torch.cos(torch.pi * (r - inner_fraction) / outer_fraction)
    m = (cos_term + 1) * (r <= 1) / 2
    m[r < inner_fraction] = 1.0
    return m


def trim_torch(image, new_n=512):
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
        new_image = torch.empty((new_n, new_n), dtype=image.dtype)
        new_image[new_n2 - n2:new_n2 + n2, new_n2 - n2:new_n2 + n2] = image
        return new_image


def mft2_torch(in_arr,
               dout,
               D,
               nout,
               direction,
               xoffset=0,
               yoffset=0,
               xc=0,
               yc=0):
    # -----------------------------------------------------------------------
    # Compute a matrix fourier transform.  Based on Soummer et al. 2007.
    # Written by Dimitri Mawet (JPL)
    # March 2010
    # Copyright 2012 California Institute of Technology
    # ------------------------------------------------------------------------
    nin = in_arr.shape[0]
    b1 = torch.arange(nin) - nin / 2
    x = b1 - xc
    y = b1 - yc
    nout = np.fix(nout)
    b2 = torch.arange(nout) - nout / 2
    u = (b2 - xoffset / dout) * dout / D
    v = (b2 - yoffset / dout) * dout / D
    two_pi_i = 2 * torch.pi * 1j
    if direction == -1:
        two_pi_i *= -1
    exp_xu = torch.exp(two_pi_i * torch.outer(x, u)).type(torch.complex128)
    exp_vy = torch.exp(two_pi_i * torch.outer(v, y)).type(torch.complex128)
    chain = torch.matmul(exp_vy, torch.matmul(in_arr, exp_xu))
    return dout / D * chain


def cbm_vvc_mft_torch(
    wavefront,
    charge,
    spot_rad,
    offset,
    ramp_sign,
    beam_ratio,
    d_occulter_lyotcoll,
    fl_lyotcoll,
    d_lyotcoll_lyotstop,
):
    # Get parameters
    n = proper.prop_get_gridsize(wavefront)
    sampling = proper.prop_get_sampling(wavefront)
    pupil_diam_pix = n * beam_ratio
    nvvc = n * 4
    # Note: the only way i've found to improve contrast is to increase the
    # simulation gridsize (n). Each factor of 2 in gridsize lowers the contrast
    # floor by roughly an order of magnitude.

    # Phase ramp construction

    # Need to add in correct dimensions, (nvvc,1) x (1, nvvc) -> (nvvc, nvvc)
    y = torch.outer(torch.arange(nvvc) - nvvc / 2, torch.ones(nvvc))
    # x = y.T
    theta = torch.arctan(y / y.T)
    vvc = torch.exp((offset + ramp_sign * charge * theta) * 1j)
    # Middle pixel will be NaN, so set to 0
    vvc[nvvc // 2, nvvc // 2] = 0

    # Propagate a copy of the wavefront to the lyot stop pupil
    proper.prop_propagate(wavefront, d_occulter_lyotcoll)
    proper.prop_lens(wavefront, fl_lyotcoll)
    proper.prop_propagate(wavefront, d_lyotcoll_lyotstop)
    pupil = proper.prop_get_wavefront(wavefront)

    # Resample the psf at two different resolutions (inner,outer)

    # Define the inner/outer boundary in l/d
    # Magnified region lambda/d radius
    window_rad_lod = np.max((1.1 * (spot_rad / sampling) * beam_ratio, 2))
    window_rolloff = 0.5

    # Calculate the outer field, multiple by window and vvc
    window_rad_pix = window_rad_lod * nvvc / (beam_ratio * n)

    cwindow = 1 - cos_window_torch(nvvc, window_rad_pix, window_rolloff)
    # To focus using FFTW
    outer = fftsw_torch(trim_torch(torch.from_numpy(pupil), nvvc),
                        -1) * cwindow * trim_torch(vvc, nvvc)
    # To pupil using FFTW
    pupil_outer = trim_torch(fftsw_torch(outer, 1), n)

    # Calculate the inner field, multiple by window, spot and vvc.
    # MFT the original wavefront to the fpm at high sampling, apply the reverse
    # tapered window to block out the region outside of "inner_rad", and apply
    # the vortex and occulting spot. Using mft to create the high-res psf
    # because in order to use fft, we would need pad the native pupil image to
    # an array ~40x larger to get the right beam ratio.

    # -2 ensures a perimeter of zeros around the window function
    window_rad_pix = n / 2 - 2
    inner_dx_lod = window_rad_lod / window_rad_pix
    cwindow = cos_window_torch(n, window_rad_pix, window_rolloff)
    spot = 1
    if spot_rad > 0:
        spot_rad_pix = (spot_rad / sampling) * (beam_ratio / inner_dx_lod)
        spot = draw_ellipse_torch(
            n,
            spot_rad_pix,
            spot_rad_pix,
            n // 2,
            n // 2,
            dark=True,
        )
    inner = mft2_torch(torch.from_numpy(pupil), inner_dx_lod, pupil_diam_pix,
                       n, -1) * cwindow * spot * trim_torch(vvc, n)
    pupil_inner = mft2_torch(inner, inner_dx_lod, pupil_diam_pix, n, 1)

    # Combine fields
    pupil = pupil_inner + pupil_outer
    pupil = pupil.cpu().numpy()

    # Prop back
    wavefront.wfarr = proper.prop_shift_center(pupil)
    proper.prop_propagate(wavefront, -d_lyotcoll_lyotstop)
    proper.prop_lens(wavefront, -fl_lyotcoll)
    proper.prop_propagate(wavefront, -d_occulter_lyotcoll)


# ==============================================================================
# NUMPY VERSION OF THE CODE
# ==============================================================================


def draw_ellipse_np(n, xrad, yrad, xcenter, ycenter, dark=False):
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


def fftsw_np(data, direction=-1):
    nx, ny = data.shape
    a = np.roll(data, (-nx // 2, -ny // 2), (0, 1))
    direction_str = 'FFTW_FORWARD' if direction == -1 else 'FFTW_BACKWARD'
    proper.prop_fftw(a, direction_str)
    a = np.roll(a, (nx // 2, ny // 2), (0, 1))
    return a


def cos_window_np(n, rad_pix, outer_fraction):
    x = np.repeat(np.arange(n)[None, :], n, axis=0) - n // 2
    r = np.sqrt(x**2 + (x.T)**2) / rad_pix
    inner_fraction = 1 - outer_fraction
    cos_term = np.cos(np.pi * (r - inner_fraction) / outer_fraction)
    m = (cos_term + 1) * (r <= 1) / 2
    m[r < inner_fraction] = 1.0
    return m


def trim_np(image, new_n=512):
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


def mft2_np(in_arr,
            dout,
            D,
            nout,
            direction,
            xoffset=0,
            yoffset=0,
            xc=0,
            yc=0):
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


def cbm_vvc_mft_np(
    wavefront,
    charge,
    spot_rad,
    offset,
    ramp_sign,
    beam_ratio,
    d_occulter_lyotcoll,
    fl_lyotcoll,
    d_lyotcoll_lyotstop,
):
    # Get parameters
    n = proper.prop_get_gridsize(wavefront)
    sampling = proper.prop_get_sampling(wavefront)
    pupil_diam_pix = n * beam_ratio
    nvvc = n * 4
    # Note: the only way i've found to improve contrast is to increase the
    # simulation gridsize (n). Each factor of 2 in gridsize lowers the contrast
    # floor by roughly an order of magnitude.

    # Phase ramp construction

    # Need to add in correct dimensions, (nvvc,1) x (1, nvvc) -> (nvvc, nvvc)
    y = np.outer(np.arange(nvvc) - nvvc / 2, np.ones(nvvc))
    # A divide by zero occurs from the `x` array, so ignore the warning
    with np.errstate(all='ignore'):
        # x = y.T
        theta = np.arctan(y / y.T)
    vvc = np.exp((offset + ramp_sign * charge * theta) * 1j)
    # Middle pixel will be NaN, so set to 0
    vvc[nvvc // 2, nvvc // 2] = 0

    # Propagate a copy of the wavefront to the lyot stop pupil
    proper.prop_propagate(wavefront, d_occulter_lyotcoll)
    proper.prop_lens(wavefront, fl_lyotcoll)
    proper.prop_propagate(wavefront, d_lyotcoll_lyotstop)
    pupil = proper.prop_get_wavefront(wavefront)

    # Resample the psf at two different resolutions (inner,outer)

    # Define the inner/outer boundary in l/d
    # Magnified region lambda/d radius
    window_rad_lod = np.max((1.1 * (spot_rad / sampling) * beam_ratio, 2))
    window_rolloff = 0.5

    # Calculate the outer field, multiple by window and vvc
    window_rad_pix = window_rad_lod * nvvc / (beam_ratio * n)

    cwindow = 1 - cos_window_np(nvvc, window_rad_pix, window_rolloff)
    # To focus using FFTW
    outer = fftsw_np(trim_np(pupil, nvvc), -1) * cwindow * trim_np(vvc, nvvc)
    # To pupil using FFTW
    pupil_outer = trim_np(fftsw_np(outer, 1), n)

    # Calculate the inner field, multiple by window, spot and vvc.
    # MFT the original wavefront to the fpm at high sampling, apply the reverse
    # tapered window to block out the region outside of "inner_rad", and apply
    # the vortex and occulting spot. Using mft to create the high-res psf
    # because in order to use fft, we would need pad the native pupil image to
    # an array ~40x larger to get the right beam ratio.

    # -2 ensures a perimeter of zeros around the window function
    window_rad_pix = n / 2 - 2
    inner_dx_lod = window_rad_lod / window_rad_pix
    cwindow = cos_window_np(n, window_rad_pix, window_rolloff)
    spot = 1
    if spot_rad > 0:
        spot_rad_pix = (spot_rad / sampling) * (beam_ratio / inner_dx_lod)
        spot = draw_ellipse_np(
            n,
            spot_rad_pix,
            spot_rad_pix,
            n // 2,
            n // 2,
            dark=True,
        )
    inner = mft2_np(pupil, inner_dx_lod, pupil_diam_pix, n,
                    -1) * cwindow * spot * trim_np(vvc, n)
    pupil_inner = mft2_np(inner, inner_dx_lod, pupil_diam_pix, n, 1)

    # Combine fields
    pupil = pupil_inner + pupil_outer

    # Prop back
    wavefront.wfarr = proper.prop_shift_center(pupil)
    proper.prop_propagate(wavefront, -d_lyotcoll_lyotstop)
    proper.prop_lens(wavefront, -fl_lyotcoll)
    proper.prop_propagate(wavefront, -d_occulter_lyotcoll)


# ==============================================================================
# CHOOSE WHICH ONE
# ==============================================================================


def cbm_vvc_mft(
    wavefront,
    charge,
    spot_rad,
    offset,
    ramp_sign,
    beam_ratio,
    d_occulter_lyotcoll,
    fl_lyotcoll,
    d_lyotcoll_lyotstop,
    use_torch=False,
):
    func = cbm_vvc_mft_torch if use_torch else cbm_vvc_mft_np
    func(
        wavefront,
        charge,
        spot_rad,
        offset,
        ramp_sign,
        beam_ratio,
        d_occulter_lyotcoll,
        fl_lyotcoll,
        d_lyotcoll_lyotstop,
    )
