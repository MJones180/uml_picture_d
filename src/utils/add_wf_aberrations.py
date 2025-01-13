import numpy as np
import proper

# Cache the Zernike fields so that they do not need to be recomputed for every
# wavefront simulation
zernike_fields_cache = {}
# The number of Zernike equations that are hardcoded
MAX_ZERNIKE_TERM = 28


def add_wf_aberrations(wavefront, zernike_terms, aberration_values):
    """
    Add aberrations to a PROPER wavefront. This code is very fast if the
    Zernike terms are in the range of 2-28 because then the
    `proper.prop_zernikes` function does not need to be called.

    Parameters
    ----------
    wavefront : PROPER Wavefront class object
        The wavefront to add aberrations to.
    zernike_terms : list[int]
        Noll Zernike indexes to add aberrations for.
    aberration_values : list[float]
        Amount of aberration to add to each Zernike term.

    Returns
    -------
    np.array
        The field containing the superposition of all the aberrations.
    """

    zernike_high = zernike_terms[-1]
    # If the equations for the requested Zernike terms are hardcoded at the
    # bottom of this file, then use those. The `proper.prop_zernikes` function
    # does work for any Zernike terms, but it is much slower.
    if zernike_high <= MAX_ZERNIKE_TERM:
        # Grab the parameters from the wavefront
        grid_points = proper.n
        sampling = proper.prop_get_sampling(wavefront)
        beam_rad = proper.prop_get_beamradius(wavefront)
        # The key that the Zernike fields are cached under
        cache_key = (grid_points, sampling, beam_rad, zernike_terms[0],
                     zernike_high)
        # Check if the Zernike fields have been computed yet
        if cache_key in zernike_fields_cache:
            zernike_fields = zernike_fields_cache[cache_key]
        else:
            # The Zernike fields are not yet in the cache
            zernike_fields = _compute_zernike_fields(grid_points, sampling,
                                                     beam_rad, zernike_terms)
            zernike_fields_cache[cache_key] = zernike_fields
        # Compute the field of the superposition of all the Zernike aberrations
        aberration_map = np.zeros((grid_points, grid_points), dtype=np.float64)
        for term, aberration in zip(zernike_terms, aberration_values):
            if aberration != 0:
                aberration_map += zernike_fields[term] * aberration
        # Need to update the actual wavefront with the aberrations
        wavefront.wfarr *= np.exp(1j * 2 * np.pi / wavefront.lamda *
                                  proper.prop_shift_center(aberration_map))
    else:
        # We have to call the slow PROPER function :(
        aberration_map = proper.prop_zernikes(wavefront, zernike_terms,
                                              aberration_values)
    return aberration_map


def _compute_zernike_fields(grid_points, sampling, beam_rad, zernike_terms):
    y_grid = (np.arange(grid_points) - grid_points // 2) * sampling / beam_rad
    # Copy the vector along each column, that means rows have all the same value
    y_grid = np.tile(y_grid[:, None], grid_points)
    x_grid = y_grid.T
    # Shorthand to access the grid values
    x = x_grid
    y = y_grid
    # Math that ends up being done more than once
    xy = x * y
    x2 = x**2
    x4 = x2**2
    y2 = y**2
    xpy = x2 + y2
    xpy2 = xpy**2
    xpy3 = xpy**3
    xmy = x2 - y2
    sqrt6 = np.sqrt(6)
    sqrt8 = np.sqrt(8)
    sqrt10 = np.sqrt(10)
    sqrt12 = np.sqrt(12)
    sqrt14 = np.sqrt(14)
    # Manually compute each Zernike field for terms 2 through 28. The reason
    # that there are a bunch of `if` statements is so that unused Zernike term
    # fields are not computed. The cartesian equations are from the paper
    # "Zernike polynomials and their applications" by Niu and Tian.
    zf = {}
    if 2 in zernike_terms:
        zf[2] = 2 * x
    if 3 in zernike_terms:
        zf[3] = 2 * y
    if 4 in zernike_terms:
        zf[4] = np.sqrt(3) * (2 * xpy - 1)
    if 5 in zernike_terms:
        zf[5] = 2 * sqrt6 * xy
    if 6 in zernike_terms:
        zf[6] = sqrt6 * xmy
    if 7 in zernike_terms:
        zf[7] = sqrt8 * y * (3 * xpy - 2)
    if 8 in zernike_terms:
        zf[8] = sqrt8 * x * (3 * xpy - 2)
    if 9 in zernike_terms:
        zf[9] = sqrt8 * y * (3 * x2 - y2)
    if 10 in zernike_terms:
        zf[10] = sqrt8 * x * (x2 - 3 * y2)
    if 11 in zernike_terms:
        zf[11] = np.sqrt(5) * (6 * xpy2 - 6 * xpy + 1)
    if 12 in zernike_terms:
        zf[12] = sqrt10 * xmy * (4 * xpy - 3)
    if 13 in zernike_terms:
        zf[13] = 2 * sqrt10 * xy * (4 * xpy - 3)
    if 14 in zernike_terms:
        zf[14] = sqrt10 * (xpy2 - 8 * x2 * y2)
    if 15 in zernike_terms:
        zf[15] = 4 * sqrt10 * xy * xmy
    if 16 in zernike_terms:
        zf[16] = sqrt12 * x * (10 * xpy2 - 12 * xpy + 3)
    if 17 in zernike_terms:
        zf[17] = sqrt12 * y * (10 * xpy2 - 12 * xpy + 3)
    if 18 in zernike_terms:
        zf[18] = sqrt12 * x * (x2 - 3 * y2) * (5 * xpy - 4)
    if 19 in zernike_terms:
        zf[19] = sqrt12 * y * (3 * x2 - y2) * (5 * xpy - 4)
    if 20 in zernike_terms:
        zf[20] = sqrt12 * x * (16 * x4 - 20 * x2 * xpy + 5 * xpy2)
    if 21 in zernike_terms:
        zf[21] = sqrt12 * y * (16 * y**4 - 20 * y2 * xpy + 5 * xpy2)
    if 22 in zernike_terms:
        zf[22] = np.sqrt(7) * (20 * xpy3 - 30 * xpy2 + 12 * xpy - 1)
    if 23 in zernike_terms:
        zf[23] = 2 * sqrt14 * xy * (15 * xpy2 - 20 * xpy + 6)
    if 24 in zernike_terms:
        zf[24] = sqrt14 * (x2 - y2) * (15 * xpy2 - 20 * xpy + 6)
    if 25 in zernike_terms:
        zf[25] = 4 * sqrt14 * xy * xmy * (6 * xpy - 5)
    if 26 in zernike_terms:
        zf[26] = sqrt14 * (xpy2 - 8 * x2 * y2) * (6 * xpy - 5)
    if 27 in zernike_terms:
        zf[27] = sqrt14 * xy * (32 * x4 - 32 * x2 * xpy + 6 * xpy2)
    if 28 in zernike_terms:
        zf[28] = sqrt14 * (32 * x**6 - 48 * x4 * xpy + 18 * x2 * xpy2 - xpy3)
    return zf
