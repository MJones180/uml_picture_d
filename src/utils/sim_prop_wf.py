import numpy as np
import proper
from utils.downsample_data import downsample_data, resize_pixel_grid
from utils.path import get_abs_path, make_dir
from utils.plots.plot_intensity_field import plot_intensity_field


def sim_prop_wf(
    init_beam_d,
    ref_wl,
    beam_ratio,
    optical_train,
    ccd_pixels,
    ccd_sampling,
    zernike_terms,
    aberration_values,
    grid_points=1024,
    plot_path=None,
    use_only_aberration_map=False,
    disable_proper_logs=True,
):
    """
    Propagate a wavefront through an optical setup.

    Parameters
    ----------
    init_beam_d : float
        Diameter of the beam.
    ref_wl : float
        Reference wavelength in meters.
    beam_ratio : float
        Ratio of the beam to the grid.
    optical_train : list
        The optical train that the wavefront will pass through.
    ccd_pixels : int
        Number of pixels on the resampled CCD.
    ccd_sampling : float
        Sampling of the resampled CCD.
    zernike_terms : list[int]
        Noll Zernike terms that will have aberrations.
    aberration_values : list[float]
        Aberration amount in meters to add to each Zernike term.
    grid_points : int, optional
        Number of grid points, defaults to 1024.
    plot_path : str, optional
        If set, then plots will be saved under the folder this points to; if
        None then no plots will be saved.
    use_only_aberration_map : bool, optional
        If True, then use only the aberration map instead of propagating through
        the optical setup, default is False.
    disable_proper_logs : bool, optional
        If True, then proper logs are turned off, default is True.

    Returns
    -------
    [np.array, np.array, float] where the values consist of the resampled
    CCD intensity wavefront, full intensity wavefront, and full sampling.
    """

    if disable_proper_logs:
        # Ignore all proper logs
        proper.print_it = False

    plot_idx = 0

    def _plot_intensity(wf_or_intensity, title):
        if plot_path is None:
            return
        nonlocal plot_idx
        linear_path = f'{plot_path}/linear'
        log_path = f'{plot_path}/log'
        # Needs to be done for each simulation
        if plot_idx == 0:
            make_dir(linear_path)
            make_dir(log_path)
        # If it is a NP array, then it is the final intensity on the CCD,
        # otherwise it is a PROPER wavefront object
        if isinstance(wf_or_intensity, np.ndarray):
            intensity = wf_or_intensity
            plot_sampling = ccd_sampling
        else:
            intensity = proper.prop_get_amplitude(wf_or_intensity)**2
            plot_sampling = proper.prop_get_sampling(wf_or_intensity)

        def _get_plot_path(sub_dir):
            return get_abs_path(f'{sub_dir}/step_{plot_idx}.png')

        plot_intensity_field(intensity, plot_sampling, title,
                             _get_plot_path(linear_path))
        plot_intensity_field(intensity, plot_sampling, title,
                             _get_plot_path(log_path), True)
        plot_idx += 1

    wavefront = proper.prop_begin(init_beam_d, ref_wl, grid_points, beam_ratio)
    # Define the initial aperture
    proper.prop_circular_aperture(wavefront, init_beam_d / 2)
    # Set this as the entrance to the train
    proper.prop_define_entrance(wavefront)
    # Add in the aberrations to the wavefront
    aberration_map = proper.prop_zernikes(wavefront, zernike_terms,
                                          aberration_values)
    _plot_intensity(wavefront, 'Entrance')
    if use_only_aberration_map:
        # Pixels where the circle is
        aperture_mask = proper.prop_get_amplitude(wavefront)**2 > 0
        # If only using the aberration map, then no need to propagate
        # the wavefront through the optical setup.
        wavefront_intensity = aberration_map * aperture_mask
        _plot_intensity(wavefront_intensity, 'Aberration Map')
        # For the CCD, we will just resize the grid so that the number
        # of pixels matches. No need to do any resampling.
        wf_int_ds = resize_pixel_grid(wavefront_intensity, ccd_pixels)
        sampling = ccd_sampling
    else:
        # Loop through the train
        for step in optical_train:
            # Nested list means step is eligible for plotting
            if type(step) is list:
                step[1](wavefront)
                _plot_intensity(wavefront, step[0])
            else:
                step(wavefront)
        # The final wavefront intensity and sampling of its grid
        (wavefront_intensity, sampling) = proper.prop_end(wavefront)
        # Downsample to the CCD
        wf_int_ds = downsample_data(wavefront_intensity, sampling,
                                    ccd_sampling, ccd_pixels)
    # Plot the downsampled CCD intensity
    _plot_intensity(wf_int_ds, 'CCD Resampled')
    # Returns CCD intensity wf, full intensity wf, and full sampling.
    return [wf_int_ds, wavefront_intensity, sampling]
