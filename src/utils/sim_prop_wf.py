import numpy as np
import proper
from utils.constants import (CCD_INTENSITY, CCD_SAMPLING, FULL_INTENSITY,
                             FULL_SAMPLING, ZERNIKE_COEFFS, ZERNIKE_TERMS)
from utils.downsample_data import downsample_data, resize_pixel_grid
from utils.path import get_abs_path, make_dir
from utils.printing_and_logging import step_ri
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


def multi_worker_sim_prop_many_wf(
    pool,
    core_count,
    init_beam_d,
    ref_wl,
    beam_ratio,
    optical_train,
    ccd_pixels,
    ccd_sampling,
    zernike_terms,
    aberrations,
    grid_points=1024,
    plot_path=None,
    use_only_aberration_map=False,
    disable_proper_logs=True,
    enable_logs=True,
    sim_post_cb=None,
    worker_post_cb=None,
):
    """
    Use multiple workers to propagate many wavefronts through the optical setup.

    Parameters
    ----------
    pool : pathos.multiprocessing.ProcessPool
        Pool to deploy workers from.
    core_count : int
        Number of cores being used in the pool.
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
    aberrations : np.array
        The aberrations for each simulation.
    grid_points : int, optional
        Number of grid points, defaults to 1024.
    plot_path : str, optional
        If set, then plots will be saved under the folder this points to; if
        None then no plots will be saved. Plots will be saved in subfolders
        under the opinionated name of `w_{worker_idx}_sim_{sim_idx}`.
    use_only_aberration_map : bool, optional
        If True, then use only the aberration map instead of propagating through
        the optical setup, default is False.
    disable_proper_logs : bool, optional
        If True, then proper logs are turned off, default is True.
    enable_logs : bool, optional
        If True, then progress logs will be printed out, default is True.
    sim_post_cb : func, optional
        If a function is passed, then it will be called after every simulation
        is finished with the parameters of:
            (worker_idx, sim_idx, simulation_data).
    worker_post_cb : func, optional
        If a function is passed, then it will be called after a worker finishes
        all simulations with the parameters of:
            (worker_idx, simulation_data).

    Returns
    -------
    A dictionary with all the data.
    """

    # Data will be simulated and written out by this function (will be called
    # independently by each worker)
    def worker_sim_and_write(worker_idx, aberrations_chunk):
        sim_count = aberrations_chunk.shape[0]
        worker_str = f'Worker [{worker_idx}]'
        if sim_count == 0:
            if enable_logs:
                print(f'{worker_str} not assigned any simulations')
            return
        if enable_logs:
            print(f'{worker_str} assigned {sim_count} simulation(s)')
        # The data that will be written out
        simulation_data = {
            # Noll zernike term indices that are being used for aberrations
            ZERNIKE_TERMS: zernike_terms,
            # The rms error in meters associated with each of the zernike terms
            ZERNIKE_COEFFS: aberrations_chunk,
            CCD_INTENSITY: [],
            CCD_SAMPLING: ccd_sampling,
            FULL_INTENSITY: [],
            FULL_SAMPLING: [],
        }
        for sim_idx in range(sim_count):
            if enable_logs:
                print(f'[{worker_idx}] Simulation, {sim_idx + 1}/{sim_count}')
            plot_path_complete = None
            if plot_path is not None:
                plot_path_complete = f'{plot_path}/w_{worker_idx}_sim_{sim_idx}'
            ccd_wf, full_wf, full_sampling = sim_prop_wf(
                init_beam_d,
                ref_wl,
                beam_ratio,
                optical_train,
                ccd_pixels,
                ccd_sampling,
                zernike_terms,
                aberrations_chunk[sim_idx],
                grid_points=grid_points,
                plot_path=plot_path_complete,
                use_only_aberration_map=use_only_aberration_map,
                disable_proper_logs=disable_proper_logs,
            )
            simulation_data[CCD_INTENSITY].append(ccd_wf)
            simulation_data[FULL_INTENSITY].append(full_wf)
            simulation_data[FULL_SAMPLING].append(full_sampling)
            if sim_post_cb is not None:
                sim_post_cb(worker_idx, sim_idx, simulation_data)
        if worker_post_cb is not None:
            worker_post_cb(worker_idx, simulation_data)
        return simulation_data

    if enable_logs:
        step_ri('Creating the chunks for the workers')
        nrows = aberrations.shape[0]
        print(f'Splitting {nrows} simulations across {core_count} core(s)')
    # Allow identification of individual workers
    worker_indexes = np.arange(core_count)
    # Split the rows into chunks to pass to each worker
    aberrations_chunks = np.array_split(aberrations, core_count)

    if enable_logs:
        step_ri('Beginning to run simulations')
    # Since each worker writes out its own data, no need to aggregate at the end
    results = pool.map(worker_sim_and_write, worker_indexes,
                       aberrations_chunks)
    merged_results = results[0]
    # Merge together all the worker results
    for result_dict in results[1:]:
        if result_dict is None:
            continue
        keys = [ZERNIKE_COEFFS, CCD_INTENSITY, FULL_INTENSITY, FULL_SAMPLING]
        for key in keys:
            merged_results[key] = np.concatenate(
                (merged_results[key], result_dict[key]))
    return merged_results
