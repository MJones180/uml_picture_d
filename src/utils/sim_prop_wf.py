import numpy as np
import proper
from utils.add_wf_aberrations import add_wf_aberrations
from utils.constants import (CAMERA_INTENSITY, CAMERA_SAMPLING, FULL_INTENSITY,
                             FULL_SAMPLING, PLOTTING_LINEAR_INT,
                             PLOTTING_LINEAR_PHASE,
                             PLOTTING_LINEAR_PHASE_NON0_INT, PLOTTING_LOG_INT,
                             PLOTTING_PATH, ZERNIKE_COEFFS, ZERNIKE_TERMS)
from utils.downsample_data import downsample_data, resize_pixel_grid
from utils.path import get_abs_path, make_dir
from utils.printing_and_logging import step_ri
from utils.plots.plot_wavefront import plot_wavefront


def sim_prop_wf(
    init_beam_d,
    ref_wl,
    beam_ratio,
    optical_train,
    camera_pixels,
    camera_sampling,
    zernike_terms,
    aberration_values,
    extra_params={},
    grid_points=1024,
    plotting={},
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
    camera_pixels : int
        Number of pixels on the resampled camera.
    camera_sampling : float
        Sampling of the resampled camera.
    zernike_terms : list[int]
        Noll Zernike terms that will have aberrations.
        These aberrations will be applied at the entrance.
    aberration_values : list[float]
        Aberration amount in meters to add to each Zernike term.
    extra_params : dict, optional
        Some optical train steps may require additional parameters. This
        dictionary will be passed to any optical train steps that accept
        two arguments.
    grid_points : int, optional
        Number of grid points, defaults to 1024.
    plotting: dict, optional
        If the 'path' key is set, then plots will be saved in a folder at that
        path. The following bool keys will determine which plots are created:
        `linear_intensity`, `log_intensity`, `linear_phase`, and
        `linear_phase_nonzero_intensity` (phase where intensity is nonzero).
    use_only_aberration_map : bool, optional
        If True, then use only the aberration map instead of propagating through
        the optical setup, default is False.
    disable_proper_logs : bool, optional
        If True, then proper logs are turned off, default is True.

    Returns
    -------
    [np.array, np.array, float] where the values consist of the resampled
    camera intensity wavefront, full intensity wavefront, and full sampling.
    """

    if disable_proper_logs:
        # Ignore all proper logs
        proper.print_it = False

    plot_idx = 0

    def _plot(title, wf_obj=None, intensity_arr=None):
        plot_path = plotting.get(PLOTTING_PATH)
        if plot_path is None:
            return
        nonlocal plot_idx

        def _grab_path(key):
            file_path = None
            if plotting.get(key) is True:
                folder_path = f'{plot_path}/{key}'
                # Create the folder if it does not yet exist
                if plot_idx == 0:
                    make_dir(folder_path)
                file_path = get_abs_path(f'{folder_path}/step_{plot_idx}.png')
            return file_path

        # Grab the folders for each potential plot
        linear_int_path = _grab_path(PLOTTING_LINEAR_INT)
        linear_phase_path = _grab_path(PLOTTING_LINEAR_PHASE)
        linear_phase_non0_int_path = _grab_path(PLOTTING_LINEAR_PHASE_NON0_INT)
        log_int_path = _grab_path(PLOTTING_LOG_INT)
        # Grab the data based on whether the whole wavefront object was passed
        # or just the intensity
        if wf_obj is not None:
            plot_sampling = proper.prop_get_sampling(wf_obj)
            intensity_data = proper.prop_get_amplitude(wf_obj)**2
            # Phase is grabbed in radians
            phase_data = proper.prop_get_phase(wf_obj)
            # Put the phase into nm
            phase_data *= ref_wl / (2 * np.pi) * 1e9
        else:
            plot_sampling = camera_sampling
            intensity_data = intensity_arr
            phase_data = None

        # Wrapper around the plotting function
        def _plot_wf(data, cbar_title, path, log=False):
            plot_wavefront(data, cbar_title, plot_sampling, title, path, log)

        # Do the plotting now
        if linear_int_path is not None:
            _plot_wf(intensity_data, 'Intensity', linear_int_path)
        if log_int_path is not None:
            _plot_wf(intensity_data, 'log(Intensity)', log_int_path, True)
        # The phase data may not be passed, so plotting is not always possible
        if phase_data is not None:
            if linear_phase_path is not None:
                _plot_wf(phase_data, 'Phase [nm]', linear_phase_path)
            # Only plot the phase where the intensity is nonzero
            if linear_phase_non0_int_path is not None:
                phase_data[intensity_data == 0] = 0
                _plot_wf(phase_data, 'Phase [nm]', linear_phase_non0_int_path)
        plot_idx += 1

    wavefront = proper.prop_begin(init_beam_d, ref_wl, grid_points, beam_ratio)
    # Define the initial aperture
    proper.prop_circular_aperture(wavefront, init_beam_d / 2)
    # Set this as the entrance to the train
    proper.prop_define_entrance(wavefront)
    # Add in the aberrations to the wavefront
    aberration_map = add_wf_aberrations(wavefront, zernike_terms,
                                        aberration_values)
    _plot('Entrance', wf_obj=wavefront)
    if use_only_aberration_map:
        # Pixels where the circle is
        aperture_mask = proper.prop_get_amplitude(wavefront)**2 > 0
        # If only using the aberration map, then no need to propagate
        # the wavefront through the optical setup.
        wavefront_intensity = aberration_map * aperture_mask
        _plot('Aberration Map', intensity_arr=wavefront_intensity)
        # For the camera, we will just resize the grid so that the number
        # of pixels matches. No need to do any resampling.
        wf_int_ds = resize_pixel_grid(wavefront_intensity, camera_pixels)
        sampling = camera_sampling
    else:
        # Loop through the train
        for step in optical_train:
            # Nested list means step is eligible for plotting
            plot_title, func = step if type(step) is list else (None, step)
            # https://stackoverflow.com/a/10865355
            if func.__code__.co_argcount == 2:
                func(wavefront, extra_params)
            else:
                func(wavefront)
            if plot_title is not None:
                _plot(plot_title, wf_obj=wavefront)
        # The final wavefront intensity and sampling of its grid
        (wavefront_intensity, sampling) = proper.prop_end(wavefront)
        # Downsample to the camera
        wf_int_ds = downsample_data(wavefront_intensity, sampling,
                                    camera_sampling, camera_pixels)
    # Plot the downsampled camera wavefront
    _plot('Camera Resampled', intensity_arr=wf_int_ds)
    # Returns camera intensity wf, full intensity wf, and full sampling.
    return [wf_int_ds, wavefront_intensity, sampling]


def multi_worker_sim_prop_many_wf(
    pool,
    core_count,
    init_beam_d,
    ref_wl,
    beam_ratio,
    optical_train,
    camera_pixels,
    camera_sampling,
    zernike_terms,
    aberrations,
    extra_params={},
    save_full_intensity=False,
    grid_points=1024,
    plotting={},
    use_only_aberration_map=False,
    disable_proper_logs=True,
    enable_logs=True,
    sim_post_cb=None,
    worker_post_cb=None,
    do_not_return_data=False,
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
    camera_pixels : int
        Number of pixels on the resampled camera.
    camera_sampling : float
        Sampling of the resampled camera.
    zernike_terms : list[int]
        Noll Zernike terms that will have aberrations.
    aberrations : np.array
        The aberrations for each simulation.
    extra_params : dict, optional
        Some optical train steps may require additional parameters. This
        dictionary will be passed to any optical train steps that accept
        two arguments.
    save_full_intensity, bool, optional
        If True, will save the full intensity field and return it, but this will
        take up much more memory, default is False.
    grid_points : int, optional
        Number of grid points, defaults to 1024.
    plotting: dict, optional
        If the 'path' key is set, then plots will be saved in subfolders
        under the opinionated name of `w_{worker_idx}_sim_{sim_idx}`.
        The following bool keys will determine which plots are created:
        `linear_intensity`, `log_intensity`, `linear_phase`, and
        `linear_phase_nonzero_intensity` (phase where intensity is nonzero).
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
    do_not_return_data : bool, optional
        If True, then the data will not be returned and it should be instead
        written out during the callback functions. The default is False. The
        reason for this parameter is sometimes the `pool.map` can hang if too
        much data is being passed.

    Returns
    -------
    A dictionary with all the data.
    """

    # Data will be simulated by this function (will be called independently by
    # each worker)
    def worker_prop_wfs(worker_idx, aberrations_chunk, extra_params_chunk):
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
            CAMERA_INTENSITY: [],
            CAMERA_SAMPLING: camera_sampling,
        }
        if save_full_intensity:
            simulation_data[FULL_INTENSITY] = []
            simulation_data[FULL_SAMPLING] = []
        plotting_dict = plotting.copy()
        plot_path = plotting_dict.pop(PLOTTING_PATH, None)
        for sim_idx in range(sim_count):
            if enable_logs:
                print(f'[{worker_idx}] Simulation, {sim_idx + 1}/{sim_count}')
            if plot_path is not None:
                path = f'{plot_path}/w_{worker_idx}_sim_{sim_idx}'
                plotting_dict[PLOTTING_PATH] = path
            # Grab the extra params that correspond to the current simulation
            extra_params_local = {
                key: array[sim_idx]
                for key, array in extra_params_chunk.items()
            }
            camera_wf, full_wf, full_sampling = sim_prop_wf(
                init_beam_d,
                ref_wl,
                beam_ratio,
                optical_train,
                camera_pixels,
                camera_sampling,
                zernike_terms,
                aberrations_chunk[sim_idx],
                extra_params=extra_params_local,
                grid_points=grid_points,
                plotting=plotting_dict,
                use_only_aberration_map=use_only_aberration_map,
                disable_proper_logs=disable_proper_logs,
            )
            simulation_data[CAMERA_INTENSITY].append(camera_wf)
            if save_full_intensity:
                simulation_data[FULL_INTENSITY].append(full_wf)
                simulation_data[FULL_SAMPLING].append(full_sampling)
            if sim_post_cb is not None:
                sim_post_cb(worker_idx, sim_idx, simulation_data)
        if worker_post_cb is not None:
            worker_post_cb(worker_idx, simulation_data)
        if do_not_return_data:
            return None
        return simulation_data

    if enable_logs:
        step_ri('Creating the chunks for the workers')
        nrows = aberrations.shape[0]
        print(f'Splitting {nrows} simulations across {core_count} core(s)')
    # Allow identification of individual workers
    worker_indexes = np.arange(core_count)
    # Split the rows into chunks to pass to each worker
    aberrations_chunks = np.array_split(aberrations, core_count)
    # Since this is a dictionary, the inner arrays must first be split into
    # chunks. Then, new dictionaries can be created with the chunked arrays.
    extra_params_split_arrays = {
        key: np.array_split(array, core_count)
        for key, array in extra_params.items()
    }
    extra_params_chunks = [{
        key: array_chunks[idx]
        for key, array_chunks in extra_params_split_arrays.items()
    } for idx in range(core_count)]
    # This dictionary is no longer needed
    del extra_params_split_arrays
    if enable_logs:
        step_ri('Beginning to run simulations')
    # There is a chance that if the data being returned is large, then this will
    # hang after finishing the map. In cases like this, the data should be
    # written out using a callback function and the `do_not_return_data`
    # argument should be passed.
    results = pool.map(
        worker_prop_wfs,
        worker_indexes,
        aberrations_chunks,
        extra_params_chunks,
    )
    # Merge together all the worker results
    merged_results = results[0]
    for result_dict in results[1:]:
        if result_dict is None:
            continue
        merge_keys = [ZERNIKE_COEFFS, CAMERA_INTENSITY]
        if save_full_intensity:
            merge_keys.extend([FULL_INTENSITY, FULL_SAMPLING])
        for key in merge_keys:
            merged_results[key] = np.concatenate((
                merged_results[key],
                result_dict[key],
            ))
    return merged_results
