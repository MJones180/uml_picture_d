import numpy as np
from utils.load_optical_train import load_optical_train
from utils.model import Model
from utils.printing_and_logging import step_ri
from utils.response_matrix import ResponseMatrix
from utils.sim_prop_wf import sim_prop_wf
from utils.terminate_with_message import terminate_with_message

# In case the simulated control loop is being called multiple times,
# cache the things that need to be loaded in
loaded_optical_trains = {}
loaded_neural_networks = {}
loaded_response_matrices = {}


def iterate_simulated_control_loop(
    control_loop_steps,
    zernike_terms,
    K_p,
    K_i,
    K_d,
    train_name,
    ref_wl,
    grid_points=1024,
    print_logs=False,
    use_nn=None,
    use_rm=None,
    early_stopping=None,
):
    """
    Iterate over a simulated control loop using either a neural network or
    response matrix.

    Parameters
    ----------
    control_loop_steps : np.array
        2D array where each row contains a control loop step. Each step provides
        the input signal and should have the following columns:
            [step idx, cumulative time, delta time, *zernike coefficients].
    zernike_terms : list[int]
        Noll Zernike terms that should be controlled.
    K_p : float
        Proportional gain value (range -1 to 0).
    K_i : float
        Integral gain value (range -1 to 0).
    K_d : float
        Derivative gain value (range -1 to 0).
    train_name : str
        Name of the optical train that should be used for simulations.
    ref_wl : float
        Reference wavelength in meters for the simulations.
    grid_points : int, optional
        Number of grid points for wavefront simulations, defaults to 1024.
    print_logs : bool, optional
        Whether to display control loop logs.
    use_nn : list[str, str], optional
        List containing the tag and epoch of the neural network to use.
        The neural network must be trained with a base field subtracted off.
        Can only use this or the `use_rm` argument.
    use_rm : str, optional
        Tag of the response matrix to use.
        Can only use this or the `use_nn` argument.
    early_stopping : float, optional
        Iterations can stop early if all Zernike coefficients fall within the
        range of [-threshold, threshold]. The value passed in defines the
        threshold. The default value of `None` means early stopping is disabled
        and all control loop steps will be performed.

    Returns
    -------
    [float, np.array, np.array]
        The total cumulative time iterated over, the true error
        (input signal + corrections) history, and the measured
        error (model outputs) history.
    """

    # ======================
    # Verify the gain values
    # ======================

    def _verify_gain(name, var):
        if print_logs:
            print(f'{name}: {var}')
        if not (-1 <= var <= 0):
            terminate_with_message(f'{name} must be between -1 and 0')

    if print_logs:
        step_ri('Verifying gain values')
    _verify_gain('K_p', K_p)
    _verify_gain('K_i', K_i)
    _verify_gain('K_d', K_d)

    # ======================
    # Grab the optical train
    # ======================

    if train_name not in loaded_optical_trains:
        if print_logs:
            step_ri('Loading in the optical train')
        loaded_optical_trains[train_name] = load_optical_train(train_name)
    (init_beam_d, beam_ratio, optical_train, camera_pixels,
     camera_sampling) = loaded_optical_trains[train_name]

    # ==============
    # Grab the model
    # ==============

    if use_nn:
        nn_key = (use_nn[0], use_nn[1])
        if nn_key not in loaded_neural_networks:
            if print_logs:
                step_ri('Loading in the neural network')
            loaded_neural_networks[nn_key] = Model(*use_nn)
        model = loaded_neural_networks[nn_key]

        def call_model(inputs):
            # Subtract off the base field so that we have the delta field
            # intensity and then normalize the data
            preprocessed_inputs = model.norm_data(
                model.subtract_basefield(inputs))
            # We need to add an extra dimension to represents the batch size
            preprocessed_inputs = preprocessed_inputs[None, :, :, :]
            # Since we are only passing in one row, we only need to grab the
            # first row from the batch size dimension
            output = model(preprocessed_inputs)[0]
            # Denormalize the data
            return model.denorm_data(output)

    elif use_rm:
        if use_rm not in loaded_response_matrices:
            if print_logs:
                step_ri('Loading in the response matrix')
            loaded_response_matrices[use_rm] = ResponseMatrix(use_rm)
        response_matrix_obj = loaded_response_matrices[use_rm]

        def call_model(inputs):
            # A 1D array should be passed in
            return response_matrix_obj(total_int_field=inputs.reshape(-1))
    else:
        terminate_with_message('Must pass either a neural network (`use_nn`) '
                               'or response matrix (`use_rm`)')

    # ====================
    # Run the control loop
    # ====================

    zerinke_count = len(zernike_terms)
    corrections = np.zeros(zerinke_count)
    # We need the cumulative sum for each Zernike term to compute integral grain
    running_zernike_sum = np.zeros(zerinke_count)
    # Keep track of the history to return at the end
    true_error_history = []
    meas_error_history = []
    # The total number of steps
    total_steps = control_loop_steps.shape[0]
    if print_logs:
        step_ri('Running the control loop')
    for step in control_loop_steps:
        row_idx, cumulative_time, delta_time, *zernike_coeffs = step
        if print_logs:
            print(f'Step: {int(row_idx + 1)}/{total_steps}')
        # Aberrations should be the sum of the signal and the correction
        aberrations = zernike_coeffs + corrections
        # Potentially stop iterations early if enabled
        if early_stopping and np.all(np.abs(aberrations) <= early_stopping):
            if print_logs:
                print('Ending iterations early due to all coefficients being '
                      f'between [-{early_stopping}, {early_stopping}]')
            break
        true_error_history.append(aberrations)
        # Simulate the camera image that represents these Zernike coeffs
        camera_image, _, _ = sim_prop_wf(
            init_beam_d,
            ref_wl,
            beam_ratio,
            optical_train,
            camera_pixels,
            camera_sampling,
            zernike_terms,
            aberrations,
            grid_points,
        )
        # This will output the model's coefficients (nn or response matrix)
        model_output = call_model(camera_image)
        meas_error_history.append(model_output)
        # The new set of corrections are in addition to the corrections from the
        # last time step, so we can just add each term as we go.
        corrections += K_p * model_output  # proportional term
        # Update the running sum to compute the integral term.
        running_zernike_sum += model_output * delta_time
        corrections += K_i * running_zernike_sum  # integral term
        # Do not calculate the time derivative if this is the first step
        if len(meas_error_history) > 1:
            dzdt = (model_output - meas_error_history[-2]) / delta_time
            corrections += K_d * dzdt  # derivative term
    if print_logs:
        print('Finished running the control loop')

    return (cumulative_time, np.array(true_error_history),
            np.array(meas_error_history))
