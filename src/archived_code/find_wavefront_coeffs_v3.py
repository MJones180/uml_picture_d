"""
This file is copied from `find_wavefront_coeffs_v2.py`. The difference is
instead of fitting based off of the propagated basis vectors, this code properly
does a forward model fit. That means, each time a coefficient is adjusted, the
Zernike wavefront is propagated through the optical setup.

The code to simulate data in this script was taken from `sim_data.py`. If that
code becomes modularized, then the simulation code in this script needs to be
rewritten.

Additionally, most of the code between the three versions of this file are
exactly the same. If any of these become non-archived scripts, then the shared
parts should be modularized and the plots should be put into separate files.

Commands to run this script:
    python3 main.py find_wavefront_coeffs_v3 \
        no_aberrations all_10nm -100 100 --save-plots --cores 4
"""

import matplotlib.pyplot as plt
import numpy as np
from pathos.multiprocessing import ProcessPool
import proper
from time import time
from utils.constants import ARGS_F, RANDOM_P, RAW_SIMULATED_DATA_P
from utils.downsample_data import downsample_data
from utils.json import json_load
from utils.load_optical_train import load_optical_train
from utils.load_raw_sim_data_chunks import load_raw_sim_data_chunks
from scipy.optimize import minimize
from utils.printing_and_logging import step_ri, title
from utils.proper_use_fftw import proper_use_fftw
from utils.stats_and_error import sum_of_abs


def find_wavefront_coeffs_v3_parser(subparsers):
    subparser = subparsers.add_parser(
        'find_wavefront_coeffs_v3',
        help='test the coefficients of a wavefront',
    )
    subparser.set_defaults(main=find_wavefront_coeffs_v3)
    subparser.add_argument(
        'base_field_ds',
        help=('name of the dataset containing the base field, this should be '
              'a raw dataset and there should only be one row containing no '
              'aberrations (the `no_aberrations` simulation arg), this '
              'dataset will be used to load in the simulation parameters'),
    )
    subparser.add_argument(
        'test_ds',
        help=('name of the dataset to find the wavefront coefficients for, '
              'this should be a raw dataset'),
    )
    subparser.add_argument(
        'init_coeff_bounds',
        type=int,
        nargs=2,
        help='initial bounds to pick first guess between',
    )
    subparser.add_argument(
        '--row-idx',
        type=int,
        default=0,
        help='the row to find the wavefront coefficients for',
    )
    subparser.add_argument(
        '--use-full-field',
        action='store_true',
        help='use the full field instead of the CCD field',
    )
    subparser.add_argument(
        '--save-plots',
        action='store_true',
        help='save plots instead of displaying them',
    )
    subparser.add_argument(
        '--cores',
        default=1,
        type=int,
        help=('number of cores to split the simulations between, more cores '
              'means faster but more memory consumption'),
    )


def find_wavefront_coeffs_v3(cli_args):
    title('Find wavefront coeffs v3 script')

    step_ri('Loading in CLI args')
    base_field_ds = cli_args['base_field_ds']
    test_ds = cli_args['test_ds']
    init_coeff_bounds = cli_args['init_coeff_bounds']
    row_idx = cli_args['row_idx']
    use_full_field = cli_args['use_full_field']
    save_plots = cli_args['save_plots']
    cores = cli_args['cores']

    step_ri('Creating the process pool')
    cores = cli_args['cores']
    print(f'Using {cores} core(s)')
    pool = ProcessPool(ncpus=cores)

    step_ri('Ensuring FFTW is being used')
    proper_use_fftw()

    step_ri('Loading in the base field dataset')
    base_field = load_raw_sim_data_chunks(base_field_ds, use_full_field)[0]

    step_ri('Loading in simulation args')
    # Loading in the args used to simulate data from the base field
    bf_cli_args = json_load(f'{RAW_SIMULATED_DATA_P}/{base_field_ds}/{ARGS_F}')
    train_name = bf_cli_args['train_name']
    ref_wl = float(bf_cli_args['ref_wl'])
    grid_points = int(bf_cli_args['grid_points'])

    step_ri('Loading in the optical train')
    (init_beam_d, beam_ratio, optical_train, ccd_pixels,
     ccd_sampling) = load_optical_train(train_name)

    step_ri('Loading in the test dataset')
    test_ds_data = load_raw_sim_data_chunks(test_ds, use_full_field)
    zernike_terms = test_ds_data[2]
    zernike_count = len(zernike_terms)
    target_wavefront = test_ds_data[0][row_idx] - base_field
    truth_zernike_coeffs = test_ds_data[1][row_idx]
    print('TRUTH: ', truth_zernike_coeffs)

    step_ri('Computing coefficients')
    # Number of times a forward model call is done
    call_idx = 0

    def forward_model_prop(coeffs_vectors):
        start_time = time()
        nonlocal call_idx
        call_idx += 1
        print(f'[{call_idx}] Forward model call ({len(coeffs_vectors)} rows)')

        # This function is called by each worker
        def worker_sim(aberrations_chunk):
            sim_count = aberrations_chunk.shape[0]
            if sim_count == 0:
                return
            # Ignore all proper logs
            proper.print_it = False
            ccd_intensity = []
            full_intensity = []
            for sim_idx in range(sim_count):
                # Create the wavefront that will be passed through the train
                wavefront = proper.prop_begin(init_beam_d, ref_wl, grid_points,
                                              beam_ratio)
                # Define the initial aperture
                proper.prop_circular_aperture(wavefront, init_beam_d / 2)
                # Set this as the entrance to the train
                proper.prop_define_entrance(wavefront)
                # Add in the aberrations to the wavefront
                proper.prop_zernikes(wavefront, zernike_terms,
                                     aberrations_chunk[sim_idx])
                # Loop through the train
                for step in optical_train:
                    step_func = step if type(step) is not list else step[1]
                    step_func(wavefront)
                # The final wavefront intensity and sampling of its grid
                (wavefront_intensity, sampling) = proper.prop_end(wavefront)
                # Downsample to the CCD
                wf_int_ds = downsample_data(wavefront_intensity, sampling,
                                            ccd_sampling, ccd_pixels)
                # Add the data to the output arrays
                ccd_intensity.append(wf_int_ds)
                full_intensity.append(wavefront_intensity)
            return [ccd_intensity, full_intensity]

        # The coeffs should be in nm, this will be our list of aberrations
        coeffs_vectors_nm = coeffs_vectors * 1e-9
        aberrations_chunks = np.array_split(coeffs_vectors_nm, cores)
        # Each worker returns both the ccd and full intensity
        sim_results = pool.map(worker_sim, aberrations_chunks)
        # Need to choose between the full field and the ccd field
        field_type = 1 if use_full_field else 0
        # Need to combine the results back together from each worker
        fields = sim_results[0][field_type]
        for worker_sim_results in sim_results[1:]:
            if worker_sim_results is not None:
                fields = np.vstack((fields, worker_sim_results[field_type]))
        print(f'\tCall time: {time() - start_time}')
        return fields - base_field

    def minimize_func(coeffs):
        # The finite difference amount
        STEP_SIZE = 0.01
        # To calculate the gradient, we need to perturb each coeff by itself
        coeff_vectors = np.repeat(coeffs[None], zernike_count, axis=0)
        coeff_vectors[np.diag_indices_from(coeff_vectors)] += STEP_SIZE
        # To minimize forward model calls, also add on the unperturbed coeffs
        coeff_vectors = np.vstack((coeff_vectors, coeffs))
        fields = forward_model_prop(coeff_vectors)
        # The last field is what the current set of coeffs represent
        grad_wfs = fields[:-1]
        current_wf = fields[-1]
        # The wavefront error can be calculated as the sum of the absolute
        # differences between the target wavefront and current wavefront
        wavefront_error = sum_of_abs(target_wavefront - current_wf)
        # Calculate the error for each perturbed coefficient
        grad_wfe = sum_of_abs(target_wavefront - grad_wfs, (1, 2))
        # Calculate the gradient
        grad = (grad_wfe - wavefront_error) / STEP_SIZE
        return wavefront_error, grad

    def post_iteration_cb(intermediate_result):
        print(f'\tCost function: {intermediate_result.fun}')

    minimization = minimize(
        minimize_func,
        # Initial coefficient guesses
        np.random.uniform(*init_coeff_bounds, zernike_count),
        jac=True,
        tol=1e-8,
        method='L-BFGS-B',
        callback=post_iteration_cb,
    )
    print(minimization)
    coeffs = minimization.x
    print(coeffs)

    step_ri('Plotting a bar plot of wavefront coefficients')
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.8
    indices = np.arange(zernike_count)
    plt.plot(indices, np.zeros(zernike_count), color='black', linewidth=1)
    plt.bar(
        indices,
        coeffs * 1e-9,
        width=bar_width,
        color='blue',
        alpha=0.5,
        label='Obtained',
    )
    plt.bar(
        indices,
        truth_zernike_coeffs,
        width=bar_width,
        linewidth=2,
        edgecolor='black',
        color='none',
        label='Truth',
    )
    plt.title('Basis Terms')
    plt.ylabel('RMS Error (nm)')
    plt.xlabel('Zernike Terms')
    plt.xticks(indices, [term for term in zernike_terms])
    plt.legend()
    if save_plots:
        fig.tight_layout()
        base_path = f'{RANDOM_P}/{base_field_ds}_{test_ds}_row_{row_idx}'
        path = f'{base_path}_bar_graph.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    reconstructed_wf = forward_model_prop(np.array([coeffs]))
    ax[0].imshow(target_wavefront[0])
    ax[0].set_title('Original Wavefront')
    ax[1].imshow(reconstructed_wf[0])
    ax[1].set_title('Reconstructed Wavefront')
    if save_plots:
        fig.tight_layout()
        path = f'{base_path}_wavefront.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    # Close the pool to any new jobs and remove it
    pool.close()
    pool.clear()
