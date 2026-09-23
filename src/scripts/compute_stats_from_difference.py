import numpy as np
from utils.cli_args import save_cli_args
from utils.constants import (CHOLESKY_L, DATA_F, MEAN, PROC_DATA_P, RAW_DATA_P,
                             STD)
from utils.hdf_read_and_write import HDFWriteModule, read_hdf
from utils.path import make_dir
from utils.plots.plot_line import plot_line
from utils.plots.plot_2d import plot_2d
from utils.printing_and_logging import step_ri, title


def compute_stats_from_difference_parser(subparsers):
    subparser = subparsers.add_parser(
        'compute_stats_from_difference',
        help='compute the mean and std of the difference between two datasets',
    )
    subparser.set_defaults(main=compute_stats_from_difference)
    subparser.add_argument(
        'output_tag',
        help='tag to give to the stats; will be put in the raw data dir',
    )
    subparser.add_argument(
        'first_tag',
        help='tag of the first dataset; must be a processed dataset',
    )
    subparser.add_argument(
        'first_dataset_table',
        help='name of the table to use from the first dataset',
    )
    subparser.add_argument(
        'second_tag',
        help='tag of the second dataset; must be a processed dataset',
    )
    subparser.add_argument(
        'second_dataset_table',
        help='name of the table to use from the second dataset',
    )
    subparser.add_argument(
        '--use-first-n-values',
        type=int,
        help='only use the first N values',
    )
    subparser.add_argument(
        '--zero-small-means',
        type=int,
        help=('mean values set to zero when the following condition is met: '
              'SEM (Standard Error of the Mean) >= |mean| / N; N is passed'),
    )
    subparser.add_argument(
        '--calculate-cov',
        action='store_true',
        help='calculate the covariance and Cholesky factor',
    )
    subparser.add_argument(
        '--calculate-cov-mode-limit',
        type=int,
        help='zero out all covariance after a given mode',
    )


def compute_stats_from_difference(cli_args):
    title('Compute stats from difference script')

    step_ri('Creating output directory')
    output_tag = cli_args['output_tag']
    out_dir = f'{RAW_DATA_P}/{output_tag}'
    print(f'Creating {out_dir}')
    make_dir(out_dir)

    step_ri('Saving CLI args')
    save_cli_args(out_dir, cli_args, 'compute_stats_from_difference')

    step_ri('Loading data')

    def load_data(identifier, cli_arg_tag, cli_arg_table_name):
        tag = cli_args[cli_arg_tag]
        table_name = cli_args[cli_arg_table_name]
        print(f'{identifier} tag: {tag}')
        print(f'{identifier} table: {table_name}')
        datafile = read_hdf(f'{PROC_DATA_P}/{tag}/{DATA_F}')
        data = datafile[table_name][:]
        print(f'Data shape: {data.shape}')
        return data

    first_ds = load_data('First', 'first_tag', 'first_dataset_table')
    second_ds = load_data('Second', 'second_tag', 'second_dataset_table')

    use_first_n_values = cli_args.get('use_first_n_values')
    if use_first_n_values is not None:
        step_ri(f'Using first {use_first_n_values} values')
        first_ds = first_ds[:, :use_first_n_values]
        second_ds = second_ds[:, :use_first_n_values]
        print(f'New shape: {first_ds.shape}')

    step_ri('Computing difference')
    diff = first_ds - second_ds
    print(f'Diff shape: {diff.shape}')

    step_ri('Computing mean and std')
    mean = np.mean(diff, axis=0)
    std = np.std(diff, axis=0)
    print(f'Mean shape: {mean.shape}')
    print(f'STD shape: {std.shape}')
    out_data = {MEAN: mean, STD: std}

    step_ri('Saving plots')
    plot_line(mean, 'Difference Mean', 'Index', 'Mean', f'{out_dir}/mean.png')
    plot_line(std, 'Difference STD', 'Index', 'STD', f'{out_dir}/std.png')

    zero_small_means = cli_args.get('zero_small_means')
    if zero_small_means is not None:
        step_ri('Zeroing small values using SEM')
        print(f'N: {zero_small_means}')
        # Standard Error of the Mean
        sem = std / np.sqrt(len(diff))
        # Zero out the required mean values
        mean[sem >= np.abs(mean) / zero_small_means] = 0
        plot_line(mean, f'Difference Mean (Zero Means, N={zero_small_means})',
                  'Index', 'Mean', f'{out_dir}/mean_sem.png')

    if cli_args['calculate_cov']:
        step_ri('Calculating the covariance')
        # Perform the covariance calculation in float32 for precision
        cov = np.cov(diff.astype(np.float64), rowvar=False)
        print(f'Covariance shape: {cov.shape}')
        # Potentially limit the number of correlated modes; modes that end up
        # being uncorrelated will still have their independent STDs
        mode_limit = cli_args.get('calculate_cov_mode_limit')
        if mode_limit is not None:
            print(f'Zeroing out all modes after index {mode_limit}')
            cov_mask = np.zeros_like(cov, dtype=bool)
            cov_mask[mode_limit:, :] = True
            cov_mask[:, mode_limit:] = True
            np.fill_diagonal(cov_mask, False)
            cov[cov_mask] = 0
        # Calculate the Cholesky factor, L @ L.T = cov
        L_matrix = np.linalg.cholesky(cov).astype(np.float32)
        print(f'L (Cholesky factor) shape: {L_matrix.shape}')
        out_data[CHOLESKY_L] = L_matrix
        # Plot out the eigenvalues of the covariance matrix
        plot_line(np.linalg.eigvalsh(cov), 'Covariance Matrix Eigenvalues',
                  'Index', 'Eigenvalue', f'{out_dir}/cov_eigvals.png')
        # Plot out the correlation
        diag_std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(diag_std, diag_std)
        plot_2d(corr, 'Correlation', 'Index', 'Index',
                f'{out_dir}/correlation.png')
        # Plot out the absolute Cholesky factor
        plot_2d(np.abs(L_matrix), 'Magnitude of L (Cholesky Factor)',
                'Input Index', 'Output Index',
                f'{out_dir}/cholesky_factor.png')

    step_ri('Writing out mean and std')
    datafile_path = f'{out_dir}/0_{DATA_F}'
    print(f'Path: {datafile_path}')
    HDFWriteModule(datafile_path).create_and_write_hdf_simple(out_data)
