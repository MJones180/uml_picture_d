"""
Analyze modes from a basis set (for instance, SVD modes).
"""

import numpy as np
from utils.constants import MODE_ANALYSIS_P
from utils.create_grid_mask import create_grid_mask
from utils.hdf_read_and_write import read_hdf
from utils.load_raw_sim_data import raw_sim_data_chunk_paths
from utils.path import make_dir
from utils.plots.plot_wavefront import plot_wavefront
from utils.printing_and_logging import dec_print_indent, step, step_ri, title
from utils.stats_and_error import mse


def analyze_basis_modes_parser(subparsers):
    subparser = subparsers.add_parser(
        'analyze_basis_modes',
        help='analyze the different modes in a basis set',
    )
    subparser.set_defaults(main=analyze_basis_modes)
    subparser.add_argument(
        'modes_tag',
        help=('tag of the raw dataset; the data must be a single 2D table '
              'with modes along each row'),
    )
    subparser.add_argument(
        'modes_table_name',
        help='name of the modes table in the data',
    )
    subparser.add_argument(
        '--transpose-modes',
        action='store_true',
        help='transpose the modes data',
    )
    subparser.add_argument(
        '--modes-are-complex',
        type=int,
        help=('the modes are complex with real components stacked on top of '
              'imaginary components; one argument expected: axis to split'),
    )
    subparser.add_argument(
        '--display-as-circle',
        nargs=2,
        type=float,
        help=('display the modes on a circle; two arguments expected: '
              'grid size and relative circle scaling size'),
    )
    subparser.add_argument(
        '--display-with-hole',
        type=float,
        help=('used in conjunction with the `--display-as-circle` arg; '
              'display the circle with a hole; one argument expected: '
              'relative circle scaling size for the hole'),
    )
    subparser.add_argument(
        '--plot-modes-specific',
        nargs='+',
        type=int,
        help='plot individual modes at the given indices',
    )
    subparser.add_argument(
        '--plot-modes-range',
        nargs=2,
        type=int,
        help=('plot individual modes between the provided range; '
              'two arguments expected: low mode idx, high mode idx '),
    )
    subparser.add_argument(
        '--reconstruct-data',
        nargs=6,
        help=('reconstruct data in terms of the basis being analyzed; six '
              'arguments are expected: raw datafile tag, datafile table name, '
              'row index to reconstruct, number of modes to use, whether the '
              'circle mask needs to be applied, use the real (0) or imag (1) '
              'component if the data is complex'),
    )
    subparser.add_argument(
        '--reconstruct-data-trim',
        nargs=4,
        type=int,
        help=('used in conjunction with the `--reconstruct-data` arg; '
              'trim the data, four argument expected: x0, x1, y0, y1'),
    )
    subparser.add_argument(
        '--reconstruct-data-plots',
        action='store_true',
        help=('used in conjunction with the `--reconstruct-data` arg; '
              'plot the row and the reconstruction'),
    )


def analyze_basis_modes(cli_args):
    title('Analyze basis modes script')

    step_ri('Loading data')
    modes_tag = cli_args['modes_tag']
    print(f'Tag: {modes_tag}')
    modes_path = raw_sim_data_chunk_paths(modes_tag)[0]
    print(f'Modes path: {modes_path}')
    modes_table_name = cli_args['modes_table_name']
    print(f'Table name: {modes_table_name}')
    modes_data = read_hdf(modes_path)[modes_table_name][:]
    print(f'Modes shape: {modes_data.shape}')

    if cli_args.get('transpose_modes'):
        step_ri('Transposing the modes')
        modes_data = modes_data.T
        print(f'Modes shape: {modes_data.shape}')

    modes_are_complex = cli_args.get('modes_are_complex')
    if modes_are_complex is not None:
        step_ri('Splitting mode data into real and imaginary components')
        modes_data_real, modes_data_imag = np.split(
            modes_data,
            2,
            axis=modes_are_complex,
        )
        print(f'Real modes shape: {modes_data_real.shape}')
        print(f'Imag modes shape: {modes_data_imag.shape}')

    step_ri('Creating output directory')
    output_dir = f'{MODE_ANALYSIS_P}/{modes_tag}'
    make_dir(output_dir)
    print(f'Path: {output_dir}')

    display_as_circle = cli_args.get('display_as_circle')
    if display_as_circle is not None:
        step_ri('Setting display mask')
        grid_size = int(display_as_circle[0])
        print(f'Grid size: {grid_size}')
        circle_size = display_as_circle[1]
        print(f'Circle size: {circle_size}')
        grid_mask = create_grid_mask(grid_size, circle_size)
        display_with_hole = cli_args.get('display_with_hole')
        if display_with_hole is not None:
            print(f'Hole size: {display_with_hole}')
            grid_mask += create_grid_mask(grid_size, display_with_hole)
            grid_mask[grid_mask == 2] = 0
        # The indexes for all the active pixels
        active_pixel_idxs = grid_mask == 1
        grid_plot_path = f'{output_dir}/grid_mask.png'
        print(f'Saving grid mask to {grid_plot_path}')
        plot_wavefront(
            grid_mask,
            'Active Pixels',
            1,
            'Grid Mask',
            grid_plot_path,
            disable_plot_ticks=True,
        )
        total_active_pixels = np.sum(grid_mask)
        print(f'Total of {total_active_pixels} active pixels')

    def _plot_grid(plot_data, filename, plot_title, print_prefix):
        mode_plot_path = f'{output_dir}/{filename}.png'
        if display_as_circle is not None:
            plot_data_2d = np.zeros((grid_size, grid_size))
            plot_data_2d[active_pixel_idxs] = plot_data
            plot_data = plot_data_2d
        plot_wavefront(
            plot_data,
            '',
            1,
            plot_title,
            mode_plot_path,
            disable_plot_ticks=True,
        )
        print(f'{print_prefix}: {mode_plot_path}')

    plot_modes_specific = cli_args.get('plot_modes_specific')
    plot_modes_range = cli_args.get('plot_modes_range')
    if (plot_modes_specific is not None) or (plot_modes_range is not None):
        step_ri('Plotting modes')
        modes_to_plot = []
        if plot_modes_specific is not None:
            print(f'Specific: {plot_modes_specific}')
            modes_to_plot.extend(plot_modes_specific)
        if plot_modes_range is not None:
            idx_low = plot_modes_range[0]
            idx_high = plot_modes_range[1]
            print(f'Range: {idx_low} - {idx_high}')
            modes_to_plot.extend(list(range(idx_low, idx_high + 1)))
        # Remove any duplicates
        modes_to_plot = list(set(modes_to_plot))
        print(f'Will plot {len(modes_to_plot)} modes in total')
        for mode_idx in modes_to_plot:
            print_prefix = f'Mode {mode_idx}'
            if modes_are_complex is not None:
                _plot_grid(
                    modes_data_real[mode_idx],
                    f'mode_{mode_idx}_real',
                    f'Real Mode {mode_idx}',
                    print_prefix,
                )
                _plot_grid(
                    modes_data_imag[mode_idx],
                    f'mode_{mode_idx}_imag',
                    f'Imag Mode {mode_idx}',
                    print_prefix,
                )
            else:
                _plot_grid(
                    modes_data[mode_idx],
                    f'mode_{mode_idx}',
                    f'Mode {mode_idx}',
                    print_prefix,
                )

    reconstruct_data = cli_args.get('reconstruct_data')
    if reconstruct_data is not None:
        step_ri('Data reconstruction')
        (datafile_tag, table_name, row_idx, number_of_modes, apply_mask,
         complex_component) = reconstruct_data
        print(f'Datafile tag: {datafile_tag}')
        print(f'Table name: {table_name}')
        print(f'Row index: {row_idx}')
        print(f'Number of modes: {number_of_modes}')
        print(f'Apply mask: {apply_mask}')
        print(f'Complex component: {complex_component}')

        datafile_path = raw_sim_data_chunk_paths(datafile_tag)[0]
        step('Loading data')
        print(f'Path: {datafile_path}')
        datafile_data = read_hdf(datafile_path)[table_name][:]
        print(f'Datafile shape: {datafile_data.shape}')
        row_data = datafile_data[int(row_idx)]
        print(f'Row shape: {row_data.shape}')
        dec_print_indent()

        reconstruct_data_trim = cli_args.get('reconstruct_data_trim')
        if reconstruct_data_trim is not None:
            step('Trimming row')
            x0, x1, y0, y1 = reconstruct_data_trim
            row_data = row_data[x0:x1]
            row_data = row_data[:, y0:y1]
            print(f'Row shape: {row_data.shape}')
            dec_print_indent()

        if int(apply_mask):
            step('Applying mask')
            row_data[~active_pixel_idxs] = 0
            dec_print_indent()

        step('Formatting the data')
        print('Flattening')
        row_data = np.reshape(row_data, -1)
        print('Removing inactive pixels')
        row_data = row_data[row_data != 0]
        print(f'Row shape: {row_data.shape}')
        dec_print_indent()

        if modes_are_complex is not None:
            step('The data is complex')
            if int(complex_component):
                print('Using the imag component')
                modes_data = modes_data_imag
            else:
                print('Using the real component')
                modes_data = modes_data_real
            dec_print_indent()

        step('Taking desired number of modes')
        modes_data = modes_data[:int(number_of_modes)]
        print(f'Modes shape: {modes_data.shape}')
        dec_print_indent()

        step('Inverting the modes to find the new basis coeffs')
        modes_inv = np.linalg.pinv(modes_data)
        new_basis_coeffs = row_data @ modes_inv
        print(f'New basis coeffs shape: {new_basis_coeffs.shape}')
        reconstructed_row_data = new_basis_coeffs @ modes_data
        # The error when switching to the new basis representation
        error = mse(row_data, reconstructed_row_data)
        print(f'Reconstruction MSE error of {error:0.3e}')
        dec_print_indent()

        if cli_args.get('reconstruct_data_plots'):
            step('Plotting the reconstruction')
            base_filename = f'{datafile_tag}_row{row_idx}'
            _plot_grid(
                row_data,
                f'{base_filename}_orig',
                f'Row {row_idx}: Original',
                'Original',
            )
            _plot_grid(
                reconstructed_row_data,
                f'{base_filename}_reconstructed',
                f'Row {row_idx}: Reconstructed',
                'Reconstructed',
            )
            _plot_grid(
                row_data - reconstructed_row_data,
                f'{base_filename}_diff',
                f'Row {row_idx}: Original - Reconstructed',
                'Diff',
            )
            dec_print_indent()
