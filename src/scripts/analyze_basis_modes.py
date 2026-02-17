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
from utils.printing_and_logging import step_ri, title


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

            def _plot_mode(flat_mode_data, filename, plot_title):
                mode_plot_path = f'{output_dir}/{filename}.png'
                if display_as_circle is not None:
                    mode_data = np.zeros((grid_size, grid_size))
                    mode_data[active_pixel_idxs] = flat_mode_data
                else:
                    mode_data = flat_mode_data
                plot_wavefront(
                    mode_data,
                    'Weight',
                    1,
                    plot_title,
                    mode_plot_path,
                    disable_plot_ticks=True,
                )
                print(f'Mode {mode_idx}: {mode_plot_path}')

            if modes_are_complex is not None:
                _plot_mode(
                    modes_data_real[mode_idx],
                    f'mode_{mode_idx}_real',
                    f'Real Mode {mode_idx}',
                )
                _plot_mode(
                    modes_data_imag[mode_idx],
                    f'mode_{mode_idx}_imag',
                    f'Imag Mode {mode_idx}',
                )
            else:
                _plot_mode(
                    modes_data[mode_idx],
                    f'mode_{mode_idx}',
                    f'Mode {mode_idx}',
                )
