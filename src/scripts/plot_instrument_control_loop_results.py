from astropy.io import fits
import numpy as np
from utils.constants import CONTROL_LOOP_RESULTS_P
from utils.path import make_dir
from utils.plots.plot_control_loop_zernikes_subplots import plot_control_loop_zernikes_subplots  # noqa: E501
from utils.printing_and_logging import step_ri, title


def plot_instrument_control_loop_results_parser(subparsers):
    subparser = subparsers.add_parser(
        'plot_instrument_control_loop_results',
        help='plot control loop results that were run on PICTURE',
    )
    subparser.set_defaults(main=plot_instrument_control_loop_results)
    subparser.add_argument(
        'tag',
        help=('name of the folder located in `output/control_loop_results`; '
              'this folder must contain the `XXX.fits`, `XXX.fits`, and '
              '`XXX.fits` files; a new nested folder will be created with '
              'the name of `plots`'),
    )
    subparser.add_argument(
        '--zernike-range',
        type=int,
        nargs=2,
        help='two arguments expected: low Zernike index, high Zernike index',
    )


def plot_instrument_control_loop_results(cli_args):
    title('Plot instrument control loop results script')

    step_ri('Loading in the data')
    tag = cli_args['tag']
    base_path = f'{CONTROL_LOOP_RESULTS_P}/{tag}'
    print(f'Base path: {base_path}')

    def _load_data(filename, scale_data=True):
        filepath = f'{base_path}/{filename}.fits'
        print(f'Loading `{filename}` ({filepath})')
        data = fits.getdata(filepath)
        if scale_data:
            # The data is going from nm surface error -> m wavefront error
            data *= 2e-9
        return data

    total_time = _load_data('time', scale_data=False)[-1]
    zcal = _load_data('zcal')
    zmes = _load_data('zmes')

    step_ri('Zernike terms')
    zernike_low, zernike_high = cli_args['zernike_range']
    zernike_terms = np.arange(zernike_low, zernike_high + 1)
    print(zernike_terms)

    step_ri('Plotting')
    out_dir = f'{base_path}/plots/'
    print(f'Creating plot dir: {out_dir}')
    make_dir(out_dir)
    plot_control_loop_zernikes_subplots(
        zernike_terms,
        zcal,
        'Injected Zernike Aberrations',
        total_time,
        6,
        4,
        f'{out_dir}/zcal.png',
    )
    plot_control_loop_zernikes_subplots(
        zernike_terms,
        zmes,
        'Measured Zernike Aberrations',
        total_time,
        6,
        4,
        f'{out_dir}/zmes.png',
        add_std=True,
    )
