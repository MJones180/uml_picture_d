"""
Generate a CSV file with time steps containing different Zernike coefficients.
This can then be plugged into the `control_loop_run` script.
"""

from utils.printing_and_logging import step_ri, title
from utils.shared_argparser_args import shared_argparser_args


def gen_zernike_time_steps_parser(subparsers):
    subparser = subparsers.add_parser(
        'gen_zernike_time_steps',
        help='',
    )
    subparser.set_defaults(main=gen_zernike_time_steps)
    shared_argparser_args(subparser, ['tag', 'epoch'])
    shared_argparser_args(subparser, ['force_cpu'])


def gen_zernike_time_steps(cli_args):
    title('Generate zernike time steps script')
