"""
Run a control loop to flatten out a wavefront.

A diagram of how the control loop works can be found at
`diagrams/wfs_control_loop.png`.
"""

from utils.printing_and_logging import step_ri, title
from utils.shared_argparser_args import shared_argparser_args


def control_loop_run_parser(subparsers):
    subparser = subparsers.add_parser(
        'control_loop_run',
        help='',
    )
    subparser.set_defaults(main=control_loop_run)
    shared_argparser_args(subparser, ['tag', 'epoch'])
    shared_argparser_args(subparser, ['force_cpu'])


def control_loop_run(cli_args):
    title('Control loop run script')
