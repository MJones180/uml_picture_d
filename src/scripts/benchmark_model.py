"""
This script times how long it takes to run a trained model. The `network_info`
script has the same functionality for an untrained model (just a network).
"""

import time
from utils.model import Model
from utils.printing_and_logging import step, step_ri, title
from utils.shared_argparser_args import shared_argparser_args


def benchmark_model_parser(subparsers):
    subparser = subparsers.add_parser(
        'benchmark_model',
        help='benchmark how long it takes to run a model',
    )
    subparser.set_defaults(main=benchmark_model)
    shared_argparser_args(subparser, ['tag', 'epoch'])
    shared_argparser_args(subparser, ['force_cpu'])
    subparser.add_argument(
        'benchmark',
        type=int,
        help='benchmark network by running n rows',
    )


def benchmark_model(cli_args):
    title('Benchmark model script')

    step_ri('Loading in the model')
    force_cpu = cli_args.get('force_cpu')
    model = Model(cli_args['tag'], cli_args['epoch'], force_cpu=force_cpu)
    input_data = model.network.example_input()

    iterations = cli_args['benchmark']
    step_ri(f'Running benchmark ({iterations} iterations)')
    start_time = time.time()
    for i in range(iterations):
        model(input_data)
    avg_time = (time.time() - start_time) / iterations
    step('Average time for the nn to run one row')
    print(f'Seconds (s): {avg_time:0.6f}')
    print(f'Milliseconds (ms): {(avg_time * 1e3):0.3f}')
