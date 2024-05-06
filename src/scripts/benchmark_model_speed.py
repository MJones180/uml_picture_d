"""
This script times how long it takes a trained neural network to process one row
of data.
"""

import time
import torch
from utils.constants import OUTPUT_MIN_X, OUTPUT_MAX_MIN_DIFF
from utils.load_model import LoadModel
from utils.norm import min_max_denorm
from utils.printing_and_logging import step_ri, title


def benchmark_model_speed_parser(subparsers):
    """
    Example commands:
    python3 main.py benchmark_model_speed v1a 110
    """
    subparser = subparsers.add_parser(
        'benchmark_model_speed',
        help='test a trained model',
    )
    subparser.set_defaults(main=benchmark_model_speed)
    subparser.add_argument(
        'tag',
        help='tag of the model',
    )
    subparser.add_argument(
        'epoch',
        help='epoch of the trained model to test (just the number part)',
    )
    subparser.add_argument(
        '--iterations',
        default=1000,
        help='number of iterations to perform',
    )


def benchmark_model_speed(cli_args):
    title('Benchmark model speed script')

    tag = cli_args['tag']
    epoch = cli_args['epoch']
    iterations = cli_args['iterations']

    loaded_model = LoadModel(tag, epoch, eval_mode=True)
    model = loaded_model.get_model()
    network = loaded_model.get_network()
    norm_values = loaded_model.get_norm_values()

    output_max_min_diff = norm_values[OUTPUT_MAX_MIN_DIFF]
    output_min_x = norm_values[OUTPUT_MIN_X]

    example_input = network.example_input()

    def _run_model():
        with torch.no_grad():
            model_output = model(example_input).numpy()
        return model_output

    def _run_model_and_denorm():
        model_output = _run_model()
        min_max_denorm(
            model_output,
            output_max_min_diff,
            output_min_x,
        )

    def _benchmark(func):
        total_time = 0
        for i in range(iterations):
            start = time.time()
            func()
            finish = time.time()
            total_time += finish - start
        return total_time / iterations

    step_ri(f'Running benchmark ({iterations} iterations)')
    print('Average time for the nn to process one row: ',
          _benchmark(_run_model))
    print('Average time for nn to process one row and denormalize the data: ',
          _benchmark(_run_model_and_denorm))
