import time
from utils.printing_and_logging import step, step_ri


def benchmark_nn(iterations, nn_call_wrapper, model=None):
    title_str = f'[{model}] ' if model is not None else ''
    title_str += f'Running benchmark ({iterations} iterations)'
    step_ri(title_str)
    start_time = time.time()
    for i in range(iterations):
        nn_call_wrapper()
    avg_time = (time.time() - start_time) / iterations
    step('Average time for the nn to run one row')
    print(f'Seconds (s): {avg_time:0.6f}')
    print(f'Milliseconds (ms): {(avg_time * 1e3):0.3f}')
