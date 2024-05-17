import proper
from utils.printing_and_logging import dec_print_indent, step
from utils.terminate_with_message import terminate_with_message


def proper_use_fftw():
    if proper.use_fftw:
        print('Using pyFFTW for PROPER.')
    else:
        step('Enabling pyFFTW for PROPER...')
        proper.prop_use_fftw()
        if proper.use_fftw:
            print('pyFFTW is now being used!')
            dec_print_indent()
        else:
            terminate_with_message('pyFFTW is still not being used?')
