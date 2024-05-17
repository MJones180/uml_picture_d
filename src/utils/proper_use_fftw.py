import proper
from utils.terminate_with_message import terminate_with_message


def proper_use_fftw():
    if proper.use_fftw:
        print('Using pyFFTW')
    else:
        print('Enabling pyFFTW...')
        proper.prop_use_fftw()
        if proper.use_fftw:
            print('pyFFTW is now being used')
        else:
            terminate_with_message('pyFFTW is still not being used?')


# print('Adding wisdom')
# proper.prop_fftw_wisdom(2048)
