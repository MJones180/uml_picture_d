import proper

if proper.use_fftw:
    print('Using pyFFTW')
else:
    print('Enabling pyFFTW...')
    proper.prop_use_fftw()
    if proper.use_fftw:
        print('pyFFTW is now being used')
    else:
        print('pyFFTW is still not being used?')
        quit()

# print('Adding wisdom')
# proper.prop_fftw_wisdom(2048)
