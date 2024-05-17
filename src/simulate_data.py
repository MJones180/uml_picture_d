import add_packages_dir_to_path  # noqa: F401

import proper
import matplotlib.pyplot as plt
import numpy as np
from utils.load_optical_train import load_optical_train
from utils.proper_use_fftw import proper_use_fftw

proper_use_fftw()

# Number of grid points in each direction
GRID_POINTS = 1024
# Reference wavelength is 600 nm
REF_WL = 600e-9
# Diameter of 9 mm
INIT_BEAM_D = 9e-3

# Plot the log of the values
SHOW_LOG_PLOT = False
# Save the plots instead of displaying them
SAVE_PLOTS = False

PLOTTING = False


def plot_wf_intensity(wf, title, plot_idx):
    amp = proper.prop_get_amplitude(wf)
    intensity = amp**2
    colorbar_label = 'intensity'
    if SHOW_LOG_PLOT:
        colorbar_label = 'log10(intensity)'
        intensity = np.log10(intensity)
        intensity[intensity == -np.inf] = 0
    plt.clf()
    plt.title(title)
    plt.imshow(intensity, cmap='Greys_r')
    plt.colorbar(label=colorbar_label)
    if SAVE_PLOTS:
        plt.savefig(f'plot_output/{plot_idx}.png', dpi=300)
    else:
        plt.show()


LYOT_STOP_HOLE_D = INIT_BEAM_D * 0.9
LYOT_STOP_OUTER_D = 50.8e-3
BEAM_RATIO = INIT_BEAM_D / LYOT_STOP_OUTER_D * 0.95


def simple_test():
    wavefront = proper.prop_begin(INIT_BEAM_D, REF_WL, GRID_POINTS, BEAM_RATIO)

    plot_idx = 0
    x = load_optical_train('v84')
    print(x)
    quit()
    for step in train:
        if type(step) is list:
            step[1](wavefront)
            if PLOTTING:
                plot_wf_intensity(wavefront, step[0], plot_idx)
                plot_idx += 1
        else:
            step(wavefront)

    return proper.prop_end(wavefront)


(wf, sampling) = simple_test()
print(sampling)
quit()
