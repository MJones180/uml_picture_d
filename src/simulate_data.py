import add_packages_dir_to_path  # noqa: F401

from cbm_vvc_mft import cbm_vvc_mft
import proper
import matplotlib.pyplot as plt
import numpy as np
from utils.proper_use_fftw import proper_use_fftw

proper_use_fftw()

# Number of grid points in each direction
GRID_POINTS = 1024
# Reference wavelength is 600 nm
REF_WL = 600e-9
# Diameter of 9 mm
INIT_BEAM_D = 9e-3
# Lyot stop
LYOT_STOP_HOLE_D = INIT_BEAM_D * 0.9
LYOT_STOP_OUTER_D = 50.8e-3
# BEAM_RATIO = INIT_BEAM_D / LYOT_STOP_OUTER_D
BEAM_RATIO = INIT_BEAM_D / LYOT_STOP_OUTER_D * 0.95
print(BEAM_RATIO)

VCC_CHARGE = 6

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


outer_r = LYOT_STOP_OUTER_D / 2
inner_r = LYOT_STOP_HOLE_D / 2
train = [
    lambda wf: proper.prop_circular_aperture(wf, INIT_BEAM_D / 2),
    [
        'Entrance',
        lambda wf: proper.prop_define_entrance(wf),
    ],
    [
        'Prop to OAP3 [From HODM 1]',
        lambda wf: proper.prop_propagate(wf, 0.7251065),
    ],
    lambda wf: proper.prop_lens(wf, 0.511, 'OAP3'),
    [
        'Prop to VVC [From OAP3]',
        lambda wf: proper.prop_propagate(wf, 0.511),
    ],
    [
        'VVC',
        lambda wf: cbm_vvc_mft(
            wavefront=wf,
            charge=VCC_CHARGE,
            spot_rad=10e-6,
            offset=0,
            ramp_sign=1,
            beam_ratio=BEAM_RATIO,
            d_occulter_lyotcoll=0.511,
            fl_lyotcoll=0.511,
            d_lyotcoll_lyotstop=0.2966085,
        ),
    ],
    [
        'Prop to OAP4 [From VVC]',
        lambda wf: proper.prop_propagate(wf, 0.511),
    ],
    lambda wf: proper.prop_lens(wf, 0.511),
    [
        'Prop to Lyot Stop [From OAP4]',
        lambda wf: proper.prop_propagate(wf, 0.2966085),
    ],
    lambda wf: proper.prop_elliptical_aperture(wf, outer_r, outer_r),
    [
        'Lyot Stop',
        lambda wf: proper.prop_elliptical_obscuration(wf, inner_r, inner_r),
    ],
    [
        'Final Lens [From Lyot Stop]',
        lambda wf: proper.prop_propagate(wf, 0.1016),
    ],
    lambda wf: proper.prop_lens(wf, 0.25165),
    [
        'CCD [From final lens]',
        lambda wf: proper.prop_propagate(wf, 0.247396),
    ],
]


def simple_test():
    wavefront = proper.prop_begin(INIT_BEAM_D, REF_WL, GRID_POINTS, BEAM_RATIO)

    plot_idx = 0
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
