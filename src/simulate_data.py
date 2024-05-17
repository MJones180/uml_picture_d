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
    # plt.imshow(intensity)
    plt.colorbar(label=colorbar_label)
    if SAVE_PLOTS:
        plt.savefig(f'plot_output/{plot_idx}.png', dpi=300)
    else:
        plt.show()


def simple_test():
    wf = proper.prop_begin(INIT_BEAM_D, REF_WL, GRID_POINTS, BEAM_RATIO)

    proper.prop_circular_aperture(wf, INIT_BEAM_D / 2)

    # Define entrance
    proper.prop_define_entrance(wf)
    plot_wf_intensity(wf, 'Entrance', 0)

    proper.prop_propagate(wf, 0.7251065, 'OAP3 [From HODM 1]')
    plot_wf_intensity(wf, 'Prop to OAP3 [From HODM 1]', 1)

    proper.prop_lens(wf, 0.511, 'OAP3')

    proper.prop_propagate(wf, 0.511, 'VVC [From OAP3]')
    plot_wf_intensity(wf, 'Prop to VVC [From OAP3]', 2)

    cbm_vvc_mft(
        wavefront=wf,
        charge=VCC_CHARGE,
        spot_rad=10e-6,
        offset=0,
        ramp_sign=1,
        beam_ratio=BEAM_RATIO,
        d_occulter_lyotcoll=0.511,
        fl_lyotcoll=0.511,
        d_lyotcoll_lyotstop=0.2966085,
    )
    plot_wf_intensity(wf, 'VVC', 3)

    proper.prop_propagate(wf, 0.511, 'OAP4 [From VVC]')
    plot_wf_intensity(wf, 'Prop to OAP4 [From VVC]', 4)

    proper.prop_lens(wf, 0.511, 'OAP4')

    proper.prop_propagate(wf, 0.2966085, 'Lyot Stop [From OAP4]')
    plot_wf_intensity(wf, 'Prop to Lyot Stop [From OAP4]', 5)

    outer_r = LYOT_STOP_OUTER_D / 2
    inner_r = LYOT_STOP_HOLE_D / 2
    proper.prop_elliptical_aperture(wf, outer_r, outer_r)
    proper.prop_elliptical_obscuration(wf, inner_r, inner_r)
    plot_wf_intensity(wf, 'Lyot Stop', 6)

    proper.prop_propagate(wf, 0.1016, 'Final Lens [From Lyot Stop]')
    plot_wf_intensity(wf, 'Final Lens [From Lyot Stop]', 7)

    proper.prop_lens(wf, 0.25165, 'Final Lens')

    proper.prop_propagate(wf, 0.247396, 'CCD [From final lens]')
    plot_wf_intensity(wf, 'CCD [From final lens]', 8)

    # Pixel size for the CCD: 4.5 microns
    # Final image should be 32 x 32 pixels, greys
    # Take sampling which is x microns, and rebin to match

    return proper.prop_end(wf)


(wf, sampling) = simple_test()
