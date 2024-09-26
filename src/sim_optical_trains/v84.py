"""
This file is for V84 with a 32x32 pixel camera and a sampling of 7.4 microns.
"""

from cbm_vvc_mft import cbm_vvc_mft
import proper
from utils.constants import VVC_CHARGE

# Diameter of the initial beam
INIT_BEAM_D = 9e-3

# Lyot stop
lyot_stop_outer_d = 50.8e-3
lyot_stop_outer_r = lyot_stop_outer_d / 2
lyot_stop_hole_r = INIT_BEAM_D * 0.9 / 2

# Ratio of the beam to the grid
# = INIT_BEAM_D / lyot_stop_outer_d * 0.95
BEAM_RATIO = 0.1683

# Number of pixels and sampling size for the final camera
CAMERA_PIXELS = 32
CAMERA_SAMPLING = 7.4e-6

# All distances are in meters. Assume the beam starts at HODM 1. Treat the
# DMs as if they are not there.
OPTICAL_TRAIN = [
    [
        'Prop to OAP3 [From HODM 1]',
        lambda wf: proper.prop_propagate(wf, 0.7251065),
    ],
    # OAP3
    lambda wf: proper.prop_lens(wf, 0.511),
    [
        'Prop to VVC [From OAP3]',
        lambda wf: proper.prop_propagate(wf, 0.511),
    ],
    [
        'VVC',
        lambda wf: cbm_vvc_mft(
            wavefront=wf,
            charge=VVC_CHARGE,
            offset=0,
            ramp_sign=1,
            spot_rad=10e-6,
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
    # OAP4
    lambda wf: proper.prop_lens(wf, 0.511),
    [
        'Prop to Lyot Stop [From OAP4]',
        lambda wf: proper.prop_propagate(wf, 0.2966085),
    ],
    lambda wf: proper.prop_elliptical_aperture(
        wf,
        lyot_stop_outer_r,
        lyot_stop_outer_r,
    ),
    [
        'Lyot Stop',
        lambda wf: proper.prop_elliptical_obscuration(
            wf,
            lyot_stop_hole_r,
            lyot_stop_hole_r,
        ),
    ],
    [
        'Final Lens [From Lyot Stop]',
        lambda wf: proper.prop_propagate(wf, 0.1016),
    ],
    # Final lens
    lambda wf: proper.prop_lens(wf, 0.25),
    [
        'Camera [From final lens]',
        # lambda wf: proper.prop_propagate(wf, 0.247396),
        lambda wf: proper.prop_propagate(wf, 0.24835),
    ],
]
