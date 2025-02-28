"""
This file is for V84 with a 32x32 pixel camera and a sampling of 7.4 microns.
This optical train goes from HODM 1 to the science camera.
A diagram of PICTURE-D can be found in the `diagrams` folder under the
`optical_design_v84.png` file. The VVC function is used without MFT.
"""

from cbm_vvc_mft import cbm_vvc_approx
import proper
from utils.constants import VVC_CHARGE

# Diameter of the initial beam
INIT_BEAM_D = 9e-3

# Lyot stop
lyot_stop_hole_r = INIT_BEAM_D * 0.9 / 2

# Ratio of the beam to the grid
BEAM_RATIO = 0.5

# Number of pixels and sampling size for the final camera
CAMERA_PIXELS = 100
CAMERA_SAMPLING = 7.4e-6

# Number of actuators and spacing between them on the DM
DM_ACTUATOR_COUNT = 34
DM_ACTUATOR_COUNT_HALF = DM_ACTUATOR_COUNT / 2
DM_ACTUATOR_SPACING = 0.003  # 3 mm

# All distances are in meters. Assume the beam starts at HODM 1. Treat the
# DMs as if they are not there.
OPTICAL_TRAIN = [
    [
        'HODM 1',
        # The real DM is circular, this one is square
        lambda wf, actuator_heights: proper.prop_dm(
            wf,
            actuator_heights,
            DM_ACTUATOR_COUNT_HALF,
            DM_ACTUATOR_COUNT_HALF,
            DM_ACTUATOR_SPACING,
        ),
    ],
    [
        'Prop to HODM 2 [From HODM 1]',
        lambda wf: proper.prop_propagate(wf, 0.40894),
    ],
    [
        'HODM 2',
        # The real DM is circular, this one is square
        lambda wf, actuator_heights: proper.prop_dm(
            wf,
            actuator_heights,
            DM_ACTUATOR_COUNT_HALF,
            DM_ACTUATOR_COUNT_HALF,
            DM_ACTUATOR_SPACING,
        ),
    ],
    [
        'Prop to OAP 3 [From HODM 2]',
        lambda wf: proper.prop_propagate(wf, 0.319024),
    ],
    # OAP3
    lambda wf: proper.prop_lens(wf, 0.511),
    [
        'Prop to VVC [From OAP3]',
        lambda wf: proper.prop_propagate(wf, 0.511),
    ],
    [
        'VVC',
        lambda wf: cbm_vvc_approx(
            wavefront=wf,
            charge=VVC_CHARGE,
            offset=0,
            ramp_sign=1,
            center_spot_scaling=1.75,
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
        lambda wf: proper.prop_propagate(wf, 0.3101),
    ],
    [
        'Lyot Stop',
        lambda wf: proper.prop_elliptical_aperture(
            wf,
            lyot_stop_hole_r,
            lyot_stop_hole_r,
        ),
    ],
    [
        'Prop to Spatial Filter Lens [From Lyot Stop]',
        lambda wf: proper.prop_propagate(wf, 0.038),
    ],
    # Lens before spatial filter
    lambda wf: proper.prop_lens(wf, 0.124968),
    [
        'Prop to Spatial Filter [From Spatial Filter Lens]',
        lambda wf: proper.prop_propagate(wf, 0.124968),
    ],
    [
        'Spatial Filter',
        lambda wf: proper.prop_elliptical_aperture(
            wf,
            0.0127,
            0.0127,
        ),
    ],
    [
        'Prop to Spatial Filter Lens [From Spatial Filter]',
        lambda wf: proper.prop_propagate(wf, 0.124968),
    ],
    # Lens after spatial filter
    lambda wf: proper.prop_lens(wf, 0.124968),
    [
        'Final Lens [From Spatial Filter Lens]',
        lambda wf: proper.prop_propagate(wf, 0.0772668),
    ],
    # Final lens
    lambda wf: proper.prop_lens(wf, 0.2),
    [
        'Camera [From final lens]',
        lambda wf: proper.prop_propagate(wf, 0.1824736),
    ],
]
