"""
This file is for V84 with a 32x32 pixel camera and a sampling of 7.4 microns.
This optical train goes from HODM 1 to the science camera.
A diagram of PICTURE-D can be found in the `diagrams` folder under the
`optical_design_v84.png` file. The VVC function is used without MFT.
"""

from cbm_vvc_mft import cbm_vvc_approx
import proper
from utils.constants import DM_ACTUATOR_HEIGHTS, VVC_CHARGE

# Diameter of the initial beam
INIT_BEAM_D = 9e-3

# Lyot stop
lyot_stop_hole_r = INIT_BEAM_D * 0.9 / 2

# Ratio of the beam to the grid
BEAM_RATIO = 0.5

# Number of pixels and sampling size for the final camera
CAMERA_PIXELS = 100
CAMERA_SAMPLING = 7.4e-6

# The DMs are assumed to be a square, so this is the number of rows and
# columns of actuators
DM_ACTUATOR_COUNT = 34
DM_ACTUATOR_COUNT_HALF = DM_ACTUATOR_COUNT / 2
# The spacing between each actuator
DM_ACTUATOR_SPACING = 0.003  # 3 mm

# All distances are in meters. Assume the beam starts at HODM 1.
OPTICAL_TRAIN = [
    [
        'HODM 1',
        # The real DM is circular, this one is square
        lambda wf, extra_params: proper.prop_dm(
            wf,
            extra_params[DM_ACTUATOR_HEIGHTS][0],
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
        lambda wf, extra_params: proper.prop_dm(
            wf,
            extra_params[DM_ACTUATOR_HEIGHTS][1],
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
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~ The spatial filter is being   ~~~~~
    # ~~~~~ ignored in this optical train ~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    [
        'Prop to Final Lens [From Lyot Stop]',
        lambda wf: proper.prop_propagate(wf, 0.3652028),
    ],
    # Final lens
    lambda wf: proper.prop_lens(wf, 0.2),
    [
        'Camera [From final lens]',
        lambda wf: proper.prop_propagate(wf, 0.1824736),
    ],
]
