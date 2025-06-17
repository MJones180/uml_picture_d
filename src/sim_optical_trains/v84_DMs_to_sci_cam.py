"""
This file is for V84 with a 32x32 pixel camera and a sampling of 7.4 microns.
This optical train goes from HODM 1 to the science camera.
A diagram of PICTURE-D can be found in the `diagrams` folder under the
`optical_design_v84.png` file. The VVC function is used without MFT.
"""

from cbm_vvc_mft import cbm_vvc_approx
import proper
from utils.constants import (DM_ACTUATOR_HEIGHTS, DM_ACTUATOR_SPACING, DM_MASK,
                             VVC_CHARGE)
from utils.create_grid_mask import create_grid_mask

# Diameter of the initial beam
INIT_BEAM_D = 9e-3

# Lyot stop
lyot_stop_hole_r = INIT_BEAM_D * 0.9 / 2

# Ratio of the beam to the grid
BEAM_RATIO = 0.5
# INIT_BEAM_D / lyot_stop_outer_d * 0.95

# Number of pixels and sampling size for the final camera
CAMERA_PIXELS = 100
CAMERA_SAMPLING = 7.4e-6

# Both DMs in this train are the same
DM_RADIUS = 17
DM_SPACING = 0.003  # 3 mm

# A list of each DM in the optical train, starting at index 0.
# Both DMs are circular on a 34x34 grid.
DM_LIST = {
    0: {
        DM_ACTUATOR_SPACING: DM_SPACING,
        DM_MASK: create_grid_mask(DM_RADIUS * 2, 1.06),
    },
    1: {
        DM_ACTUATOR_SPACING: DM_SPACING,
        DM_MASK: create_grid_mask(DM_RADIUS * 2, 1.06),
    }
}

# All distances are in meters. Assume the beam starts at HODM 1.
OPTICAL_TRAIN = [
    [
        'HODM 1',
        lambda wf, extra_params: proper.prop_dm(
            wf,
            extra_params[DM_ACTUATOR_HEIGHTS(0)],
            DM_RADIUS,
            DM_RADIUS,
            DM_SPACING,
        ),
    ],
    [
        'Prop to HODM 2 [From HODM 1]',
        lambda wf: proper.prop_propagate(wf, 0.40894),
    ],
    [
        'HODM 2',
        lambda wf, extra_params: proper.prop_dm(
            wf,
            extra_params[DM_ACTUATOR_HEIGHTS(1)],
            DM_RADIUS,
            DM_RADIUS,
            DM_SPACING,
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
