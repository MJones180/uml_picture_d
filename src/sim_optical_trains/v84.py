from cbm_vvc_mft import cbm_vvc_mft
import proper
from utils.constants import VVC_CHARGE

# Diameter of 9 mm
INIT_BEAM_D = 9e-3
# Lyot stop
LYOT_STOP_HOLE_D = INIT_BEAM_D * 0.9
LYOT_STOP_OUTER_D = 50.8e-3
BEAM_RATIO = INIT_BEAM_D / LYOT_STOP_OUTER_D * 0.95

outer_r = LYOT_STOP_OUTER_D / 2
inner_r = LYOT_STOP_HOLE_D / 2

OPTICAL_TRAIN = [
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
            charge=VVC_CHARGE,
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
