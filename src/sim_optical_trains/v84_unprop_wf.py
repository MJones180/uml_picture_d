# A copy of V84 used to just display the zernike aberration map

# Diameter of the initial beam
INIT_BEAM_D = 9e-3

# Ratio of the beam to the grid
BEAM_RATIO = 1

# Number of pixels and sampling size for the final CCD
CCD_PIXELS = 32
CCD_SAMPLING = INIT_BEAM_D / CCD_PIXELS

# All distances are in meters. Assume the beam starts at HODM 1. Treat the
# DMs as if they are not there.
OPTICAL_TRAIN = []
