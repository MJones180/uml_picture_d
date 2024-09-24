"""
This file is for a 32x32 pixel CCD.
"""

# Diameter of the initial beam
INIT_BEAM_D = 1

# Ratio of the beam to the grid
BEAM_RATIO = 1

# Number of pixels and sampling size for the final CCD
CCD_PIXELS = 32
CCD_SAMPLING = INIT_BEAM_D / CCD_PIXELS

# We do not want to prop through an optical train for this
OPTICAL_TRAIN = []
