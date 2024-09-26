"""
This file is for a 32x32 pixel camera.
"""

# Diameter of the initial beam
INIT_BEAM_D = 1

# Ratio of the beam to the grid
BEAM_RATIO = 1

# Number of pixels and sampling size for the final camera
CAMERA_PIXELS = 32
CAMERA_SAMPLING = INIT_BEAM_D / CAMERA_PIXELS

# We do not want to prop through an optical train for this
OPTICAL_TRAIN = []
