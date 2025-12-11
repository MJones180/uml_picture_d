import numpy as np
from utils.constants import (CAM_BIAS, CAM_BITDEPTH, CAM_DARK_RATE,
                             CAM_FULL_WELL, CAM_GAIN, CAM_READ_NOISE)
from utils.norm import sum_to_one

# The camera settings are hard coded in this file (for better or for worse).
# For now, this should be fine since there are not many cameras.
CAMERAS = {
    # 'camera_name': {
    #     CAM_BIAS: [electrons] int,
    #     CAM_BITDEPTH: int - 2^n quantization levels,
    #     CAM_DARK_RATE: [electrons/second/pixel] int,
    #     CAM_FULL_WELL: [electrons] int - number of electrons a pixel can hold,
    #     CAM_GAIN: [electrons/adu] float,
    #     CAM_READ_NOISE: [RMS electrons/pixel] float,
    # }
    'imperx_b0620': {
        CAM_BIAS: 0,
        CAM_BITDEPTH: 12,
        CAM_DARK_RATE: 1000,
        CAM_FULL_WELL: 20000,
        CAM_GAIN: 1.0,
        CAM_READ_NOISE: 16.0,
    }
}


def simulate_camera(inputs_wfs, cam_name, exposure_time, countrate):
    """
    Simulate a camera when reading in the wavefront. This adds camera noise,
    discretized light levels, and saturation.

    Parameters
    ----------
    inputs_wfs : np.array
        The input wavefronts in the shape of (wfs, pixels, pixels).
    cam_name : string
        Name of the camera to use.
    exposure_time : float
        The exposure time in seconds.
    countrate : float
        The count rate - number of photons/second hitting the camera.

    Returns
    -------
    np.array
        The wavefronts on the simulated camera.
    """

    # Grab the specs of the camera being used
    cam_specs = CAMERAS[cam_name]

    # ===============================
    # Add the noise to the wavefronts
    # ===============================

    # Create a random number generator
    rng = np.random.default_rng()
    # Number of pixels on the camera (total_pixels = pixels**2), must be square
    pixels = int(inputs_wfs.shape[1])
    # The wfs in terms of the number of counts
    wfs_counts = sum_to_one(inputs_wfs, (1, 2)) * countrate * exposure_time
    # The dark noise on the camera
    dark_noise = exposure_time * cam_specs[CAM_DARK_RATE]
    # The read noise on the camera
    read_noise = cam_specs[CAM_READ_NOISE] * rng.normal(size=(pixels, pixels))
    # Add Poisson noise, the values must be >= 0
    wfs_with_noise = rng.poisson(np.clip(wfs_counts + dark_noise, 0, None))
    # Add the read noise and the camera's bias
    wfs_with_noise = wfs_with_noise + read_noise + cam_specs[CAM_BIAS]
    # Cap any pixels that are saturated
    np.clip(wfs_with_noise, 0, cam_specs[CAM_FULL_WELL], wfs_with_noise)

    # ===================================
    # Do the analog to digital conversion
    # ===================================

    # Apply the camera gain
    wfs_with_noise /= cam_specs[CAM_GAIN]
    # Digitize the counts
    wfs_with_noise = np.floor(wfs_with_noise)
    # The largest number of electrons that can be in a single pixel
    max_electrons = 2**cam_specs[CAM_BITDEPTH] - 1
    # Clip the electron counts
    np.clip(wfs_with_noise, 0, max_electrons, wfs_with_noise)
    # Convert to float32 from float64
    wfs_with_noise = wfs_with_noise.astype(np.float32)
    return wfs_with_noise
