import numpy as np


def create_grid_mask(size, circle_size=None):
    """
    Create a grid that can be used as a mask.

    Parameters
    ----------
    size : int
        Number of rows and columns, must be square; will produce a grid of the
        shape (size, size).
    circle_size : float, optional
        If set, the scaled distance from the center that will form the circle on
        the mask. Otherwise, all points on the square grid will remain True.

    Returns
    -------
    np.array
        The 2D array of type `int`.
    """

    # Default to a square
    mask = np.ones((size, size), dtype=int)
    # If the `circle_size` is set, then the mask should be circular
    if circle_size:
        # How far each point is from the middle
        distances = np.arange(size) - (size // 2)
        # If there are an even number of points, then there cannot
        # be one at 0, so the points will be symmetric about 0
        if size % 2 == 0:
            distances = distances + 0.5
        # Scale the distances between -1 to 1
        distances = distances / np.max(distances)
        # Create a 2D grid of distances from the center
        distance_grid = np.sqrt(distances[None, :]**2 + distances[:, None]**2)
        # Mask out all pixels that form a circle
        mask[distance_grid > circle_size] = 0
    return mask
