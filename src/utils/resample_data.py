import numpy as np
from PIL import Image


def resample_data(
        data,
        sampling,
        final_sampling,
        final_points,
        scale_factor_threshold=(0.95, 1.05),
):
    """
    Resample one grid to another grid size.

    Parameters
    ----------
    data : np.array
        The 2D array containing the grid that needs to be resampled. This grid
        must be a square.
    sampling : float
        The spatial size of each point in the current grid (the `data` arg).
    final_sampling : float
        The spatial size of each point in the resampled, output grid.
    final_points : int
        The number of points in the resampled, output grid.
    scale_factor_threshold : tuple(float, float), optional
        The lower and upper bounds which determine how to resample the grid
        based on the ratio of their sizes (sampling * points).

    Returns
    -------
    np.array
        The resampled grid.

    Notes
    -----
    This utility has the following assumptions and constraints:
        - The input grid must be a 2D square
        - If the two grids are approximately the same, nothing will be done.
        - If the original grid is bigger than the output grid:
            - This code is optimized for an output grid that is much smaller
              than the current grid; an intermediary performance step is done
              for this. This step may end up being slower for comparable initial
              and final grids.
            - The output grid will be zoomed in toward the middle.
        - If the original grid is smaller than the output grid: the original
          grid will be resized to the corresponding number of points on the
          output grid, then placed in the middle of the output grid and padded
          with zeros.
        - The output grid will be normalized so that the sum stays the same.
    """
    # Grab the number of grid points
    grid_points = data.shape[0]
    # Total grid size
    grid_size = sampling * grid_points
    # Total grid size of the output
    final_grid_size = final_sampling * final_points
    # How much bigger our current grid is than the output grid
    scale_factor = grid_size / final_grid_size
    # The output grid is approximately the same size as the current grid
    if scale_factor_threshold[0] <= scale_factor <= scale_factor_threshold[1]:
        return data
    # The output grid is smaller than the current grid
    elif scale_factor > scale_factor_threshold[1]:
        # Instead of jumping right to performing the crops and resizings, we
        # perform an intermediary step with a crop first so that we do not
        # need to allocate so much memory (it is also faster).
        # Grab the number of points that cover the output grid with a tiny bit
        # of padding so that the full output grid is covered.
        point_count_crop = int(np.ceil(grid_points / scale_factor * 1.05))
        point_count_crop_half = point_count_crop // 2
        grid_points_half = int(grid_points / 2)
        lower = grid_points_half - point_count_crop_half
        upper = grid_points_half + point_count_crop_half
        # A cropped version of the data that contains mainly the output grid
        data_out = data[lower:upper, lower:upper]
        # When we resize to the larger array by this updated scale factor, the
        # `point_count_crop` points in the middle will be the points that are
        # inside of the output grid
        scale_factor *= point_count_crop / grid_points
        point_count_scaled = int(np.round(scale_factor * point_count_crop))
        # Convert the cropped data to a Pillow Image object
        data_out = Image.fromarray(data_out)
        # Resize to the larger interpolated data grid
        data_out = data_out.resize((point_count_scaled, point_count_scaled))
        # Crop out the points that correspond to the output
        point_count_scaled_half = point_count_scaled // 2
        lower = point_count_scaled_half - point_count_crop_half
        upper = point_count_scaled_half + point_count_crop_half
        data_out = data_out.crop([lower, lower, upper, upper])
        # The sum of all the points on the output grid, this will be used for
        # normalizing the final grid
        pre_sum = np.sum(np.array(data_out))
        # Resize to the actual number of points in the output
        data_out = data_out.resize((final_points, final_points), Image.NEAREST)
        # Convert back to a numpy array
        data_out = np.array(data_out)
        # The sum of all points with the new number of points
        post_sum = np.sum(data_out)
        # Normalize the values so that the sum of the points stays the same
        # after resizing to the final point count
        return data_out * pre_sum / post_sum
    # The output grid is bigger than the current grid
    else:
        # The number of points the current grid will take up on the output grid
        # rounded to the nearest even number
        resize_points = round(scale_factor * final_points / 2) * 2
        # Convert the cropped data to a Pillow Image object
        data_resized = Image.fromarray(data)
        # Resize to the new number of points for the output grid
        data_resized = data_resized.resize((resize_points, resize_points))
        # Create a 2D grid of all zeros for the output grid
        output_grid = np.zeros((final_points, final_points))
        # Find the middle of the grid to insert the resized original grid
        lower = int(final_points / 2) - int(resize_points / 2)
        upper = int(final_points / 2) + int(resize_points / 2)
        # Put the data at the middle of the output grid
        output_grid[lower:upper, lower:upper] = data_resized
        return output_grid


def resize_pixel_grid(data, final_pixels):
    # Convert the data to a Pillow Image object
    data_out = Image.fromarray(data)
    # Resize to the actual number of pixels in the output and convert to np
    return np.array(data_out.resize((final_pixels, final_pixels)))
