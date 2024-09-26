import numpy as np
from PIL import Image


def downsample_data(data, sampling, final_sampling, final_pixels):
    """
    Downsample the number of points in an array.

    Caution: this script is heavily opinionated and has the following
    assumptions and constraints:
        - The array must be a 2D square
        - The final grid size (final_sampling * final_pixels) must be smaller
          than the current grid size. Additionally, the number of pixels in the
          input data must be greater than the final_pixels.
        - This util is optimized for a final grid size that is much smaller
          than the current grid size, an intermediary performance step is done
          for this. This step may end up being slower for comparable initial and
          final grids.
        - The grid will be zoomed in towards the middle.
        - The final grid will be normalized so that the sum stays the same.
    """
    # Grab the number of grid points
    grid_points = data.shape[0]
    # Total grid size
    grid_size = sampling * grid_points
    # Total grid size of the output
    final_grid_size = final_sampling * final_pixels
    # How much bigger our current grid is than the output grid
    scale_factor = grid_size / final_grid_size
    # Instead of jumping right to performing the crops and resizings, we
    # perform an intermediary step with a crop first so that we do not
    # need to allocate so much memory (it is also faster).
    # Grab the number of points that cover the output grid with a tiny bit of
    # padding so that the full output grid is covered.
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
    # Resize to the actual number of pixels in the output
    data_out = data_out.resize((final_pixels, final_pixels), Image.NEAREST)
    # Convert back to a numpy array
    data_out = np.array(data_out)
    # The sum of all points with the new number of pixels
    post_sum = np.sum(data_out)
    # Normalize the values so that the sum of the points stays the same after
    # resizing to the final pixel count
    return data_out * pre_sum / post_sum


def resize_pixel_grid(data, final_pixels):
    # Convert the data to a Pillow Image object
    data_out = Image.fromarray(data)
    # Resize to the actual number of pixels in the output and convert to np
    return np.array(data_out.resize((final_pixels, final_pixels)))
