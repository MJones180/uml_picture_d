"""
This script will align the modes between two sets. Originally, this code was
written to align the modes between a single wavelength jacobian and a broadband
jacobian... the modes look very similar but are in a different order.
"""

from astropy.io import fits
import numpy as np


def align_modes(set1, set2):
    num_modes, num_pixels = set1.shape

    def norm_set(modes):
        # Divide by the Euclidean norm (root sum of squares)
        return modes / (np.sum(modes**2, axis=1)**0.5)[:, None]

    # Normalize the modes to have a length of 1
    # `set1_norm`/`set2_norm` shape: (num_modes, num_pixels)
    set1_norm = norm_set(set1)
    set2_norm = norm_set(set2)

    # Create a similarity matrix which gives how alike any two modes are
    # The similarity is a Cosine Similarity: (u dot v) / (|u| |v|)
    # `similarity` shape: (num_modes, num_modes)
    similarity = set1_norm @ set2_norm.T

    # The similarity values range from [-1, 1], make them range from [0, 1]
    similarity = np.abs(similarity)

    # Gives the indicies of the corresponding modes in `set2`
    # `set_alignment_mapping` shape: (num_modes)
    set_alignment_mapping = np.zeros(num_modes, dtype=int)
    # Loop through all the modes and pick out the best aligning ones each iter
    for _ in range(num_modes):
        # Returns the flattened index of the modes with the higest similarity
        best_match_idx = np.argmax(similarity)
        # The indices of the modes that best align between the two sets
        set1_mode_idx = best_match_idx // num_modes
        set2_mode_idx = best_match_idx % num_modes
        # Store the match
        set_alignment_mapping[set1_mode_idx] = set2_mode_idx
        # Mask out the row/col to ensure a one-to-one mapping
        similarity[set1_mode_idx, :] = -1
        similarity[:, set2_mode_idx] = -1

    # Put `set2` in the order of `set1`
    set2_aligned = set2[set_alignment_mapping]

    # Take the dot product between each of the aligned modes
    dot_products = np.sum(set1 * set2_aligned, axis=1)
    # In order to correct any sign mismatches, the dot product can be used:
    # a negative value means that the modes have opposite signs; the modes in
    # `set2_aligned` should be multiplied by this array to have the correct sign
    # `set2_aligned_sign_fix` shape: (num_modes, 1)
    set2_aligned_sign_fix = np.sign(dot_products)[:, None]

    # Apply the sign fix
    set2_final = set2_aligned * set2_aligned_sign_fix

    return set2_final, set_alignment_mapping, set2_aligned_sign_fix


# === EXAMPLE ===
# Data must have the shape (modes, pixels)
single_wl_ef_modes = fits.getdata('single_wl/dm1_u_matrix.fits')
broadband_ef_modes = fits.getdata('broadband/dm1_u_matrix.fits')
broadband_ef_modes_aligned, idx_mapping, sign_fix = align_modes(
    single_wl_ef_modes,
    broadband_ef_modes,
)
