"""
-- NOTE ------------------------------------------------------------------------
This file can be used to project a wavefront consisting of unpropagated Zernike
terms onto orthogonal Zernike basis terms.
--------------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import proper

REF_WL = 600e-9
INIT_BEAM_D = 9e-3
BEAM_RATIO = 1
GRID_POINTS = 1024

zernike_terms = np.arange(2, 8)
zernike_count = len(zernike_terms)

wavefront = proper.prop_begin(INIT_BEAM_D, REF_WL, GRID_POINTS, BEAM_RATIO)
proper.prop_circular_aperture(wavefront, INIT_BEAM_D / 2)
proper.prop_define_entrance(wavefront)
basis_terms = [
    proper.prop_zernikes(wavefront, [term], [1]) for term in zernike_terms
]

# Chop off corners to make a circle
valid_mask = proper.prop_get_amplitude(wavefront)**2 > 0
basis_terms = basis_terms * valid_mask

# Flatten out the pixels
basis_terms = np.reshape(basis_terms, (basis_terms.shape[0], -1))

truth = np.zeros(zernike_count)
# ==============================================================================
# MUST EITHER CREATE A NEW WAVEFRONT
# chosen_wf = proper.prop_zernikes(wavefront, [7], [2])
# truth[7 - 2] = 1
chosen_wf = proper.prop_zernikes(wavefront, [5, 7], [2, 2])
truth[5 - 2] = 1
truth[7 - 2] = 1

chosen_wf = chosen_wf * valid_mask
chosen_wf = np.reshape(chosen_wf, -1)

# OR USE A BASIS TERM
# WF_TERM_IDX_TO_USE = 5
# chosen_wf = basis_terms[WF_TERM_IDX_TO_USE]
# truth[WF_TERM_IDX_TO_USE] = 1
# ==============================================================================

# Need to normalize the bin area
bin_area = (2 / GRID_POINTS)**2 / np.pi

for term in basis_terms:
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(chosen_wf.reshape(GRID_POINTS, -1))
    ax[1].imshow(term.reshape(GRID_POINTS, -1))
    mult = term * chosen_wf
    ax[2].imshow(mult.reshape(GRID_POINTS, -1))
    ax[2].set_title(np.sum(mult) * bin_area)
    plt.show()

crosstalk_between_terms = np.array(
    [basis_terms @ term for term in basis_terms]) * bin_area
print(crosstalk_between_terms)

coeffs = (basis_terms @ chosen_wf) * bin_area
print(coeffs)
print(truth)

plt.scatter(zernike_terms, truth, color='green', label='truth')
plt.scatter(zernike_terms, coeffs, color='red', label='obtained')
plt.show()

plt.imshow(chosen_wf.reshape(GRID_POINTS, -1))
plt.colorbar()
plt.show()
recovered_wavefront = basis_terms.T @ coeffs
plt.imshow(recovered_wavefront.reshape(GRID_POINTS, -1))
plt.colorbar()
plt.show()
