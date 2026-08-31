from h5py import File
import numpy as np

data = File('../../data/raw/pol_41728_phase_and_amp_log10_int_diff/0_data.h5')
phase = data['phase'][:]
# Grab only the active pixels
phase = phase[phase != 0]
# Compute the RMS of the phase
phase = np.std(phase)
# Convert from radians to nm
radians_to_nm = 600 / (2 * np.pi)
# Compute the RMS WFE in nm
phase *= radians_to_nm
print(f'RMS WFE (nm): {phase}')
