"""
# Code by Michael Jones (Michael_Jones6@student.uml.edu).
#
# Will take a folder that has been uncompressed from the Google Drive link at
# https://drive.google.com/drive/u/0/folders/1TA4R11xS-Lsn-TKVBSecxFMyboVPCshU
# (LLOWFS Data).
#
# Right now, to save time, all constants and arguments have been hardcoded.
"""
from fits2hdf.io import fitsio, hdfio
from glob import glob
from h5py import File
import json
import numpy as np
from os import sep
from pathlib import Path
from shutil import copyfile

IN_DIR = 'raw_data/picture_c_llowfs_data_05_training'
# OUT_DIR = 'processed_data/training_data_global_out_norm'
# MIN_MAX_NORM = {
#     'global_input_norm': True,
#     'global_output_norm': True,
#     'individual_output_norm': False,
# }
# OUT_DIR = 'processed_data/training_data_individual_out_norm'
# MIN_MAX_NORM = {
#     'global_input_norm': True,
#     'global_output_norm': False,
#     'individual_output_norm': True,
# }
OUT_DIR = 'processed_data/training_data_no_out_norm'
MIN_MAX_NORM = {
    'global_input_norm': True,
    'global_output_norm': False,
    'individual_output_norm': False,
}
TRAINING_DATA = True

IN_DIR = 'raw_data/picture_c_llowfs_data_03_testing'
OUT_DIR = 'processed_data/testing_data'
MIN_MAX_NORM = {
    'global_input_norm': True,
    'global_output_norm': False,
    'individual_output_norm': False,
}
TRAINING_DATA = False

# =============================================================================
# Create the new directories
# =============================================================================

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
Path(f'{OUT_DIR}{sep}inputs').mkdir(parents=True, exist_ok=True)

# =============================================================================
# Copy over the outputs CSV and convert the FITS to HDF
# =============================================================================

copyfile(glob(f'{IN_DIR}{sep}*.csv')[0], f'{OUT_DIR}{sep}output.csv')

fits_paths = sorted(glob(f'{IN_DIR}{sep}*.fits'))
for idx, input_image_path in enumerate(fits_paths):
    input_filename = input_image_path.split(sep)[-1]
    print(f'{input_filename} ({idx + 1} / {len(fits_paths)})')
    fits_file = fitsio.read_fits(input_image_path)
    hdfio.export_hdf(fits_file, f'{OUT_DIR}{sep}inputs{sep}{idx}.hdf')

# =============================================================================
# Create a merged HDF file with all the inputs and outputs together
# =============================================================================

input_data = []
hdf_paths = sorted(glob(f'{OUT_DIR}{sep}inputs{sep}*.hdf'))
for idx, hdf_path in enumerate(hdf_paths):
    print(f'{hdf_path}, ({idx + 1} / {len(hdf_paths)})')
    input_data.append(File(hdf_path, 'r')['PRIMARY']['DATA'][:])
input_data = np.array(input_data)
# Remove the first column and first row from the input images since all the
# pixels in those locations are all zero
input_data = input_data[:, 1:, 1:]

output_data = np.genfromtxt(f'{OUT_DIR}{sep}output.csv', delimiter=',')
# Chop off the ID column
output_data = output_data[:, 1:]


# Min-max normalize the data
def _min_max_norm(data, globally=True):
    if globally:
        min_x = np.min(data)
        max_min_diff = np.max(data) - min_x
    else:
        min_x = np.min(data, axis=0)
        max_min_diff = np.max(data, axis=0) - min_x
    # (data - minX) / (maxX - minX)
    # (data - minX) / maxMinDiff
    normed = (data - min_x) / max_min_diff
    return min_x, max_min_diff, normed


# Now this code for doing the min-max norm isn't great, but it's quick and will
# get the job done for now
norm_values = {}
if MIN_MAX_NORM['global_input_norm']:
    min_x, max_min_diff, normed = _min_max_norm(input_data)
    norm_values['input_min_x'] = min_x
    norm_values['input_max_min_diff'] = max_min_diff
    input_data = normed
if MIN_MAX_NORM['global_output_norm']:
    min_x, max_min_diff, normed = _min_max_norm(output_data)
    # For the output normalization, it is easier if there is a norm value for
    # every single element
    min_x = np.repeat(min_x, output_data.shape[1])
    max_min_diff = np.repeat(max_min_diff, output_data.shape[1])
    norm_values['output_min_x'] = min_x
    norm_values['output_max_min_diff'] = max_min_diff
    output_data = normed
if MIN_MAX_NORM['individual_output_norm']:
    min_x, max_min_diff, normed = _min_max_norm(output_data, globally=False)
    norm_values['output_min_x'] = min_x
    norm_values['output_max_min_diff'] = max_min_diff
    output_data = normed
with open(f'{OUT_DIR}{sep}normalization.json', 'w') as norm_out:

    # https://stackoverflow.com/a/47626762
    class NumpyEncoder(json.JSONEncoder):

        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    json.dump(
        norm_values,
        norm_out,
        ensure_ascii=False,
        indent=4,
        cls=NumpyEncoder,
    )

# Operations that should only be performed on the training data
if TRAINING_DATA:
    # Now this code is a little bit silly, but since this is an unoffical test
    # before I create the offfical framework, let's role with it. Instead of
    # using TensorFlow Keras based image augmentations, we will just create
    # three copies of every training row where the input image is: flipped
    # horizontally, flipped vertically, and flipped both horizontally and
    # vertically. The benefits of doing this is that we artifically obtain
    # more data and help prevent against some overfitting. All and all, this
    # will create 4x the amount of data.
    input_data = np.concatenate((
        input_data,
        input_data[:, :, ::-1],
        input_data[:, ::-1],
        input_data[:, ::-1, ::-1],
    ))
    output_data = np.tile(output_data, (4, 1))

    # Shuffle all the data, this does not need to be done for testing
    random_shuffle_idxs = np.random.permutation(len(input_data))
    input_data = input_data[random_shuffle_idxs]
    output_data = output_data[random_shuffle_idxs]

# Write out the merged and preprocessed data
with File(f'{OUT_DIR}{sep}merged.h5', 'w') as out_hdf_file:
    out_hdf_file['inputs'] = input_data
    out_hdf_file['outputs'] = output_data
