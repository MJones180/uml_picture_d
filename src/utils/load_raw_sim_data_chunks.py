from glob import glob
import numpy as np
from utils.constants import (ABERRATIONS_F, CAMERA_INTENSITY, CAMERA_SAMPLING,
                             DATA_F, FULL_INTENSITY, FULL_SAMPLING,
                             RAW_SIMULATED_DATA_P, ZERNIKE_COEFFS,
                             ZERNIKE_TERMS)
from utils.hdf_read_and_write import read_hdf
from utils.printing_and_logging import dec_print_indent, inc_print_indent


def load_raw_sim_data_chunks(raw_data_tag, full_intensity=False):
    base_path = f'{RAW_SIMULATED_DATA_P}/{raw_data_tag}'
    # Instead of globbing the paths, it is safer to load in the datafiles using
    # their chunk number so that they are guaranteed to be in order
    chunk_vals = sorted([
        # Grab the number associated with each chunk
        int(path.split('/')[-1][:-len(DATA_F) - 1])
        # All datafiles should follow the format [chunk]_[DATA_F]
        for path in glob(f'{base_path}/*_{DATA_F}')
    ])
    input_data = []
    output_data = []
    intensity_tag = FULL_INTENSITY if full_intensity else CAMERA_INTENSITY
    sampling_tag = FULL_SAMPLING if full_intensity else CAMERA_SAMPLING
    print(f'Tag: {raw_data_tag}')
    inc_print_indent()
    for idx, chunk_val in enumerate(chunk_vals):
        path = f'{base_path}/{chunk_val}_{DATA_F}'
        print(f'Path: {path}')
        data = read_hdf(path)
        # For our models, we will want to feed in our intensity fields and
        # output the associated Zernike polynomials
        input_data.extend(data[intensity_tag][:])
        output_data.extend(data[ZERNIKE_COEFFS][:])
        # This data will be the same across all chunks, so only read it once
        if idx == 0:
            zernike_terms = data[ZERNIKE_TERMS][:]
            sampling = data[sampling_tag][()]
    dec_print_indent()
    input_data = np.array(input_data)
    output_data = np.array(output_data)
    return input_data, output_data, zernike_terms, sampling


def load_raw_sim_data_aberrations_file(raw_data_tag):
    file_path = f'{RAW_SIMULATED_DATA_P}/{raw_data_tag}/{ABERRATIONS_F}'
    # Load in the aberration coefficients
    aberrations = np.loadtxt(file_path, delimiter=',')
    # Grab the Zernike terms that correspond to each coefficient
    with open(file_path) as file:
        header_line = file.readline()[1:-1].split(',')
    zernike_terms = [int(term) for term in header_line]
    return aberrations, zernike_terms
