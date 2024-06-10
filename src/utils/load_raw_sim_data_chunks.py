from glob import glob
import numpy as np
from utils.constants import (CCD_INTENSITY, CCD_SAMPLING, DATA_F,
                             RAW_SIMULATED_DATA_P, ZERNIKE_COEFFS,
                             ZERNIKE_TERMS)
from utils.hdf_read_and_write import read_hdf


def load_raw_sim_data_chunks(raw_data_tag):
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
    for idx, chunk_val in enumerate(chunk_vals):
        path = f'{base_path}/{chunk_val}_{DATA_F}'
        print(f'Path: {path}')
        data = read_hdf(path)
        # For our models, we will want to feed in our intensity fields and
        # output the associated Zernike polynomials
        input_data.extend(data[CCD_INTENSITY][:])
        output_data.extend(data[ZERNIKE_COEFFS][:])
        # This data will be the same across all chunks, so only read it once
        if idx == 0:
            # Other data that will be written out for reference
            ccd_sampling = data[CCD_SAMPLING][()]
            zernike_terms = data[ZERNIKE_TERMS][:]
    input_data = np.array(input_data)
    output_data = np.array(output_data)
    return input_data, output_data, zernike_terms, ccd_sampling
