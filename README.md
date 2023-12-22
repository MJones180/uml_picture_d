# uml_picture_d

## Installation

Dependencies can be installed using any package manager that you prefer.
I personally am using Conda, but Python3 `venv` also works fine.
The necessary packages are: `astropy`, `h5py`, `numpy`, `pytorch`, and `torchvision`.
Once these packages have been installed, `fits2hdf` must be cloned and installed manually (tutorial at https://fits2hdf.readthedocs.io/en/latest/getting_started.html#installation).

## Structure

.
├── README.md
├── data
│   ├── processed/
│   └── raw/
├── external_packages/
├── output
│   ├── analysis/
│   └── trained_models/
└── src
    ├── main.py
    ├── scripts/
    └── utils/

## Data

Data is zipped and located in Google Drive at:
https://drive.google.com/drive/u/0/folders/1TA4R11xS-Lsn-TKVBSecxFMyboVPCshU
