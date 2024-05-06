# uml_picture_d

Michael Jones (Michael_Jones6@student.uml.edu)

## Installation

The following Conda command will create the environment and install the necessary dependencies:

    conda env create -f environment.yml 

Then, to activate the environment:

    conda activate picture_d

If the dependencies change at any point, the `environment.yml` can be updated via:

    conda env export --no-builds | grep -v "^prefix: " > environment.yml

A helpful cheatsheet with useful Conda commands: https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf

## Structure

    .
    ├── README.md
    ├── data
    │   ├── processed/
    │   └── raw_fits/
    ├── output
    │   ├── analysis/
    │   └── trained_models/
    └── src
        ├── main.py
        ├── script_parsers.py
        ├── networks/
        ├── scripts/
        └── utils/

## Data

Data is zipped and located in Google Drive at:
https://drive.google.com/drive/u/0/folders/1TA4R11xS-Lsn-TKVBSecxFMyboVPCshU

The unzipped data should be placed in a folder in the `/data/raw_fits/` directory.
The only thing the dataset should consist of are FITS files and the input CSV (`*input*.csv`).

Raw data is then processed by the `/src/scripts/preprocess_data.py` script and saved in the `/data/processed/` directory.
There are three separate datasets saved: training, validation, and testing.
Each processed dataset ends up consisting of two files:

- `data.h5`: all data is in the `inputs` and `outputs` tables
- `norm.json`: normalization values of the training dataset

## Model

All trained models will automatically be outputted in the `output/trained_models/` folder.
Each model is stored under the unique tag given to it during training.
Along with each epoch, models store the following files:

- `args.json`: the CLI args used to the model training script
- `norm.json`: normalization values used in the training dataset

## Networks

All networks (the structure of a given model) must be stored in the `src/networks` folder.
Each network must have the class name of `Network`.
Additionally, each class must have a static function named `example_input` which returns an example array which could be fed in to the network.

## Docstrings

Docstrings throughout the code are mostly formatted using `numpydoc` (https://numpydoc.readthedocs.io/en/latest/format.html).

## Future Updates

Please refer to the document located at https://docs.google.com/document/d/1EMN_9PPYlUP_mUWAjVyWF4GXGowUV1RvHwAa1ZzonKw/edit.
