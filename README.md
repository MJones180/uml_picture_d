# uml_picture_d

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
    │   └── raw/
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

## Model

All trained models will automatically be outputted in the `output/trained_models/` folder.
Each model is stored under the unique tag given to it during training.
The JSON file located at `output/trained_models/tag_lookup.json` will be created/updated with the hyperparameters associated with the model's tag after training has started.
Along with each epoch, models store the following files:

- `norm.json`: normalization values used in the training dataset
- `hyperparameters.json`: the hyperparameters used to train the model

## Networks

All networks (the structure of a given model) must be stored in the `src/networks` folder.
Each network must have the class name of `Network`.
Additionally, each class must have a static function named `example_input` which returns an example array which could be fed in to the network.

## Docstrings

Docstrings throughout the code are mostly formatted using `numpydoc` (https://numpydoc.readthedocs.io/en/latest/format.html).
