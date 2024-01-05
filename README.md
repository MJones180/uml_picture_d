# uml_picture_d

## Installation

The following Conda command will create the environment and install the necessary dependencies:

    conda env create -f environment.yml 

Then, to activate the environment:

    conda activate picture_d

If the dependencies change at any point, the `environment.yml` can be updated via:

    conda env export --no-builds | grep -v "^prefix: " > environment.yml

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
