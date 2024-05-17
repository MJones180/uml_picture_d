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

### PROPER

Proper in Python 3 must also be installed.
For this, install PROPER from https://proper-library.sourceforge.net/ and store the unzipped `proper` direcotry under `packages`.
This will result in `packages/proper/` containing the necessary Python files.

## Structure

    .
    ├── environment.yml
    ├── README.md
    ├── PROPER_manual_v3.2.7.pdf
    ├── data
    │   ├── processed/
    │   ├── proper_sim/
    │   └── raw_fits/
    ├── diagram_reference/
    ├── output
    │   ├── analysis/
    │   └── trained_models/
    ├── packages
    │   ├── cbm_vvc_mft.py
    │   └── proper/
    └── src
        ├── main.py
        ├── script_parsers.py
        ├── networks/
        ├── scripts/
        └── utils/

## Data

### Pre-existing Data

Data is zipped and located in Google Drive at:
https://drive.google.com/drive/u/0/folders/1TA4R11xS-Lsn-TKVBSecxFMyboVPCshU

The unzipped data should be placed in a folder in the `/data/raw_fits/` directory.
The only thing the dataset should consist of are FITS files and the input CSV (`*input*.csv`).

Raw data is then processed by the `/src/scripts/preprocess_data.py` script and saved in the `/data/processed/` directory.
There are three separate datasets saved: training, validation, and testing.
Each processed dataset ends up consisting of two files:

- `data.h5`: all data is in the `inputs` and `outputs` tables
- `norm.json`: normalization values of the training dataset

### Simulate Data

Data can now be simulated in this repo using PROPER in Python 3.

For information on the optical train files, look at the Optical Trains section.

## Model

All trained models will automatically be outputted in the `output/trained_models/` folder.
Each model is stored under the unique tag given to it during training.
Along with each epoch, models store the following files:

- `args.json`: the CLI args used to the model training script
- `norm.json`: normalization values used in the training dataset

To easily lookup a model by its tag, there exists a JSON file at `output/tag_lookup.json` that can be referenced.

## Networks

All networks (the structure of a given model) must be stored in the `src/networks` folder.
Each network must have the class name of `Network`.
Additionally, each class must have a static function named `example_input` which returns an example array which could be fed in to the network.

## Optical Trains

All optical trains (the setup for a simulation) must be stored in the `src/sim_optical_trains` folder.
Each optical train must have the following variables:
- `INIT_BEAM_D`: Diameter of the initial beam in meters.
- `BEAM_RATIO`: Ratio of space that the beam takes up on the grid.
- `OPTICAL_TRAIN`: A list specifying the steps of the train. Each `proper` call must be wrapped in a lambda that takes the `wf` oject. Additionally, a nested list can be passed if that step should have the option to be plotted. An example list would be `[ lambda wf: proper.prop_circular_aperture(wf, 1), [ 'Entrance', lambda wf: proper.prop_define_entrance(wf) ] ]`.

## Optical Setup

The `diagram_reference` directory contains diagrams that can be referenced about the optical setup.

The following terminology is used:
- OAP: Off-axis parabola
- OAE: Off-axis ellipse
- OAH: Off-axis hyperbola
- M1: Primary mirror (parabola for PICTURE-D)
- M2: Secondary mirror (ellipse for PICTURE-D)
- M3: OAP

## Docstrings

Docstrings throughout the code are mostly formatted using `numpydoc` (https://numpydoc.readthedocs.io/en/latest/format.html).

## Future Updates

Please refer to the document located at https://docs.google.com/document/d/1EMN_9PPYlUP_mUWAjVyWF4GXGowUV1RvHwAa1ZzonKw/edit.
