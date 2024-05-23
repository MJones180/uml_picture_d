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

## Data

### Simulate Data

Data can be simulated in this repo by using PROPER in Python 3.
For all simulations use the `sim_data` script.

A simulated dataset will be outputted at `data/raw_simulated/[tag]`, where `tag` is the name of the dataset.
The following files will be saved:

- `args.json`: the arguments passed to the script to simulate the data
- `data.h5`: all the data generated during the simulation (includes the intensity fields and Zernike aberrations)
- `plots`: optional plots of the wavefront as it passes through the optical train

### Preprocess Data

To train and test a model with the simulated data, it will first need to be preprocessed.
After running `sim_data.py`, the preprocessing script `preprocess_zernike_data.py` should be ran.
The preprocessed data is then outputted in `data/processed/` under three separate folders for training, validation, and testing.

Each processed dataset ends up consisting of four files:

- `args.json`: the arguments passed to the script to preprocess the data
- `data.h5`: all data is in the `inputs` and `outputs` tables
- `ds_raw_info.json`: unused tables from the raw dataset that may be helpful
- `norm.json`: normalization values of the training dataset

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
- `CCD_PIXELS`: Number of pixels on the output grid that represents the CCD.
- `CCD_SAMPLING`: The sampling for each pixel (grid point) on the CCD.

## Optical Setup

The `diagram_reference` directory contains diagrams that can be referenced about the optical setup.

The following terminology is used:
- OAP: Off-axis parabola
- OAE: Off-axis ellipse
- OAH: Off-axis hyperbola
- M1: Primary mirror (parabola for PICTURE-D)
- M2: Secondary mirror (ellipse for PICTURE-D)
- M3: OAP

## Zernike Polynomials

The Zernike Polynomials can be described via `n` and `m` ($`Z_n^m`$).
More information on this can be found at en.wikipedia.org/wiki/Zernike_polynomials.
For this repo, Zernike terms will be described by which number they are (given by Noll's indices).
A table of the first 22 is given below (names taken from page 200 of `PROPER_manual_v3.2.7.pdf`):

| Noll Number | `n`   | `m`   | Name                         |
| :---:       | :---: | :---: | :---:                        |
| 1           | 0     | 0     | Piston                       |
| 2           | 1     | 1     | X tilt (horizontal)          |
| 3           | 1     | -1    | Y tilt (vertical)            |
| 4           | 2     | 0     | Focus                        |
| 5           | 2     | -2    | 45 deg astigmatism           |
| 6           | 2     | 2     | 0  deg astigmatism           |
| 7           | 3     | -1    | Y coma                       |
| 8           | 3     | 1     | X coma                       |
| 9           | 3     | -3    | Y clover (trefoil)           |
| 10          | 3     | 3     | X clover (trefoil)           |
| 11          | 4     | 0     | 3rd order spherical          |
| 12          | 4     | 2     | 5th order 0  deg astigmatism |
| 13          | 4     | -2    | 5th order 45 deg astigmatism |
| 14          | 4     | 4     | X quadrafoil                 |
| 15          | 4     | -4    | Y quadrafoil                 |
| 16          | 5     | 1     | 5th order X coma             |
| 17          | 5     | -1    | 5th order Y coma             |
| 18          | 5     | 3     | 5th order X clover           |
| 19          | 5     | -3    | 5th order Y clover           |
| 20          | 5     | 5     | X pentafoil                  |
| 21          | 5     | -5    | Y pentafoil                  |
| 22          | 6     | 0     | 5th order spherical          |

## Docstrings

Docstrings throughout the code are mostly formatted using `numpydoc` (https://numpydoc.readthedocs.io/en/latest/format.html).

## Future Updates

Please refer to the document located at https://docs.google.com/document/d/1EMN_9PPYlUP_mUWAjVyWF4GXGowUV1RvHwAa1ZzonKw/edit.
