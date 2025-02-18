# uml_picture_d

Michael Jones (Michael_Jones6@student.uml.edu)

Code was originally written for the reflective Lyot-stop low order wavefront sensor (LLOWFS) neural network (NN) on PICTURE-D.
There is plenty of other code written for PICTURE-D, but this repository contains just the code that I have written.
Since this repository was originally created for the LLOWFS NN, a lot of the code has been tailored towards it (such as networks and optical trains).
However, this repository can be used to simulate arbitrary optical train data, develop neural network models, and run simulated control loops.
Therefore, if scripts do not clearly state what project they are for, it is safe to assume they are for the LLOWFS NN.

## Installation

The following Conda command will create the environment and install the necessary dependencies:

    # CPU only
    conda env create -f environment.yml 

    # GPU [NVIDIA CUDA]
    conda env create -f environment_cuda.yml 

Then, to activate the environment:

    conda activate picture_d

If the dependencies change at any point, the environment file can be updated via:

    # CPU env
    conda env export --no-builds | grep -v "^prefix: " > environment.yml

    # GPU env
    conda env export --no-builds | grep -v "^prefix: " > environment_cuda.yml

A helpful cheatsheet with useful Conda commands: docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf

### Notes

- [Bug] When installing on Ubuntu 22.04, there is sometimes a bug with `pyFFTW` that prevents it from running.
When this happens, uninstall `pyFFTW` from Conda and install with pip:

        conda uninstall pyfftw
        pip install pyfftw

- [Bug] The error below may arise when calling scripts that use the `pathos` library (such as `sim_data`) and can be fixed by running `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` or by using `main_scnp.py` instead of `main.py`.

        objc[...]: +[__NSCFConstantString initialize] may have been in progress in another thread when fork() was called.
        objc[...]: +[__NSCFConstantString initialize] may have been in progress in another thread when fork() was called. We cannot safely call it or ignore it in the fork() child process. Crashing instead. Set a breakpoint on objc_initializeAfterForkError to debug.

- Some environments have trouble with the correct version of the `libgfortran` dependency. When this happens, the dependency can be manually updated to `libgfortran>=3.0.0`.

- The `PyTorch` environment with GPU support is configured with CUDA 12.4.
    - The NVIDIA Display Driver 550 must be download separately (nvidia.com/download/driverResults.aspx/230357/en-us/).
    - The CUDA Toolkit 12.4 does not need to be installed manually as `PyTorch` installs a CUDA runtime for itself.

- If the environment files are not working to properly install things, then a new Conda environment can be created from scratch:

        # Create and activate the new Conda env
        conda create --name picture_d
        conda activate picture_d
        # Install the dependencies
        conda install astropy h5py hdf5 matplotlib numpy pathos pillow prettytable pyfftw scipy
        pip install mpl-scatter-density onnx onnxscript onnxruntime
        # Grab command to install PyTorch dependencies from pytorch.org/get-started/locally/

### PROPER

Proper in Python 3 must also be installed.
For this, install PROPER from https://proper-library.sourceforge.net/ and store the unzipped `proper` directory under `packages`.
This will result in `packages/proper/` containing the necessary Python files.

**Important speed notes**:
- The PROPER package in Python makes two calls to `gc.collect()`, but this is unnecessary.
The following two lines should be commented out for faster simulation speeds:
    - Line 90 of `prop_propagate.py`
    - Line 166 of `prop_lens.py`
- When making calls to the VVC in an optical train, the `cbm_vvc_approx` function is much faster than the `cbm_vvc_mft` function.
The `cbm_vvc_approx` does not do MFT, but the propagated wavefront ends up being about the same.

## Calling Scripts

Navigate to the `src` folder and call the `main.py` file, all scripts are available as sub-commands.
In the event that one of the scripts requires NumPy on a single core, then call `main_scnp.py` instead.
For example, due to Pathos multiprocessing, the `sim_data` script should be called via `main_scnp.py`.

Alternatively, if a script should only be run using `n` cores, then the `taskset` command can be used in Linux.
As an example, to use only three cores:

    taskset --cpu-list 1,2,3 python3 main.py

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
- `extra_variables.h5`: variables that may be required later (such as normalization)

## Model

All trained models will automatically be outputted in the `output/trained_models/` folder.
Each model is stored under the unique tag given to it during training.
Along with each epoch, models store the following files:

- `args.json`: the CLI args used to the model training script
- `extra_variables.h5`: variables that may be required later (such as normalization)

To easily lookup a model by its tag, there exists a JSON file at `output/tag_lookup.json` that can be referenced.

## Networks

All networks (the structure of a given model) must be stored in the `src/networks` folder.
Each network must have the class name of `Network`.
Additionally, each class must have a static function named `example_input` which returns an example array which can be fed in to the network.

Old networks that are not being used anymore are placed in an archive folder located at `src/networks/archived`.
To use these networks, they must first be moved back to the root of the `src/networks` folder.

## Optical Trains

All optical trains (the setup for a simulation) must be stored in the `src/sim_optical_trains` folder.
Each optical train must have the following variables:
- `INIT_BEAM_D`: Diameter of the initial beam in meters.
- `BEAM_RATIO`: Ratio of space that the beam takes up on the grid.
- `OPTICAL_TRAIN`: A list specifying the steps of the train. Each `proper` call must be wrapped in a lambda that takes the `wf` oject. Additionally, a nested list can be passed if that step should have the option to be plotted. An example list would be `[ lambda wf: proper.prop_circular_aperture(wf, 1), [ 'Entrance', lambda wf: proper.prop_define_entrance(wf) ] ]`.
- `CAMERA_PIXELS`: Number of pixels on the output grid that represents the camera.
- `CAMERA_SAMPLING`: The sampling for each pixel (grid point) on the camera.

## Optical Setup

The `diagrams` directory contains diagrams that can be referenced about the optical setup.
For example, `diagrams/optical_design_v84.png` contains the optical setup for V84.

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
A table of the first 24 is given below ("Manual Names" taken from page 200 of `PROPER_manual_v3.2.7.pdf`):

Noll Number | `n`   | `m`   | Manual Names                 | Shorthand Names
:---:       | :---: | :---: | :---:                        | :---:
1           | 0     | 0     | Piston                       | 
2           | 1     | 1     | X tilt (horizontal)          | Tilt X
3           | 1     | -1    | Y tilt (vertical)            | Tilt Y
4           | 2     | 0     | Focus                        | Power
5           | 2     | -2    | 45 deg astigmatism           | Astig 1
6           | 2     | 2     | 0  deg astigmatism           | Astig 2
7           | 3     | -1    | Y coma                       | Coma 1
8           | 3     | 1     | X coma                       | Coma 2
9           | 3     | -3    | Y clover (trefoil)           | Trefoil 1
10          | 3     | 3     | X clover (trefoil)           | Trefoil 2
11          | 4     | 0     | 3rd order spherical          | Spherical
12          | 4     | 2     | 5th order 0  deg astigmatism | 2nd Astig 1
13          | 4     | -2    | 5th order 45 deg astigmatism | 2nd Astrig 2
14          | 4     | 4     | X quadrafoil                 | Tetrafoil 1
15          | 4     | -4    | Y quadrafoil                 | Tetrafoil 2
16          | 5     | 1     | 5th order X coma             | 2nd Coma 1
17          | 5     | -1    | 5th order Y coma             | 2nd Coma 2
18          | 5     | 3     | 5th order X clover           | 2nd Trefoil 1
19          | 5     | -3    | 5th order Y clover           | 2nd Trefoil 2
20          | 5     | 5     | X pentafoil                  | Pentafoil 1
21          | 5     | -5    | Y pentafoil                  | Pentafoil 2
22          | 6     | 0     | 5th order spherical          | 2nd Spherical
23          | 6     | -2    |                              | 3rd Astig 1
24          | 6     | 2     |                              | 3rd Astig 2

The coefficient for each Zernike term represents the RMS wavefront error associated with that temr.

## Docstrings

Docstrings throughout the code are mostly formatted using `numpydoc` (https://numpydoc.readthedocs.io/en/latest/format.html).

## Notes

- When this repo started, the camera being used was a CCD.
  Therefore, all of the variables originally referred to the output camera as being a CCD.
  In order to be more general, all references to CCD were renamed to `camera`.
  However, to maintain backwards compatability, some constants still have a value that contains `ccd` instead of `camera`.
- The `src/single_core_numpy.py` code (imported by `main_scnp.py`) sets the max number of threads.
  However, this is really setting the max number of cores.
  On a linux system, if `lscpu` is typed, then the total number of CPUs is given by `cores per socket * threads per core`.
  The total number of CPUs is also referred to as the number of processors (this number can also be obtained by running `nproc`).
  If the flags in `src/single_core_numpy.py` are not defined, then they each default to the total number of processors.
  For a CPU with 16 cores and 2 threads per core, this would mean the flags are each set to 32 by default.

## Future Updates

Please refer to the document located at https://docs.google.com/document/d/1EMN_9PPYlUP_mUWAjVyWF4GXGowUV1RvHwAa1ZzonKw/edit.
