"""
This script was originally written quickly in my Downloads folder, so there are
no references to utils within this repo (everything needed was copied in).

Coral Edge TPUs utilize TFLite and could be an option to deploy the neural
networks in production. The biggest downside to them is that they require full
integer quantization in the neural network. That means, the weights,
activations, inputs, and outputs must all be 8-bit integers. Due to this, it
is important to see what the quanitzation for the inputs and outputs would look
like, and that is exactly what this script does.

It is expected that the dataset folder is placed in the same directory as this
script. Then, plots will be saved in the same folder as this script showing how
the quantization affects the input and output data.

More information on Coral requirements: coral.ai/docs/edgetpu/models-intro
"""

from glob import glob
from h5py import File
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

DATA_F = 'data.h5'
DS_TAG = 'random_50nm_large'
QUANT_POINTS = 255

# Instead of globbing the paths, it is safer to load in the datafiles using
# their chunk number so that they are guaranteed to be in order
chunk_vals = sorted([
    # Grab the number associated with each chunk
    int(path.split('/')[-1][:-len(DATA_F) - 1])
    # All datafiles should follow the format [chunk]_[DATA_F]
    for path in glob(f'{DS_TAG}/*_{DATA_F}')
])
input_data = []
output_data = []
print(f'Tag: {DS_TAG}')
for idx, chunk_val in enumerate(chunk_vals):
    path = f'{DS_TAG}/{chunk_val}_{DATA_F}'
    print(f'Path: {path}')
    data = File(path, 'r')
    # For our models, we will want to feed in our intensity fields and
    # output the associated Zernike polynomials
    input_data.extend(data['ccd_intensity'][:])
    output_data.extend(data['zernike_coeffs'][:])
    # This data will be the same across all chunks, so only read it once
input_data = np.array(input_data)
output_data = np.array(output_data)


def rounder(values):

    def f(x):
        idx = np.argmin(np.abs(values - x))
        return values[idx]

    return np.frompyfunc(f, 1, 1)


def quant_data(data):
    bins = np.linspace(np.min(data), np.max(data), QUANT_POINTS)
    bin_step_size = bins[1] - bins[0]
    quant = rounder(bins)(data).astype(float)
    return bin_step_size, quant


# Use only the first 10 rows since this takes a while
input_bin_step_size, quantized_input = quant_data(input_data[:10])
wavefront_diff = input_data[:10] - quantized_input
wavefront_errors = np.sum(np.abs(wavefront_diff), axis=(1, 2))

output_bin_step_size, quantized_output = quant_data(output_data)
output_bin_step_size_nm = output_bin_step_size * 1e9


def plot_wf_comp(idx):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    plt.suptitle(f'Idx {idx}, Sum of Absolute={wavefront_errors[idx]:0.3f}\n'
                 f'Bin Quantization={input_bin_step_size} m')
    ax[0].imshow(input_data[idx])
    ax[0].set_title('Original Wavefront')
    ax[1].imshow(quantized_input[idx])
    ax[1].set_title('Quantized Wavefront')
    im = ax[2].imshow(wavefront_diff[idx])
    ax[2].set_title('Difference')
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    fig.tight_layout()
    plot_path = f'./wf_comp_{idx}'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')


plot_wf_comp(1)
plot_wf_comp(3)
plot_wf_comp(5)


def plot_comparison_scatter_grid(regular_data, quantized_data, n_rows, n_cols):
    row_count, col_count = regular_data.shape
    subplot_args = {'figsize': (n_cols * 3, n_rows * 3)}
    fig, axs = plt.subplots(n_rows, n_cols, **subplot_args)
    plt.suptitle('Regular VS Quantized, '
                 f'Bin Quantization={output_bin_step_size_nm:0.3f} nm')
    current_col = 0
    for plot_row in range(n_rows):
        for plot_col in range(n_cols):
            if current_col >= col_count:
                fig.delaxes(axs[plot_row, plot_col])
                continue
            reg_col = regular_data[:, current_col]
            quant_col = quantized_data[:, current_col]
            axs_cell = axs[plot_row, plot_col]
            axs_cell.set_title(current_col)
            # Take the lowest and greatest values from both sets of data
            lower = min(np.amin(reg_col), np.amin(quant_col))
            upper = max(np.amax(reg_col), np.amax(quant_col))
            # Fix the bounds on both axes so they are 1-to-1
            axs_cell.set_xlim(lower, upper)
            axs_cell.set_ylim(lower, upper)
            # Draw a 1-to-1 line for the scatters
            # https://stackoverflow.com/a/60950862
            xpoints = ypoints = axs_cell.get_xlim()
            axs_cell.plot(
                xpoints,
                ypoints,
                linestyle='-',
                linewidth=2,
                color='#FFB200',
                scalex=False,
                scaley=False,
            )
            # Plot the scatter of all the points
            axs_cell.scatter(quant_col, reg_col, 0.25)
            current_col += 1
    for ax in axs.flat:
        ax.set(xlabel='Quantized Outputs', ylabel='Regular Outputs')
    fig.tight_layout()
    plot_path = './regular_vs_quantized_outputs.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')


plot_comparison_scatter_grid(output_data, quantized_output, 4, 6)
