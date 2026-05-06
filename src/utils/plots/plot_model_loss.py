import matplotlib.pyplot as plt
from utils.constants import PLOT_STYLE_FILE


def plot_model_loss(data, loss_function, plot_path):
    # Load in the style file
    plt.style.use(PLOT_STYLE_FILE)
    # Reset the plot
    plt.clf()
    fig, ax = plt.subplots()
    ax.set_title('Epoch Loss During Training')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'Loss ({loss_function})')
    epochs = data[:, 0]
    ax.plot(epochs, data[:, 1], 'b', label='Train Loss')
    ax.plot(epochs, data[:, 2], 'r', label='Validation Loss')
    plt.legend()
    plt.savefig(plot_path)
