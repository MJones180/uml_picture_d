import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.constants import PLOT_STYLE_FILE


def plot_2d(data, title, xlabel, ylabel, plot_path, add_colorbar=True):
    plt.style.use(PLOT_STYLE_FILE)
    plt.clf()
    plt.figure(figsize=(10, 10))
    image = plt.imshow(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if add_colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(image, cax=cax)
    plt.savefig(plot_path)
