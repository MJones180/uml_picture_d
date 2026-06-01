"""
Plot the learning rate curve from a trained model.
"""

import matplotlib.pyplot as plt
import numpy as np

LR_PATH = '../output/trained_models/dh_v122_25/learning_rates.csv'

# Load in the style file
plt.style.use('plot_styling.mplstyle')

data = np.loadtxt(LR_PATH, delimiter=',', skiprows=1)

fig, ax = plt.subplots(figsize=(12, 6))


def _add_vline(epoch):
    ax.axvline(x=epoch, linestyle='--', linewidth=1.5, alpha=0.5)
    ax.text(
        x=epoch + 10,
        y=0.02,
        transform=ax.get_xaxis_transform(),
        s=str(epoch),
        va='center',
        fontsize=14,
    )


def _add_hline(lr):
    ax.axhline(y=lr, linestyle='--', linewidth=1.5, alpha=0.5)
    ax.text(
        x=0.002,
        y=lr + 0.00025,
        transform=ax.get_yaxis_transform(),
        s=str(lr),
        fontsize=14,
    )


_add_vline(80)
_add_vline(680)
_add_hline(3e-3)
ax.plot(np.arange(3500) + 1, data[:, 1], linewidth=3)

x_vals = data[:, 0]
x_tick_locs = [x_vals[0], *x_vals[499::500]]
x_tick_vals = [str(int(val)) for val in x_tick_locs]
ax.set_xticks(x_tick_locs)
ax.set_xticklabels(x_tick_vals)

ax.set_yscale('log')
ax.grid(True)
ax.set_xlabel('Epoch')
ax.set_ylabel('Learning Rate')
ax.set_title('Learning Rate Schedule')
plt.savefig('learning_rate.png')
