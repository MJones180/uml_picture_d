import matplotlib.pyplot as plt
from scipy.stats import gennorm, qmc

plt.style.use('plot_styling.mplstyle')
fig, ax = plt.subplots(3, 1, figsize=(8, 15))
for axs_idx, zernike_low, zernike_high, beta, scale in [
    (0, 2, 3, 1, 80),
    (1, 4, 8, 0.5, 0.2),
    (2, 9, 24, 0.5, 0.075),
]:
    # Obtain 2**19 Sobol rows
    sampler = qmc.Sobol(d=1, scramble=True, seed=314)
    sobol_rows = sampler.random_base2(19).ravel()
    # Create the distribution
    dist_inst = gennorm(beta=beta, scale=scale)
    # Transform the Sobol rows to the distribution
    transformed_rows = dist_inst.ppf(sobol_rows)
    # Where 99.9% of the data falls within
    xlim = dist_inst.ppf(0.999)
    ax[axs_idx].hist(
        transformed_rows,
        bins=300,
        range=(-xlim, xlim),
        density=True,
        alpha=0.4,
    )
    ax[axs_idx].axvline(x=0, color='black', lw=0.5)
    ax[axs_idx].set_xlim(-xlim, xlim)
    ax[axs_idx].set_title(
        fr'$Z_{{{zernike_low}-{zernike_high}}}$ ($\beta$={beta}, scale={scale})'
    )
    ax[axs_idx].set_ylabel('Density')
    if axs_idx == 2:
        ax[axs_idx].set_xlabel('Coefficient Value (nm)')
plt.savefig('sobol.png')
