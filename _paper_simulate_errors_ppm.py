import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from tools.functions import find_boundary_from_ppm_err
from tools.plot_style import set_mpl_params_mod


def slope_from_ppm(ppm, th, b0, d):
    if d == 'ORBITRAP':
        b1 = (th - b0) / (th * (1 - ppm * 1e-6))
    elif d == 'TOF':
        b1 = (np.sqrt(th) - b0) / np.sqrt(th * (1 - ppm * 1e-6))
    else:
        raise ValueError('Invalid dataset')
    return b1


def obs_mass(th, b0, b1, d):
    if d == 'ORBITRAP':
        obs = (th - b0) / b1
    elif d == 'TOF':
        obs = ((np.sqrt(th) - b0) / b1) ** 2
    else:
        raise ValueError('Invalid dataset')
    return obs


def slope_to_ppm(b1, d):
    if d == 'ORBITRAP':
        ppm = (b1 - 1) / b1 * 1e6
    elif d == 'TOF':
        ppm = (b1 ** 2 - 1) / b1 ** 2 * 1e6
    else:
        raise ValueError('Invalid dataset')
    return ppm


set_mpl_params_mod()

N_REP = 1000
N_TRUE = 500
N_NOISE = 200

dataset = 'ORBITRAP'

means = [1, 5, 10, 20, 50]
sigmas = [0.1, 1, 2, 4, 6]

med_err = np.zeros((N_REP, len(means)))

for i, (ppm_mean, ppm_sig) in enumerate(zip(means, sigmas)):

    for rep in tqdm(range(N_REP)):

        ppm_err = np.random.normal(ppm_mean, ppm_sig, N_TRUE)

        mth_true = np.random.uniform(50, 1000, N_TRUE)
        mth_noise = np.random.uniform(50, 1000, N_NOISE)

        beta0 = \
            np.random.choice([-1, 1], 1) * np.random.uniform(0.00001, 0.0009, 1)
        beta1 = slope_from_ppm(ppm_err, mth_true, beta0, dataset)

        mobs_true = obs_mass(mth_true, beta0, beta1, dataset)

        mmin, mmax = np.min(mth_true - mobs_true), np.max(mth_true - mobs_true)
        mobs_noise = mth_noise - np.random.uniform(mmin, mmax, N_NOISE)

        mobs = np.r_[mobs_true, mobs_noise]
        mth = np.r_[mth_true, mth_noise]

        mass_mask, _, _, _, _ = \
            find_boundary_from_ppm_err(mobs, mth)

        if dataset == 'TOF':
            mdl = \
                LinearRegression(fit_intercept=True).fit(
                    np.sqrt(mobs[mass_mask]).reshape(-1, 1),
                    np.sqrt(mth[mass_mask]).reshape(-1, ))
        elif dataset == 'ORBITRAP':
            mdl = \
                LinearRegression(fit_intercept=True).fit(
                    mobs[mass_mask].reshape(-1, 1),
                    mth[mass_mask].reshape(-1, ))
        else:
            raise ValueError('Invalid dataset')

        if dataset == 'TOF':
            preds = \
                (mdl.predict(np.sqrt(mobs_true.reshape(-1, 1))).ravel()) ** 2
        elif dataset == 'ORBITRAP':
            preds = mdl.predict(mobs_true.reshape(-1, 1)).ravel()
        else:
            raise ValueError('Invalid dataset')

        med_err[rep, i] = np.median(np.abs(mth_true - preds) / mth_true * 1e6)

        if med_err[rep, i] > ppm_mean:
            raise RuntimeError('Error too large')

fig = plt.figure(dpi=300, figsize=(4, 3))
ax = fig.add_subplot(111)
ax.boxplot(med_err)
ax.set_xticklabels([1, 5, 10, 20, 50])
ax.set_ylabel(r'$|\tilde{\Delta}|$' + ' (ppm)')
ax.set_xlabel('mean true error (ppm)')
ax.set_title('Effect of recalibration on simulated data')
plt.tight_layout()
plt.savefig('E:/CALIB_PAPER/New folder/simulated_errors_' + dataset + '.pdf',
            format='pdf')

pd.DataFrame(data=med_err, columns=means).to_csv(
    'E:/CALIB_PAPER/simulated_errors_ppm_' + dataset + '.csv')


# fig, ax = plt.subplots(1, 3, dpi=300, figsize=(12, 3))
#
# # beta1hat = slope_from_ppm(ppm_mean, dataset)
# # beta1_min = slope_from_ppm(np.min(ppm_err), dataset)
# # beta1_max = slope_from_ppm(np.max(ppm_err), dataset)
#
# ax[0].hist(ppm_err, bins=30)
# ax[0].set_xlabel(r'$\delta$' + ' (ppm)')
# ax[0].set_title('Simulated true mass errors (ppm)')
#
# ax[1].scatter(
#     mobs[mass_mask], mth[mass_mask] - mobs[mass_mask], c='green',
#     label='selected', s=2)
# ax[1].scatter(
#     mobs[~mass_mask], mth[~mass_mask] - mobs[~mass_mask], c='red',
#     label='filtered', s=2)
# # ax[1].plot(
# #     mobs_true, mobs_true * (beta1hat - 1), ls='dashed', c='black', lw=1)
# # ax[1].plot(
# #     mobs_true, mobs_true * (beta1_min - 1), ls='dashed', c='black',
# #     lw=1)
# # ax[1].plot(
# #     mobs_true, mobs_true * (beta1_max - 1), ls='dashed', c='black',
# #     lw=1)
#
# ax[1].set_xlabel(r'$M^{\#}$' + ' (m/z)')
# ax[1].set_ylabel(r'$M - M^{\#}$' + ' (m/z)')
# ax[1].legend()
# ax[1].set_title('Filtered peaks')
#
# ax[2].scatter(
#     np.sqrt(mobs_true), np.sqrt(mth_true) -
#     mdl.predict(np.sqrt(mobs_true).reshape(-1, 1)).ravel(),
#     c=mass_mask[:N_TRUE], cmap='Set1', s=1)
# ax[2].set_xlabel(r'$M^{\#}_{true}$' + ' (m/z)')
# ax[2].set_ylabel('Residuals')
#
# plt.tight_layout()
# plt.savefig(
#     'E:/CALIB_PAPER/New folder/example_sim_ppm.tif', format='tif')