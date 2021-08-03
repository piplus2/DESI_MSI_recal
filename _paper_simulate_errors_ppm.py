import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from KDEpy import FFTKDE
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from tqdm import tqdm


# def find_peak_kde(vals, kde_step, rel_height, check_other_peaks,
#                   return_boundaries):
#
#     # Find ppm error density peak and 0.25 interval - These represent the first
#     # candidates peaks
#     kde = FFTKDE(kernel='gaussian', bw='silverman')
#     xphi = \
#         np.arange(np.min(vals) - kde_step, np.max(vals) + kde_step, kde_step)
#     kde.fit(vals)
#     yphifft = kde.evaluate(xphi)
#     peaks, properties = find_peaks(yphifft, rel_height=rel_height, width=3)
#     maxpeak_idx = np.argmax(yphifft[peaks])
#     left_pt = properties['left_ips'][maxpeak_idx]
#     right_pt = properties['right_ips'][maxpeak_idx]
#
#     xpeak = xphi[peaks[maxpeak_idx]]
#
#     if check_other_peaks:
#         # If multimodal, then take the points corresponding to the highest peak
#         # of residuals
#         if len(peaks) != 1:
#             # Check left peaks
#             if maxpeak_idx > 0 and left_pt < peaks[maxpeak_idx - 1]:
#                 left_pt = \
#                     peaks[maxpeak_idx] - \
#                     (peaks[maxpeak_idx] - peaks[maxpeak_idx - 1]) / 2
#             # Check right peaks
#             if maxpeak_idx < len(peaks) - 1 and right_pt > peaks[maxpeak_idx + 1]:
#                 right_pt = \
#                     peaks[maxpeak_idx] + \
#                     (peaks[maxpeak_idx + 1] - peaks[maxpeak_idx]) / 2
#
#     boundaries = []
#     if return_boundaries:
#         boundaries = np.interp(
#             x=[left_pt, right_pt],
#             xp=np.arange(len(xphi)), fp=xphi)
#
#     return peaks, maxpeak_idx, left_pt, right_pt, xpeak, boundaries
#
#
# def find_boundary_from_ppm_err(obs_masses, th_masses, max_ppm=5, kde_step=1e-3):
#
#     # Biased error estimator
#     dmz = th_masses - obs_masses
#     dppm = dmz / th_masses * 1e6
#
#     _, _, _, _, dppm_peak, dppm_bounds = \
#         find_peak_kde(dppm, kde_step=kde_step,
#                       check_other_peaks=False, return_boundaries=True,
#                       rel_height=0.95)
#     if dppm_peak - dppm_bounds[0] >= max_ppm:
#         dppm_bounds[0] = dppm_peak - max_ppm
#     if dppm_bounds[1] - dppm_peak >= max_ppm:
#         dppm_bounds[1] = dppm_peak + max_ppm
#     dppm_mask = (dppm >= dppm_bounds[0]) & (dppm <= dppm_bounds[1])
#     dmz_peak = dppm_peak * th_masses / 1e6
#
#     # Calculate the residuals from the linear model delta_mass = delta_mass_peak
#     # = th_mass * ppm_error_peak. Calculate the highest peak and its left and
#     # right points, corresponding to 1% of the peak height.
#     res_dmz = (dmz - dmz_peak)[dppm_mask]
#     res_dmz_peaks, res_dmz_max_idx, left_pt, right_pt, res_dmz_peak, _ = \
#         find_peak_kde(res_dmz, kde_step=1e-5, rel_height=0.99,
#                       check_other_peaks=True, return_boundaries=False)
#     # Estimated bias from the distribution of the residuals
#     bias = res_dmz_peak
#
#     # Unbiased estimation ------------------------------------------------------
#
#     # Biased error estimator
#     th_masses_ = th_masses - bias
#     dmz = th_masses_ - obs_masses
#     dppm = dmz / th_masses_ * 1e6
#
#     _, _, _, _, dppm_peak, dppm_bounds = \
#         find_peak_kde(dppm, kde_step=kde_step, check_other_peaks=False,
#                       return_boundaries=True, rel_height=0.25)
#     dppm_mask = (dppm >= dppm_bounds[0]) & (dppm <= dppm_bounds[1])
#     dmz_peak = dppm_peak * th_masses / 1e6
#     res_dmz = (dmz - dmz_peak)[dppm_mask]
#
#     res_dmz_peaks, res_dmz_max_idx, left_pt, right_pt, res_dmz_peak, \
#         res_bounds = \
#         find_peak_kde(res_dmz, kde_step=1e-5, rel_height=0.99,
#                       check_other_peaks=True, return_boundaries=True)
#     # Determine the final mask
#
#     # Set maximum half interval equal to 0.005 m/z = 5 ppm error for 1000 m/z
#     lo_shift = res_bounds[0]
#     if np.abs(lo_shift) >= 0.005:
#         lo_shift = np.sign(lo_shift) * 0.005
#     hi_shift = res_bounds[1]
#     if np.abs(hi_shift) >= 0.005:
#         hi_shift = np.sign(hi_shift) * 0.005
#
#     lo_bound = lo_shift + dmz_peak[dppm_mask] + bias
#     hi_bound = hi_shift + dmz_peak[dppm_mask] + bias
#
#     mass_mask = (dmz - dmz_peak >= lo_shift) & (dmz - dmz_peak <= hi_shift)
#
#     return bias, mass_mask, lo_bound, hi_bound

from tools.functions import find_boundary_from_ppm_err
from tools.plot_style import set_mpl_params_mod


set_mpl_params_mod()

N_REP = 1000
N_TRUE = 500
N_NOISE = 200

means = [1, 5, 10, 20, 50]
sigmas = [0.1, 1, 2, 4, 6]

med_err = np.zeros((N_REP, len(means)))

for i, (ppm_mean, ppm_sig) in enumerate(zip(means, sigmas)):

    for rep in tqdm(range(N_REP)):

        ppm_err = np.random.normal(ppm_mean, ppm_sig, N_TRUE)

        mth_true = np.random.uniform(50, 1000, N_TRUE)
        mth_noise = np.random.uniform(50, 1000, N_NOISE)

        delta_mass_true = ppm_err * mth_true / 1e6
        beta0 = \
            np.random.choice([-1, 1], 1) * np.random.uniform(0.00001, 0.0009, 1)
        mobs_true = mth_true - (delta_mass_true + beta0)

        mmin, mmax = np.min(mth_true - mobs_true), np.max(mth_true - mobs_true)
        mobs_noise = mth_noise - np.random.uniform(mmin, mmax, N_NOISE)

        mobs = np.r_[mobs_true, mobs_noise]
        mth = np.r_[mth_true, mth_noise]

        # plt.scatter(mobs, mth - mobs)

        mass_mask, _, _, _, _ = \
            find_boundary_from_ppm_err(mobs, mth)

        mdl = \
            LinearRegression().fit(
                mobs[mass_mask].reshape(-1, 1), mth[mass_mask].reshape(-1, ))

        beta1 = 1 / (1 - 50 * 1e-6)
        beta1_min = 1 / (1 - np.min(ppm_err) * 1e-6)
        beta1_max = 1 / (1 - np.max(ppm_err) * 1e-6)

        fig, ax = plt.subplots(1, 3, dpi=300, figsize=(12, 3))

        err_mz = mth - mdl.predict(mobs.reshape(-1, 1))

        ax[0].hist(ppm_err, bins=30)
        ax[0].set_xlabel(r'$\delta$' + ' (ppm)')
        ax[0].set_title('Simulated true mass errors (ppm)')

        ax[1].scatter(mobs[mass_mask], mth[mass_mask] - mobs[mass_mask], c='green',
                      label='selected', s=2)
        ax[1].scatter(mobs[~mass_mask], mth[~mass_mask] - mobs[~mass_mask], c='red',
                      label='filtered', s=2)
        ax[1].plot(mobs_true, mobs_true * (beta1 - 1), ls='dashed',
                   c='black', lw=1)
        ax[1].plot(mobs_true, mobs_true * (beta1_min - 1), ls='dashed',
                   c='black', lw=1)
        ax[1].plot(mobs_true, mobs_true * (beta1_max - 1), ls='dashed',
                   c='black', lw=1)

        ax[1].set_xlabel(r'$M^{\#}$' + ' (m/z)')
        ax[1].set_ylabel(r'$M - M^{\#}$' + ' (m/z)')
        ax[1].legend()
        ax[1].set_title('Filtered peaks')

        ax[2].scatter(mobs_true, mth_true - mdl.predict(mobs_true.reshape(-1, 1)).ravel(),
                      c=mass_mask[:N_TRUE], cmap='Set1', s=1)
        ax[2].set_xlabel(r'$M^{\#}_{true}$' + ' (m/z)')
        ax[2].set_ylabel('Residuals')

        plt.tight_layout()
        plt.savefig('E:/CALIB_PAPER/New folder/example_sim_ppm.tif', format='tif')

        med_err[rep, i] = \
            np.median(
                np.abs(mth_true - mdl.predict(mobs_true.reshape(-1, 1)).ravel())
                / mth_true * 1e6)

        # plt.scatter(
        #     mobs_true, mth_true - mdl.predict(mobs_true.reshape(-1, 1)).ravel(),
        #     c=err_mz[:N_TRUE])

        # plt.scatter(mobs, mth - mobs, c=err_mz)
        # plt.scatter(mobs, mdl.predict(mobs.reshape(-1, 1)).ravel() - mobs)

        if med_err[rep, i] > ppm_mean:
            print('Error too large')
            break

fig = plt.figure(dpi=300, figsize=(4, 3))
ax = fig.add_subplot(111)
ax.boxplot(med_err)
ax.set_xticklabels([1, 5, 10, 20, 50])
ax.set_ylabel(r'$|\tilde{\Delta}|$' + ' (ppm)')
ax.set_xlabel('mean true error (ppm)')
ax.set_title('Effect of recalibration on simulated data')
plt.tight_layout()
plt.savefig('E:/CALIB_PAPER/New folder/simulated_errors.tif', format='tif')

pd.DataFrame(data=med_err, columns=means).to_csv('E:/CALIB_PAPER/simulated_errors_ppm.csv')