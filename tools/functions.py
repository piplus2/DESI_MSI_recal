#   Copyright 2018 by Paolo Inglese, National Phenome Centre, Imperial College
#   London
#   All rights reserved.
#   This file is part of DESI-MSI recalibration, and is released under the
#   "MIT License Agreement".
#   Please see the LICENSE file that should have been included as part of this
#   package.


import os
from typing import Union
from tqdm import tqdm
import numpy as np
import pandas as pd
from KDEpy import FFTKDE
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
from joblib import Parallel, delayed
import multiprocessing
from scipy.signal import find_peaks
import pygam
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt


def del_all_files_dir(dirname: str) -> None:
    for filename in os.listdir(dirname):
        file_path = os.path.join(dirname, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)


def dppm_to_dmz(dppm, refmz):
    return dppm * refmz / 1e6


def dmz_to_dppm(dmz, refmz):
    return dmz / refmz * 1e6


def filter_roi_px(msiobj, roi):
    if not np.all(roi.shape == msiobj.dim_xy[::-1]):
        raise ValueError('ROI has incompatible dimensions.')
    outpx = np.where(roi.ravel() == 0)[0]
    delpx = np.where(np.isin(msiobj.pixels_indices, outpx))[0]
    delpx = np.sort(delpx)
    msiobj.del_pixel(list(delpx))
    return msiobj


# Returns the hits for a list of reference masses, given a tolerance expressed
# in ppm. The search is performed in the N most intense peaks or, optionally,
# in all peaks (top_n = -1)
def search_ref_masses(msiobj, ref_masses, max_tolerance,
                      top_n: Union[int, str] = 100):
    # top_n = -1: search in all peaks
    # top_n = 'upper': search in peaks with intensity > 0.75 quantile
    # top_n = int: search in int highest peaks
    print('Searching reference masses ...')
    tol_masses = {m: dppm_to_dmz(max_tolerance, m) for m in ref_masses}
    matches = {m: {'pixel': [], 'mz': [], 'intensity': [], 'peak': []} for m in
               ref_masses}
    for i, msp in enumerate(tqdm(msiobj.msdata)):
        if top_n != -1 and top_n != 'upper':
            top_idx = np.argsort(msp[:, 1])[::-1]
            top_idx = top_idx[:int(top_n)]
        elif top_n == 'upper':
            threshold = np.quantile(msp[:, 1], q=0.9)
            top_idx = np.where(msp[:, 1] >= threshold)[0]

        # Remove masses that are outside of the interval
        skip_masses = np.full(len(ref_masses), False, dtype=bool)
        for j, m in enumerate(ref_masses):
            if top_n == -1:
                sm = msp[:, 0]
            else:
                sm = msp[top_idx, 0]
            if m - tol_masses[m] > sm[-1] or m + tol_masses[m] < sm[0]:
                skip_masses[j] = True
        search_masses = ref_masses.copy()
        search_masses = search_masses[~skip_masses]

        # Run the search
        if top_n == -1:
            sm = msp[:, 0]
        else:
            sm = msp[top_idx, 0]
        left_mass = np.asarray([m - tol_masses[m] for m in search_masses])
        right_mass = np.asarray([m + tol_masses[m] for m in search_masses])
        hit_lx = np.searchsorted(sm, left_mass, side='left')
        hit_rx = np.searchsorted(sm, right_mass, side='right')
        for m, lx, rx in zip(search_masses, hit_lx, hit_rx):
            if top_n == -1:
                hits = np.arange(lx, rx)
            else:
                hits = top_idx[lx:rx]
            if len(hits) == 0:
                continue
            for hit in hits:
                matches[m]['pixel'].append(msiobj.pixels_indices[i])
                matches[m]['mz'].append(msp[hit, 0])
                matches[m]['intensity'].append(msp[hit, 1])
                matches[m]['peak'].append(hit)
    for m in matches.keys():
        matches[m]['pixel'] = np.asarray(matches[m]['pixel'])
        matches[m]['mz'] = np.asarray(matches[m]['mz'])
        matches[m]['intensity'] = np.asarray(matches[m]['intensity'])
        matches[m]['peak'] = np.asarray(matches[m]['peak'])
    return matches


def gen_ref_list(ion_mode, verbose: bool):
    adducts = {'ES+': [1.0073, 22.9892, 38.9632],
               'ES-': [-1.0073, 34.9694]}

    if ion_mode == 'ES-':
        lipid_classes = ['hmdb_others', 'hmdb_pg', 'hmdb_pi', 'hmdb_ps',
                         'FA_branched', 'FA_straight_chain', 'FA_unsaturated',
                         'Cer', 'PA', 'PE', 'PG', 'PS', 'PI', 'CL', 'SM',
                         'glycine_conj']
    elif ion_mode == 'ES+':
        lipid_classes = ['hmdb_cholesterol', 'hmdb_glycerides', 'hmdb_pc',
                         'MG', 'DG', 'TG', 'PC', 'PE',
                         'SM', 'cholesterol', 'fatty_esters']  # , ]
    else:
        raise ValueError('Invalid `ion_mode`.')

    db_masses = pd.DataFrame()
    for lipid_class in lipid_classes:
        tmp = pd.read_csv(
            os.path.join('./db/new/{}_curated.csv'.format(lipid_class)))
        db_masses = db_masses.append(tmp)
        del tmp
    db_masses = np.sort(db_masses['MASS'].to_numpy())
    db_masses = np.round(db_masses, 4)
    db_masses = np.unique(db_masses)

    # Additional masses - often observed in DESI
    custom_ref_masses = {
        'ES+': np.array(
            [286.2144, 103.0997, 183.0660, 342.1162, 523.3638, 576.5118]),
        'ES-': np.array(
            [90.0317, 97.9769, 146.0691, 147.0532, 176.0321, 280.2402])
    }
    if verbose:
        print('Generating lock mass table ...')
    db_masses = np.unique(np.r_[custom_ref_masses[ion_mode], db_masses])
    ref_list = []
    for m in db_masses:
        for a in adducts[ion_mode]:
            ref_list.append(m + a)
    ref_list = np.asarray(ref_list)
    ref_list = np.round(ref_list, 4).astype(np.float32)
    ref_list = np.sort(ref_list)
    # Combine the reference masses - remove those close < 2e-4 m/z (these may
    # be due to approximation errors)
    ref_list = np.delete(ref_list, np.where(np.diff(ref_list) <= 2e-4)[0] + 1)
    ref_list = np.unique(ref_list)
    ref_list = ref_list.astype(np.float32)

    if verbose:
        print('Num. test masses = {}'.format(len(ref_list)))

    return ref_list


def find_kde_max(x, y, kde_values, remove_zeros=True):
    # Remove all zeros
    if remove_zeros:
        all_zeros = np.all(kde_values == 0, axis=0)
    else:
        all_zeros = np.full(kde_values.shape[0], False)

    max_kde = np.argmax(kde_values, axis=0)
    return x[~all_zeros], y[max_kde[~all_zeros]]


def ts_cv(x, y, s):

    cv = KFold(n_splits=5)
    error = []
    for trg, tst in cv.split(x):
        mdl = UnivariateSpline(x[trg], y[trg], s=s)
        yhat = mdl(x[tst])
        error.append(np.mean((yhat - y[tst])**2))
    
    return np.mean(error)


def fit_spline_kde(pixels, match_masses, ref_mass, plot_dir=None):
    if np.var(match_masses) <= 1e-6:
        s, u = np.unique(pixels, return_counts=True)
        spx = s
        smz = np.zeros(len(spx))
        for i in range(len(s)):
            if u[i] == 1:
                smz[i] = match_masses[pixels == s[i]]
            else:
                smz_ = match_masses[pixels == s[i]]
                smz[i] = smz_[np.argmin(np.abs(smz_ - ref_mass))]
        mdl = UnivariateSpline(x=spx, y=smz - ref_mass)
        use_kde = False
    else:
        use_kde = True
        data = np.c_[pixels, match_masses - ref_mass].astype(np.float64)
        data = data.astype(float)
        scaler = MinMaxScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        bandwidth = 2.575 * data.std() * data.shape[0] ** (-1 / 5)
        kde = FFTKDE(bw=bandwidth, kernel='tri').fit(data)
        grid, points = kde.evaluate(400)
        grid_mask = np.all((grid >= 0) & (grid <= 1), axis=1)
        grid = grid[grid_mask, :]
        x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
        z = points[grid_mask]
        z = z.reshape((len(x), len(y))).T
        xyi = scaler.inverse_transform(np.c_[x, y])
        xmax_kde, ymax_kde = find_kde_max(x=xyi[:, 0], y=xyi[:, 1],
                                          kde_values=z, remove_zeros=True)
        if np.var(xmax_kde) == 0:
            mdl = UnivariateSpline(x=xmax_kde, y=ymax_kde)
        else:
            # lam = np.logspace(-3, 3, 7)
            # n_splines = np.arange(5, 25, 5)
            # comb = np.array(np.meshgrid(lam, n_splines)).T.reshape(-1, 2)
            # gcv = Parallel(n_jobs=5)(delayed(gam)(xmax_kde.reshape(-1, 1),
            # ymax_kde.reshape(-1, ), l, int(n)) for (l, n) in zip(comb[:, 0],
            # comb[:, 1]))

            # mdl = pygam.LinearGAM(pygam.s(0, n_splines=10))
            # mdl.gridsearch(X=xmax_kde.reshape(-1, 1), y=ymax_kde.reshape(-1, ),
            #                progress=False)

            # use_gam = False
            s_vals = np.linspace(0.1, 0.9, 9)
            mse = []
            for s_ in s_vals:
                mse.append(
                    ts_cv(xmax_kde.reshape(-1, 1), ymax_kde.reshape(-1, ), s_))
            mdl = \
                UnivariateSpline(x=xmax_kde, y=ymax_kde,
                                 s=s_vals[np.argmin(mse)])

    # if use_gam:
    #     yhat = mdl.predict(pixels.reshape(-1, 1)).ravel()
    # else:
    yhat = mdl(pixels)

    # Plot
    if plot_dir is not None:
        fig = plt.figure(dpi=150, figsize=(4, 3))
        ax = fig.add_subplot(111)
        if use_kde:
            xx, yy = np.meshgrid(xyi[:, 0], xyi[:, 1])
            ax.pcolormesh(xx, yy, z, cmap='viridis', shading='nearest')
        ax.scatter(pixels, match_masses - ref_mass, s=3, edgecolor='white',
                   facecolor='none', linewidths=0.5)
        if use_kde:
            ax.scatter(
                xmax_kde, ymax_kde, s=3, edgecolor='red', facecolor='none',
                linewidths=0.5)
        ax.scatter(
            pixels, yhat, s=3, edgecolor='green', facecolor='none',
            linewidths=0.5)
        ax.set_xlim([pixels[0], pixels[-1]])
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'kde_' + str(ref_mass) + '.png'))
        plt.close()

    return yhat, use_kde


def kde_regress(msiobj, search_results, min_pct, max_disp, plot_dir=None):
    failed = []

    results = \
        {m: {'is_inlier': [], 'inlier_px': [], 'inlier_pct': [],
             'residuals': [], 'dispersion': []}
         for m in search_results.keys()}

    for m in tqdm(search_results.keys()):
        try:
            preds, is_kde = \
                fit_spline_kde(pixels=search_results[m]['pixel'],
                               match_masses=search_results[m]['mz'],
                               ref_mass=m, plot_dir=plot_dir)
        except:
            failed.append(m)
            continue
        results[m]['residuals'] = search_results[m]['mz'] - (m + preds)
        mad = 1.4826 * np.median(np.abs(results[m]['residuals']))
        results[m]['is_inlier'] = \
            (np.abs(results[m]['residuals']) <= 2 * mad)
        results[m]['inlier_px'] = \
            search_results[m]['pixel'][results[m]['is_inlier']]
        results[m]['inlier_pct'] = \
            len(np.unique(results[m]['inlier_px'])) / \
            len(msiobj.pixels_indices)
        results[m]['dispersion'] = np.max(2 * mad / (preds + m) * 1e6)

    # Remove failed kde models
    if len(failed) > 0:
        results = {m: results[m] for m in results.keys()
                   if m not in failed}

    # Select by coverage percentage and dispersion
    results = \
        {m: results[m] for m in results.keys() if
         (np.abs(results[m]['dispersion']) <= max_disp) &
         (results[m]['inlier_pct'] * 100 >= min_pct)}
    return results


class SMWrapper(BaseEstimator, RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors """

    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept

    # noinspection PyAttributeOutsideInit,PyPep8Naming
    def fit(self, X, y, **kwargs):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = self.model_class(y, X, **kwargs)
        self.results_ = self.model_.fit()

    # noinspection PyPep8Naming
    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.results_.predict(X)


def poly_regression(degree):
    return Pipeline([('poly', PolynomialFeatures(degree=degree)),
                     ('regressor', SMWrapper(sm.OLS))])


def poly_weighted(degree):
    return Pipeline([('poly', PolynomialFeatures(degree=degree)),
                     ('regressor', SMWrapper(sm.WLS))])


def poly_ransac(degree):
    return Pipeline([('poly', PolynomialFeatures(degree=degree)),
                     ('regressor', RANSACRegressor())])


def fit_model_thread(x, y, d, model, **kwargs):
    if model == 'ols':
        mdl = poly_regression(degree=d)
    elif model == 'wols':
        mdl = poly_weighted(degree=d)
    elif model == 'ransac':
        mdl = poly_ransac(degree=d)
    else:
        raise ValueError('Invalid model value.')
    mdl.fit(x.reshape(-1, 1), y.reshape(-1, ), **kwargs)
    return mdl


# calculate bic for regression
def calculate_bic(n, residuals, num_params):
    bic = \
        n * np.log(np.sum(residuals ** 2 + 1e-12) / n) + np.log(n) \
        * num_params
    # bic = n * np.log(error + 1e-12) + num_params * np.log(n)
    return bic


def fit_shift_model(x, y, max_degree, model='ols', error: str = 'mse',
                    **kwargs):
    if error == 'mse':
        err_func = mean_squared_error
    elif error == 'mae':
        err_func = mean_absolute_error
    else:
        raise ValueError('Invalid `error`.')

    if model != 'gam':
        if np.var(y) < 1e-10:
            model = 'ols'
            best_model = fit_model_thread(x, y, 0, model, **kwargs)
            bics = []
            best_degree = 0
        else:
            num_cores = multiprocessing.cpu_count()
            num_params = np.arange(1, max_degree + 1) + 1
            models = Parallel(n_jobs=np.min([num_cores - 1, max_degree + 1]))(
                delayed(fit_model_thread)(x, y, d, model) for d in
                range(1, max_degree + 1))
            yhats = [mdl.predict(x.reshape(-1, 1)).flatten() for mdl in models]
            resid = [y - yhat for yhat in yhats]
            bics = np.asarray(
                [calculate_bic(len(yhats[i]), resid[i], num_params[i]) for i in
                 range(len(yhats))])
            best_degree = np.arange(1, max_degree + 1)[np.argmin(bics)]
            best_model = models[np.argmin(bics)]
    else:
        gam = pygam.LinearGAM(pygam.s(0))
        gam.gridsearch(X=x.reshape(-1, 1), y=y, progress=False)
        best_model = gam
        best_degree = 0
        bics = []

    return {'bic': bics, 'best_model': best_model, 'best_degree': best_degree}


def find_boundary_from_ppm_err(obs_masses, th_masses, max_ppm=5, kde_step=1e-3):
    delta_mass = th_masses - obs_masses
    error = delta_mass / th_masses * 1e6

    kde = FFTKDE(kernel='gaussian', bw='silverman')
    xphi = \
        np.arange(np.min(error) - kde_step, np.max(error) + kde_step, kde_step)
    kde.fit(error)
    yphifft = kde.evaluate(xphi)
    peaks, properties = find_peaks(yphifft, rel_height=0.25, width=3)
    maxpeak = np.argmax(yphifft[peaks])

    left_pt = properties['left_ips'][maxpeak]
    right_pt = properties['right_ips'][maxpeak]

    xinterp = np.interp(
        x=[left_pt, right_pt],
        xp=np.arange(len(xphi)), fp=xphi)

    if xphi[peaks[maxpeak]] - xinterp[0] >= max_ppm:
        xinterp[0] = xphi[peaks[maxpeak]] - max_ppm
    if xinterp[1] - xphi[peaks[maxpeak]] >= max_ppm:
        xinterp[1] = xphi[peaks[maxpeak]] + max_ppm

    mass_peak = xphi[peaks[maxpeak]] * th_masses / 1e6
    error_mask = (error >= xinterp[0]) & (error <= xinterp[1])

    residuals = (delta_mass - mass_peak)[error_mask]
    mass_mask = (delta_mass - mass_peak >= np.min(residuals)) & \
                (delta_mass - mass_peak <= np.max(residuals))

    # Set maximum shift equal to 0.005 m/z
    lo_shift = np.min(residuals)
    if np.abs(lo_shift) >= 0.005:
        lo_shift = np.sign(lo_shift) * 0.005
    hi_shift = np.max(residuals)
    if np.abs(hi_shift) >= 0.005:
        hi_shift = np.sign(hi_shift) * 0.005

    lo_bound = lo_shift + mass_peak[error_mask]
    hi_bound = hi_shift + mass_peak[error_mask]

    return mass_mask, error_mask, mass_peak, lo_bound, hi_bound


def recal_pixel(x_fit, y_fit, x_pred, transform, max_degree):
    # Determine hits with close error in ppm
    in_mask, ppm_mask, peak_mass, lo_bound, hi_bound = \
        find_boundary_from_ppm_err(obs_masses=x_fit, th_masses=y_fit)

    # in_mask = (ppm >= xinterp[0]) & (ppm <= xinterp[1])

    if transform == 'sqrt':
        x_fit = np.sqrt(x_fit)
        y_fit = np.sqrt(y_fit)
        x_pred = np.sqrt(x_pred)

    poly_degree = np.min([max_degree, np.sum(in_mask) - 1])
    mdls = fit_shift_model(x=x_fit[in_mask], y=y_fit[in_mask],
                           max_degree=poly_degree, model='ols',
                           error='mae')
    model = mdls['best_model']

    mz_corrected = model.predict(x_pred.reshape(-1, 1)).ravel()
    if transform == 'sqrt':
        mz_corrected = mz_corrected ** 2

    if transform == 'sqrt':
        x_fit = x_fit ** 2

    return mz_corrected, model, in_mask, \
           pd.DataFrame(
               data=np.c_[x_fit[ppm_mask], peak_mass[ppm_mask], lo_bound,
                          hi_bound],
               columns=['mobs', 'peak', 'lo_bound', 'hi_bound'])
