#   Copyright 2018 by Paolo Inglese, National Phenome Centre, Imperial College
#   London
#   All rights reserved.
#   This file is part of DESI-MSI recalibration, and is released under the
#   "MIT License Agreement".
#   Please see the LICENSE file that should have been included as part of this
#   package.
import bisect
import os
from typing import Union
from tqdm import tqdm
import numpy as np
import pandas as pd
from KDEpy import FFTKDE
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
from joblib import Parallel, delayed
import multiprocessing
from scipy.stats import median_abs_deviation


def del_all_files_dir(dirname: str) -> None:
    for filename in os.listdir(dirname):
        file_path = os.path.join(dirname, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)


def dppm_to_dmz(dppm, refmz):
    return dppm * refmz / 1e6


def dmz_to_dppm(dmz, refmz):
    return dmz / refmz * 1e6


# Returns the hits for a list of reference masses, given a tolerance expressed
# in ppm. The search is performed in the N most intense peaks or, optionally,
# in all peaks (top_n = -1)
def search_ref_masses(msiobj, ref_masses, max_tolerance,
                      top_n: Union[int, str] = 100):
    # top_n = -1: search in all peaks
    # top_n = 'upper': search in peaks with intensity > 0.9 quantile
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
        else:
            top_idx = np.arange(msp.shape[0])
        for m in ref_masses:
            mass_search = msp[top_idx, 0]
            left = m - tol_masses[m]
            right = m + tol_masses[m]
            if right < mass_search[0]:
                continue
            if left > mass_search[-1]:
                continue
            hit_left = bisect.bisect_left(mass_search, left)
            hit_right = bisect.bisect_right(mass_search, right)
            hits = top_idx[hit_left:hit_right]
            # hits = top_idx[np.where((msp[top_idx, 0] >= m - tol_masses[m]) & (
            #         msp[top_idx, 0] <= m + tol_masses[m]))[0]]
            if len(hits) == 0:
                continue
            if len(hits) > 1:
                hits = [hits[np.argmax(msp[hits, 1])]]
            for hit in hits:
                matches[m]['pixel'].append(int(msiobj.pixels_indices[i]))
                matches[m]['mz'].append(float(msp[hit, 0]))
                matches[m]['intensity'].append(float(msp[hit, 1]))
                matches[m]['peak'].append(int(hit))
    for m in matches.keys():
        matches[m]['pixel'] = np.asarray(matches[m]['pixel'], dtype=int)
        matches[m]['mz'] = np.asarray(matches[m]['mz'], dtype=float)
        matches[m]['intensity'] = np.asarray(matches[m]['intensity'],
                                             dtype=float)
        matches[m]['peak'] = np.asarray(matches[m]['peak'], dtype=int)
    return matches


def gen_ref_list(ion_mode, verbose: bool):
    adducts = {'ES+': [1.0073, 22.9892],
               'ES-': [-1.0073]}

    if ion_mode == 'ES-':
        lipid_classes = ['hmdb_others', 'hmdb_pg', 'hmdb_pi', 'hmdb_ps',
                         'FA_straight_chain', 'FA_unsaturated', 'Cer',
                         'PA', 'PE']
    else:
        lipid_classes = ['hmdb_cholesterol', 'hmdb_glycerides', 'hmdb_pc',
                         'FA_hydroxy', 'MG', 'TG', 'DG']

    db_masses = pd.DataFrame()
    for lipid_class in lipid_classes:
        tmp = pd.read_csv(
            os.path.join('./db/{}_curated.csv'.format(lipid_class)))
        db_masses = db_masses.append(tmp)
        del tmp
    db_masses = np.sort(db_masses['MASS'].to_numpy())
    db_masses = np.round(db_masses, 4)
    db_masses = np.unique(db_masses)

    # Additional masses - often observed in DESI
    custom_ref_masses = {
        'ES+': np.array([309.2036, 104.1070, 184.0733, 296.0660, 381.0794,
                         562.3269, 577.5190, 782.5670, 820.5253, 867.6497,
                         907.7725]),
        'ES-': np.array([255.2330, 89.0244, 96.9696, 124.0074, 145.0619,
                         146.0459, 175.0248, 279.2329, 281.2485, 283.2643,
                         303.2330, 500.2783, 519.1508, 599.3202, 697.4814,
                         699.4970, 742.5392, 764.5236, 768.5549, 824.5002,
                         859.5342, 885.5499, 909.5499])}
    # Combine the reference masses - remove those close < 2e-4 m/z (these may
    # be due to approximation errors)
    db_masses = np.unique(np.r_[custom_ref_masses[ion_mode], db_masses])
    db_masses = np.delete(db_masses, np.where(np.diff(db_masses) < 2e-4)[0] + 1)

    if verbose:
        print('Generating lock mass table ...')
    ref_list = []
    for m in db_masses:
        for a in adducts[ion_mode]:
            ref_list.append(m + a)
    ref_list = np.asarray(ref_list)
    ref_list = np.round(ref_list, 4)
    ref_list = np.unique(ref_list)

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


def fit_spline_kde(pixels, match_masses, ref_mass):
    if np.var(match_masses) == 0:
        spline = UnivariateSpline(x=pixels, y=match_masses)
        use_kde = False
    else:
        use_kde = True
        data = np.c_[pixels, match_masses - ref_mass]
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
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
        spline = UnivariateSpline(x=xmax_kde, y=ymax_kde)

    predictions = spline(pixels)
    return predictions, use_kde


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


def fit_model_thread(x, y, d, model, **kwargs):
    if model == 'ols':
        mdl = poly_regression(degree=d)
    elif model == 'wols':
        mdl = poly_weighted(degree=d)
    else:
        raise ValueError('Invalid model value.')
    mdl.fit(x.reshape(-1, 1), y.reshape(-1, ), **kwargs)
    return mdl


# calculate bic for regression
def calculate_bic(n, mse, num_params):
    bic = n * np.log(mse) + num_params * np.log(n)
    return bic


def fit_shift_model(x, y, max_degree=5, model='ols', error: str = 'mse',
                    **kwargs):
    if error == 'mse':
        err_func = mean_squared_error
    elif error == 'mae':
        err_func = mean_absolute_error
    else:
        raise ValueError('Invalid `error`.')
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
        mses = [err_func(y, yhat) for yhat in yhats]
        bics = np.asarray(
            [calculate_bic(len(yhats[i]), mses[i], num_params[i]) for i in
             range(len(yhats))])
        best_degree = np.arange(1, max_degree + 1)[np.argmin(bics)]
        best_model = models[np.argmin(bics)]
    return {'bic': bics, 'best_model': best_model, 'best_degree': best_degree}


def recal_pixel(x_fit, y_fit, x_pred, transform, max_degree):
    # Determine hits with close error in ppm
    ppm = ((x_fit - y_fit) / y_fit * 1e6)
    mad = median_abs_deviation(ppm)
    in_mask = np.abs(ppm - np.median(ppm)) <= mad

    # Calculate weights
    w = 1 / np.abs(ppm[in_mask])
    w[ppm[in_mask] == 0] = np.inf

    if transform == 'sqrt':
        x_fit = np.sqrt(x_fit)
        y_fit = np.sqrt(y_fit)
        x_pred = np.sqrt(x_pred)

    poly_degree = np.min([max_degree, np.sum(in_mask) - 1])

    mdls = fit_shift_model(x_fit[in_mask], y_fit[in_mask],
                           max_degree=poly_degree, model='wols',
                           error='mae', weights=w)
    model = mdls['best_model']

    mz_corrected = model.predict(x_pred.reshape(-1, 1)).ravel()
    if transform == 'sqrt':
        mz_corrected = mz_corrected ** 2

    return mz_corrected, model
