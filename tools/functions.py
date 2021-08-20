#   Copyright 2018 by Paolo Inglese, National Phenome Centre, Imperial College
#   London
#   All rights reserved.
#   This file is part of DESI-MSI recalibration, and is released under the
#   "MIT License Agreement".
#   Please see the LICENSE file that should have been included as part of this
#   package.


import multiprocessing
import numbers
import os
from typing import Union, List, Dict, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygam
import statsmodels.api as sm
from KDEpy import FFTKDE
from joblib import Parallel, delayed
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from tqdm import tqdm

if TYPE_CHECKING:
    from .msi import MSI


def del_all_files_dir(dirname: str) -> None:
    for filename in os.listdir(dirname):
        file_path = os.path.join(dirname, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)


def dppm_to_dmz(
        dppm: Union[np.ndarray, float], refmz: float) -> \
        Union[np.ndarray, float]:
    return dppm * refmz / 1e6


def dmz_to_dppm(
        dmz: Union[np.ndarray, float], refmz: float) -> \
        Union[np.ndarray, float]:
    return dmz / refmz * 1e6


def filter_roi_px(msiobj: 'MSI', roi: np.ndarray) -> 'MSI':
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
def search_ref_masses(
        msiobj: 'MSI', ref_masses: Union[List[float], np.ndarray],
        max_tolerance: float, top_n: Union[int, str] = 100, parallel: bool = False) -> Dict:
    # top_n = -1: search in all peaks
    # top_n = 'upper': search in peaks with intensity > 0.75 quantile
    # top_n = int: search in int highest peaks
    
    def __thread(msp_, idx_, m_, tol_):
        md = {m: {x: [] for x in ['pixel', 'mz', 'intensity', 'peak']} 
              for m in m_}
        
        skip_masses = np.full(len(m_), False, dtype=bool)
        for j, m__ in enumerate(m_):
            if top_n == -1:
                sm_ = msp_[:, 0]
            else:
                sm_ = msp_[top_idx, 0]
            if m__ - tol_[m__] > sm_[-1] or m__ + tol_[m__] < sm_[0]:
                skip_masses[j] = True
        
        search_m = m_.copy()
        search_m = search_m[~skip_masses]
        
        if top_n != -1 and top_n != 'upper':
            top_idx = np.argsort(msp_[:, 1])[::-1]
            top_idx = top_idx[:int(top_n)]
        elif top_n == 'upper':
            threshold = np.quantile(msp_[:, 1], q=0.9)
            top_idx = np.where(msp_[:, 1] >= threshold)[0]
            
        lx_mass = np.asarray([m__ - tol_[m__] for m__ in search_m])
        rx_mass = np.asarray([m__ + tol_[m__] for m__ in search_m])
        
        hit_lx = np.searchsorted(sm_, lx_mass, side='left')
        hit_rx = np.searchsorted(sm_, rx_mass, side='right')
        
        for m__, lx, rx in zip(m_, hit_lx, hit_rx):
            if top_n == -1:
                hits = np.arange(lx, rx)
            else:
                hits = top_idx[lx:rx]
            if len(hits) == 0:
                continue
            for hit in hits:
                md[m__]['pixel'].append(idx_)
                md[m__]['mz'].append(msp_[hit, 0])
                md[m__]['intensity'].append(msp_[hit, 1])
                md[m__]['peak'].append(hit)
        return md
    
    print('Searching reference masses ...')
    tol_masses = {m: dppm_to_dmz(max_tolerance, m) for m in ref_masses}
    
    matches = {m: {'pixel': [], 'mz': [], 'intensity': [], 'peak': []} for m in
               ref_masses}
  
    if parallel:
        md_px = Parallel(n_jobs=multiprocessing.cpu_count() - 1)(
            delayed(__thread)(msp_, idx_, ref_masses, tol_masses)
            for msp_, idx_ in tqdm(zip(msiobj.msdata, msiobj.pixels_indices), total=len(msiobj.msdata)))
    else:
        md_px = []
        for i, msp in enumerate(tqdm(msiobj.msdata)):
            md_px.append(__thread(msp, msiobj.pixels_indices[i], ref_masses, tol_masses))
    for md in md_px:
        for m in md.keys():
            if len(md[m]) > 0:
                matches[m]['pixel'].append(md[m]['pixel'])
                matched[m]['mz'].append(md[m]['mz'])
                matched[m]['intensity'].append(md[m]['intensity'])
                matched[m]['peak'].append(md[m]['peak'])
    del md

    for m in matches.keys():
        matches[m]['pixel'] = np.asarray(matches[m]['pixel'])
        matches[m]['mz'] = np.asarray(matches[m]['mz'])
        matches[m]['intensity'] = np.asarray(matches[m]['intensity'])
        matches[m]['peak'] = np.asarray(matches[m]['peak'])
    return matches


def gen_ref_list(ion_mode: str, verbose: bool) -> np.ndarray:
    adducts = {'ES+': [1.0073, 22.9892, 38.9632],
               'ES-': [-1.0073, 34.9694]}

    if ion_mode == 'ES-':
        ref_tables = ['hmdb_others', 'hmdb_pg', 'hmdb_pi', 'hmdb_ps',
                      'FA_branched', 'FA_straight_chain', 'FA_unsaturated',
                      'Cer', 'PA', 'PE', 'PG', 'PS', 'PI', 'CL', 'SM',
                      'glycine_conj', 'custom_pos']
    elif ion_mode == 'ES+':
        ref_tables = ['hmdb_cholesterol', 'hmdb_glycerides', 'hmdb_pc',
                      'MG', 'DG', 'TG', 'PC', 'PE', 'custom_neg',
                      'SM', 'cholesterol', 'fatty_esters']
    else:
        raise ValueError('Invalid `ion_mode`.')

    db_masses = pd.DataFrame()
    for lipid_class in ref_tables:
        tmp = pd.read_csv(
            os.path.join('./db/{}_curated.csv'.format(lipid_class)))
        db_masses = db_masses.append(tmp)
        del tmp
    db_masses = np.sort(db_masses['MASS'].to_numpy())
    db_masses = np.round(db_masses, 4)
    db_masses = np.unique(db_masses)

    # Additional masses - often observed in DESI
    # custom_ref_masses = {
    #     'ES+': np.array(
    #         [286.2144, 103.0997, 183.0660, 342.1162, 523.3638, 576.5118]),
    #     'ES-': np.array(
    #         [90.0317, 97.9769, 146.0691, 147.0532, 176.0321, 280.2402])
    # }
    if verbose:
        print('Generating lock mass table ...')
    # db_masses = np.unique(np.r_[custom_ref_masses[ion_mode], db_masses])
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


class KDEMassRecal:
    grid_size: int
    min_pct: float
    max_disp_ppm: float
    max_poly_degree: int
    transform: str
    ref_masses: Union[List[float], np.ndarray, None]
    kde_model: str
    kde_bw: Union[str, float]
    smooth: Union[str, float]

    def __init__(
            self, min_pct: float,
            max_disp_ppm: float,
            transform: str = 'none',
            max_poly_degree: int = 5,
            kde_bw: Union[float, str] = 'silverman',
            grid_size: Union[int, float] = 2**10,
            smooth: Union[float, str] = 'cv',
            parallel: bool = False,
            plot_dir: Union[str, None] = None,
            plot_dim_xy: Union[np.ndarray, None] = None,
            plot: bool = False):

        if transform not in ['sqrt', 'none']:
            raise ValueError('`transform` can be either \'sqrt\' or \'none\'')

        if kde_bw is str and kde_bw != 'silverman':
            raise ValueError('`kde_bw` can be numeric or \'silverman\'')
        if kde_bw is numbers.Number and kde_bw <= 0.0:
            raise ValueError('`kde_bw` must be positive')
        if smooth is str and smooth != 'cv':
            raise ValueError('`smooth` can be numeric or \'cv\'')

        self.grid_size = int(grid_size)
        self.min_pct = min_pct
        self.max_disp_ppm = max_disp_ppm
        self.max_poly_degree = max_poly_degree
        self.transform = transform
        self.ref_masses = None
        self.kde_bw = kde_bw
        self.smooth = smooth

        self.__parallel = parallel

        self.__kde_max_ppm = 5,
        self.__kde_step_ppm = 1e-3
        self.__kde_height_ppm = 0.25

        self.__kde_step_resid = 1e-5
        self.__kde_height_resid = 0.99
        self.__kde_max_resid = 0.005

        if plot_dir is not None:
            if not os.path.isdir(plot_dir):
                raise IOError('`plot_dir` {} not found.')

        if plot and ((plot_dir is None) or (plot_dim_xy is None)):
            raise ValueError('plot arguments missing')

        self.__plot_dir = plot_dir
        self.__im_dimxy = plot_dim_xy
        self.__plot = plot

        self.__max_njobs = multiprocessing.cpu_count()

    # Private methods ----------------------------------------------------------

    @staticmethod
    def __find_kde_max(x, y, kde_values, remove_zeros=True):
        # Remove all zeros
        if remove_zeros:
            all_zeros = np.all(kde_values == 0, axis=0)
        else:
            all_zeros = np.full(kde_values.shape[0], False)
        max_kde = np.argmax(kde_values, axis=0)
        return x[~all_zeros], y[max_kde[~all_zeros]]

    @staticmethod
    def __ts_cv(x, y, s):
        cv = KFold(n_splits=5)
        error = []
        for trg, tst in cv.split(x):
            mdl = UnivariateSpline(x[trg], y[trg], s=s)
            yhat = mdl(x[tst])
            error.append(np.mean((yhat - y[tst]) ** 2))
        return np.mean(error)

    # calculate bic for regression
    @staticmethod
    def __calculate_bic(n, residuals, num_params):
        bic = \
            n * np.log(np.sum(residuals ** 2 + 1e-12) / n) + np.log(n) \
            * num_params
        return bic

    def __find_boundary_from_ppm_err(self, obs_masses, th_masses):
        delta_mass = th_masses - obs_masses
        error = delta_mass / th_masses * 1e6
        # Find ppm error density peak and 0.25 interval - These represent the
        # first candidates peaks
        kde = FFTKDE(kernel='gaussian', bw='silverman')
        xphi = \
            np.arange(np.min(error) - self.__kde_step_ppm,
                      np.max(error) + self.__kde_step_ppm,
                      self.__kde_step_ppm)
        kde.fit(error)
        yphifft = kde.evaluate(xphi)
        peaks, properties = \
            find_peaks(yphifft, rel_height=self.__kde_height_ppm, width=3)
        maxpeak = np.argmax(yphifft[peaks])

        left_pt = properties['left_ips'][maxpeak]
        right_pt = properties['right_ips'][maxpeak]

        xinterp = \
            np.interp(x=[left_pt, right_pt], xp=np.arange(len(xphi)), fp=xphi)

        if xphi[peaks[maxpeak]] - xinterp[0] >= self.__kde_max_ppm:
            xinterp[0] = xphi[peaks[maxpeak]] - self.__kde_max_ppm
        if xinterp[1] - xphi[peaks[maxpeak]] >= self.__kde_max_ppm:
            xinterp[1] = xphi[peaks[maxpeak]] + self.__kde_max_ppm

        delta_mass_peak = xphi[peaks[maxpeak]] * th_masses / 1e6
        error_mask = (error >= xinterp[0]) & (error <= xinterp[1])

        # Calculate the residuals from the linear model delta_mass =
        # delta_mass_peak = th_mass * ppm_error_peak. Calculate the highest
        # peak and its left and right points, corresponding to 50% of the peak
        # height.
        residuals = (delta_mass - delta_mass_peak)[error_mask]
        resphi = \
            np.arange(np.min(residuals) - self.__kde_step_resid,
                      np.max(residuals) + self.__kde_step_resid,
                      self.__kde_step_resid)
        kde.fit(residuals)
        resphikde = kde.evaluate(resphi)
        respeaks, resproperties = \
            find_peaks(resphikde, rel_height=self.__kde_height_resid, width=1)
        maxpeakres = np.argmax(resphikde[respeaks])
        left_pt_res = resproperties['left_ips'][maxpeakres]
        right_pt_res = resproperties['right_ips'][maxpeakres]

        # If multimodal, then take the points corresponding to the highest peak
        # of residuals
        if len(respeaks) != 1:
            # Check left peaks
            if maxpeakres > 0 and left_pt_res < respeaks[maxpeakres - 1]:
                left_pt_res = \
                    respeaks[maxpeakres] - \
                    (respeaks[maxpeakres] - respeaks[maxpeakres - 1]) / 2
            # Check right peaks
            if maxpeakres < len(respeaks) - 1 and \
                    right_pt_res > respeaks[maxpeakres + 1]:
                right_pt_res = \
                    respeaks[maxpeakres] + \
                    (respeaks[maxpeakres + 1] - respeaks[maxpeakres]) / 2

        resxinterp = \
            np.interp(x=[left_pt_res, right_pt_res], xp=np.arange(len(resphi)),
                      fp=resphi)

        # Set maximum half interval equal to 0.005 m/z = 5 ppm error for
        # 1000 m/z
        lo_shift = resxinterp[0]
        if np.abs(lo_shift) >= self.__kde_max_resid:
            lo_shift = np.sign(lo_shift) * self.__kde_max_resid
        hi_shift = resxinterp[1]
        if np.abs(hi_shift) >= self.__kde_max_resid:
            hi_shift = np.sign(hi_shift) * self.__kde_max_resid

        lo_bound = lo_shift + delta_mass_peak[error_mask]
        hi_bound = hi_shift + delta_mass_peak[error_mask]

        mass_mask = \
            (delta_mass - delta_mass_peak >= lo_shift) & \
            (delta_mass - delta_mass_peak <= hi_shift)

        return mass_mask, error_mask, delta_mass_peak, lo_bound, hi_bound

    def __fit_pixel_recal_model(self, x, y, d):

        def __fit_model_thread(x_, y_, d_, **kwargs):
            mdl = poly_regression(degree=d_)
            mdl.fit(x_.reshape(-1, 1), y_.reshape(-1, ), **kwargs)
            return mdl

        num_params = np.arange(1, d + 1) + 1
        # If using parallel for the main loop we don't do parallel here
        if not self.__parallel:
            models = \
                Parallel(n_jobs=np.min([self.__max_njobs - 1, d + 1]))(
                    delayed(__fit_model_thread)(x, y, d)
                    for d in range(1, d + 1))
        else:
            models = [__fit_model_thread(x, y, d) for d in range(1, d + 1)]
        yhats = [mdl.predict(x.reshape(-1, 1)).flatten() for mdl in models]
        resid = [y - yhat for yhat in yhats]
        bics = np.asarray(
            [self.__calculate_bic(len(yhats[i]), resid[i], num_params[i])
             for i in range(len(yhats))])
        best_degree = np.arange(1, d + 1)[np.argmin(bics)]
        best_model = models[np.argmin(bics)]
        return {
            'bic': bics, 'best_model': best_model, 'best_degree': best_degree}

    def __recal_pixel(self, x_fit, y_fit, x_pred):
        # Determine hits with close error in ppm
        in_mask, _, _, lo_bound, hi_bound = \
            self.__find_boundary_from_ppm_err(obs_masses=x_fit, th_masses=y_fit)

        if self.transform == 'sqrt':
            x_fit = np.sqrt(x_fit)
            y_fit = np.sqrt(y_fit)
            x_pred = np.sqrt(x_pred)

        poly_degree = np.min([self.max_poly_degree, np.sum(in_mask) - 1])
        mdls = \
            self.__fit_pixel_recal_model(x_fit[in_mask], y_fit[in_mask],
                                         poly_degree)
        model = mdls['best_model']

        mz_corrected = model.predict(x_pred.reshape(-1, 1)).ravel()
        if self.transform == 'sqrt':
            mz_corrected = mz_corrected ** 2

        return mz_corrected, model, in_mask

    def __fit_spline_kde(self, pixels, match_masses, ref_mass):

        grid_size = int(self.grid_size)

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
        else:
            data = np.c_[pixels, match_masses - ref_mass].astype(np.float64)
            data = data.astype(float)
            scaler = MinMaxScaler()
            scaler.fit(data)
            data = scaler.transform(data)
            if self.kde_bw == 'silverman':
                bandwidth = 2.576 * data.std(ddof=1) * data.shape[0] ** (-1 / 5)
            else:
                bandwidth = float(self.kde_bw)
            kde = FFTKDE(bw=bandwidth, kernel='tri').fit(data)
            grid, points = kde.evaluate(grid_size)
            grid_mask = np.all((grid >= 0) & (grid <= 1), axis=1)
            grid = grid[grid_mask, :]
            x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
            z = points[grid_mask]
            z = z.reshape((len(x), len(y))).T
            xyi = scaler.inverse_transform(np.c_[x, y])
            xmax_kde, ymax_kde = \
                self.__find_kde_max(x=xyi[:, 0], y=xyi[:, 1], kde_values=z,
                                    remove_zeros=True)
            if np.var(xmax_kde) == 0:
                mdl = UnivariateSpline(x=xmax_kde, y=ymax_kde)
            else:
                if not isinstance(self.smooth,
                                  numbers.Number) and self.smooth == 'cv':
                    s_vals = np.logspace(-4, -1, 20)
                    mse = []
                    for s_ in s_vals:
                        mse.append(
                            self.__ts_cv(xmax_kde.reshape(-1, 1),
                                         ymax_kde.reshape(-1, ), s_))
                    s_value = s_vals[np.argmin(mse)]
                else:
                    s_value = self.smooth
                mdl = \
                    UnivariateSpline(x=xmax_kde.reshape(-1, 1),
                                     y=ymax_kde.reshape(-1, ),
                                     s=s_value)
        yhat = mdl(pixels)

        return yhat

    # Public methods -----------------------------------------------------------

    def recalibrate(self, msiobj: 'MSI', search_results) -> 'MSI':
        matches = \
            {m: search_results[m] for m in search_results.keys() if
             len(np.unique(search_results[m]['pixel'])) /
             len(msiobj.pixels_indices) * 100.0 >= self.min_pct}

        def __thread(x_, y_, m_, npx_):
            yhat_ = \
                self.__fit_spline_kde(pixels=x_, match_masses=y_, ref_mass=m_)
            # Find outliers
            res_ = y_ - m_ - yhat_
            mad_ = 1.4826 * np.median(np.abs(res_))
            inliers_ = np.abs(res_) <= 2 * mad_
            inliers_px_ = x_[inliers_]
            disp_ = np.max(2 * mad_ / (yhat_ + m_)) * 1e6
            pct_ = len(np.unique(inliers_px_)) / npx_
            return res_, inliers_, inliers_px_, pct_, disp_

        def __fit_time_series(x_, y_):
            if np.var(y_) < 1e-10:
                model = poly_regression(degree=0)
                model.fit(x_.reshape(-1, 1), y_.reshape(-1, ))
            else:
                model = pygam.LinearGAM(pygam.l(0) + pygam.s(0))
                model.gridsearch(X=x_.reshape(-1, 1), y=y_, progress=False)
            return model

        print('Finding outliers in time series ...')
        inliers = {m: [] for m in matches.keys()}
        inliers_px = {m: [] for m in matches.keys()}
        inliers_pct = {m: [] for m in matches.keys()}
        disp_ppm = {m: [] for m in matches.keys()}
        residuals = {m: [] for m in matches.keys()}

        if self.__parallel:
            spline_res = Parallel(n_jobs=self.__max_njobs)(
                delayed(__thread)(matches[m]['pixel'], matches[m]['mz'], m,
                                  len(msiobj.pixels_indices))
                for m in tqdm(matches.keys()))
            for m in matches.keys():
                curr_res = spline_res.pop(0)
                residuals[m] = curr_res[0]
                inliers[m] = curr_res[1]
                inliers_px[m] = curr_res[2]
                inliers_pct[m] = curr_res[3]
                disp_ppm[m] = curr_res[4]
                del curr_res
        else:
            for m in tqdm(matches.keys()):
                shift_preds = \
                    self.__fit_spline_kde(pixels=matches[m]['pixel'],
                                          match_masses=matches[m]['mz'],
                                          ref_mass=m)
                # Find outliers
                residuals[m] = matches[m]['mz'] - m - shift_preds
                mad_resid = 1.4826 * np.median(np.abs(residuals[m]))
                inliers[m] = np.abs(residuals[m]) <= 2 * mad_resid
                inliers_px[m] = matches[m]['pixel'][inliers[m]]
                disp_ppm[m] = np.max(2 * mad_resid / (shift_preds + m) * 1e6)
                inliers_pct[m] = \
                    len(np.unique(inliers_px[m])) / len(msiobj.pixels_indices)

        self.ref_masses = \
            np.asarray(
                [m for m in inliers_pct.keys() if
                 (inliers_pct[m] * 100.0 >= self.min_pct) &
                 (np.abs(disp_ppm[m]) <= self.max_disp_ppm)], dtype=float)
        if len(self.ref_masses) == 0:
            raise RuntimeError('No reference masses found.')
        else:
            print('Using {} reference masses'.format(len(self.ref_masses)))

        # Plot image of selected references
        if self.__plot:
            print('Saving intensity images ...')
            del_all_files_dir(self.__plot_dir)
            for m in self.ref_masses:
                fig = plt.figure(dpi=150, figsize=(4, 3))
                ax = fig.add_subplot(111)
                img = np.zeros(np.prod(self.__im_dimxy))
                upx, c = np.unique(inliers_px[m], return_counts=True)
                sel_px = []
                sel_peaks = np.full(len(inliers[m]), False, dtype=bool)
                if np.any(c > 1):
                    for i, p in enumerate(upx):
                        idx = np.where(upx == p)[0]
                        if len(idx) == 1:
                            sel_px.append(p)
                            sel_peaks[matches[m]['pixel'] == p] = True
                        else:
                            # If more than one hit per pixel, take the one with
                            # the smallest residual
                            sel_px.append(
                                p[np.argmin(np.abs(residuals[m][idx]))])
                            sel_peaks[matches[m]['pixel'] == p] = True
                else:
                    sel_px = inliers_px[m].copy()
                    sel_peaks = inliers[m].copy()

                img[np.asarray(sel_px)] = matches[m]['intensity'][sel_peaks]
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                im = \
                    ax.imshow(img.reshape(self.__im_dimxy[::-1]),
                              interpolation='none',
                              cmap='inferno')
                ax.set_title('{} m/z'.format(m), fontsize=6)
                ax.set_xlabel('X', fontdict={'size': 6})
                ax.set_ylabel('Y', fontdict={'size': 6})
                ax.tick_params(labelsize=4)
                plt.colorbar(im, cax=cax)
                plt.tight_layout()
                plt.savefig(os.path.join(self.__plot_dir, '{}.png'.format(m)),
                            format='png')
                plt.close()

        print('Fitting time series ... ')
        if self.__parallel:
            shift_models = \
                Parallel(n_jobs=self.__max_njobs, backend='threading')(
                    delayed(__fit_time_series)(
                        matches[m]['pixel'][inliers[m]],
                        matches[m]['mz'][inliers[m]])
                    for m in tqdm(self.ref_masses))
            shift_models = dict(zip(self.ref_masses, shift_models))
        else:
            shift_models = {m: [] for m in self.ref_masses}
            for m in tqdm(self.ref_masses):
                shift_models[m] = \
                    __fit_time_series(matches[m]['pixel'][inliers[m]],
                                      matches[m]['mz'][inliers[m]])

        print('Predicting observed masses ...')
        mz_pred = \
            np.full((len(msiobj.pixels_indices), len(self.ref_masses)), np.nan)
        for i, m in enumerate(tqdm(self.ref_masses)):
            mz_pred[:, i] = \
                shift_models[m].predict(
                    msiobj.pixels_indices.reshape(-1, 1)).ravel()
        mass_theor = np.asarray(self.ref_masses)
        arg_sort = np.argsort(mass_theor)
        mass_theor = mass_theor[arg_sort]
        mz_pred = mz_pred[:, arg_sort]

        print('Recalibrating pixels\' spectra ...')

        if len(self.ref_masses) == 1:
            print('Using ratio')

        def __thread_px(x_, y_, xp_, i_):
            if len(self.ref_masses) == 1:
                if self.transform == 'sqrt':
                    x_ = np.sqrt(x_)
                    y_ = np.sqrt(y_)
                    xp_ = np.sqrt(xp_)
                B_ = y_ / x_
                yp_ = xp_ * B_
                if self.transform == 'sqrt':
                    yp_ = yp_ ** 2
            else:
                yp_, _, _ = self.__recal_pixel(x_, y_, xp_)
            msiobj.msdata[i_][:, 0] = yp_
            return None

        if self.__parallel:
            Parallel(n_jobs=self.__max_njobs, backend='threading')(
                delayed(__thread_px)(mz_pred[idx_, :], mass_theor.copy(),
                                     msiobj.msdata[idx_][:, 0].copy(), idx_)
                for idx_ in tqdm(range(len(msiobj.pixels_indices))))
        else:
            for i in tqdm(range(len(msiobj.pixels_indices))):
                x_fit = mz_pred[i, :]
                y_fit = mass_theor.copy()
                x_pred = msiobj.msdata[i].mz

                if len(self.ref_masses) == 1:
                    if self.transform == 'sqrt':
                        B = np.sqrt(y_fit) / np.sqrt(x_fit)
                        mz_corrected = np.sqrt(x_pred) * B
                        mz_corrected = mz_corrected ** 2
                    else:
                        B = y_fit / x_fit
                        mz_corrected = x_pred * B
                else:
                    mz_corrected, mdl, in_mask = \
                        self.__recal_pixel(x_fit=x_fit.copy(),
                                           y_fit=y_fit.copy(),
                                           x_pred=x_pred.copy())
                msiobj.msdata[i][:, 0] = mz_corrected

        return msiobj
