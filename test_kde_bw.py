#   Copyright 2018 by Paolo Inglese, National Phenome Centre, Imperial College
#   London
#   All rights reserved.
#   This file is part of DESI-MSI recalibration, and is released under the
#   "MIT License Agreement".
#   Please see the LICENSE file that should have been included as part of this
#   package.


import argparse
import os
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

from tools.functions import gen_ref_list, search_ref_masses
from tools.msi import MSI
from KDEpy import FFTKDE
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import UnivariateSpline
from sklearn.model_selection import KFold
import pygam
import numbers
from joblib import Parallel, delayed
from tqdm import tqdm


def __parse_arg():
    parser_ = argparse.ArgumentParser(description='DESI-MSI recalibration.'
                                                  'Test time series parameters')
    parser_.add_argument('input', type=str, help='Input imzML file.')
    parser_.add_argument('roi', type=str,
                         help='Sample ROI mask CSV file. If set equal to '
                              '\'full\', the entire image is analyzed.')
    parser_.add_argument('--analyzer', choices=['tof', 'orbitrap'],
                         help='MS analyzer.')
    parser_.add_argument('--ion-mode', choices=['pos', 'neg'], required=True,
                         help='ES Polarization mode.')
    parser_.add_argument('--search-tol', default='auto',
                         help='Search tolerance expressed in ppm. If \'auto\', '
                              'default value for MS analyzer is used.')
    parser_.add_argument('--min-coverage', default=75.0, type=float,
                         help='Min. coverage percentage for hits filtering '
                              '(default=75.0).')
    parser_.add_argument('--num-peaks', default=3, type=int,
                         help='Number of tested masses (default=3).')
    return parser_


def set_params_dict(args_) -> Dict:
    default_max_tol = {'orbitrap': 20.0, 'tof': 100.0}
    params_ = {
        'input': args_.input,
        'roi': args_.roi,
        'analyzer': args_.analyzer,
        'ion_mode': 'ES-' if args_.ion_mode == 'neg' else 'ES+',
        'max_tol': args_.search_tol if args_.search_tol != 'auto' else
        default_max_tol[args_.analyzer],
        'min_cov': args_.min_coverage,
    }
    return params_


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
        error.append(np.mean((yhat - y[tst]) ** 2))
    return np.mean(error)


def fit_spline_kde(pixels, match_masses, ref_mass, model, kde_bw, smooth):

    grid_size = int(2**10)

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
        use_gam = False
        use_kde = False
    else:
        use_kde = True
        data = np.c_[pixels, match_masses - ref_mass].astype(np.float64)
        data = data.astype(float)
        scaler = MinMaxScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        if kde_bw == 'silverman':
            bandwidth = 2.576 * data.std(ddof=1) * data.shape[0] ** (-1 / 5)
        else:
            bandwidth = float(kde_bw)
        kde = FFTKDE(bw=bandwidth, kernel='tri').fit(data)
        grid, points = kde.evaluate(grid_size)
        grid_mask = np.all((grid >= 0) & (grid <= 1), axis=1)
        grid = grid[grid_mask, :]
        x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
        z = points[grid_mask]
        z = z.reshape((len(x), len(y))).T
        xyi = scaler.inverse_transform(np.c_[x, y])
        xmax_kde, ymax_kde = \
            find_kde_max(x=xyi[:, 0], y=xyi[:, 1], kde_values=z,
                         remove_zeros=True)
        if np.var(xmax_kde) == 0:
            use_gam = False
            mdl = UnivariateSpline(x=xmax_kde, y=ymax_kde)
        else:
            if model == 'spline':
                use_gam = False
                if not isinstance(smooth,
                                  numbers.Number) and smooth == 'cv':
                    s_vals = np.logspace(-3, -1, 20)
                    mse = []
                    for s_ in s_vals:
                        mse.append(
                            ts_cv(xmax_kde.reshape(-1, 1),
                                  ymax_kde.reshape(-1, ), s_))
                    s_value = s_vals[np.argmin(mse)]
                else:
                    s_value = smooth
                mdl = \
                    UnivariateSpline(x=xmax_kde.reshape(-1, 1),
                                     y=ymax_kde.reshape(-1, ),
                                     s=s_value)
            elif model == 'gam':
                use_gam = True
                mdl = pygam.LinearGAM(pygam.s(0, n_splines=5))
                mdl.gridsearch(X=xmax_kde.reshape(-1, 1), y=ymax_kde,
                               progress=False)
            else:
                raise ValueError('Invalid model.')

    if use_gam:
        yhat = mdl.predict(pixels.reshape(-1, 1)).ravel()
    else:
        yhat = mdl(pixels)

    return yhat, use_kde


def kde_regress(msiobj, matches, model, kde_bw, smooth):
    max_njobs = 5
    parallel = False

    def __thread(x_, y_, m_, npx_):
        yhat_, _ = fit_spline_kde(pixels=x_, match_masses=y_, ref_mass=m_,
                                  model=model, kde_bw=kde_bw, smooth=smooth)
        # Find outliers
        res_ = y_ - m_ - yhat_
        mad_ = 1.4826 * np.median(np.abs(res_))
        inliers_ = np.abs(res_) <= 2 * mad_
        inliers_px_ = x_[inliers_]
        disp_ = np.max(2 * mad_ / (yhat_ + m_)) * 1e6
        pct_ = len(np.unique(inliers_px_)) / npx_
        return res_, inliers_, inliers_px_, pct_, disp_

    print('Finding outliers in time series ...')
    inliers = {m: [] for m in matches.keys()}
    inliers_px = {m: [] for m in matches.keys()}
    inliers_pct = {m: [] for m in matches.keys()}
    disp_ppm = {m: [] for m in matches.keys()}
    residuals = {m: [] for m in matches.keys()}

    if parallel:
        spline_res = Parallel(n_jobs=max_njobs)(
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
            shift_preds, use_kde = \
                fit_spline_kde(pixels=matches[m]['pixel'],
                               match_masses=matches[m]['mz'],
                               ref_mass=m, model=model, kde_bw=kde_bw,
                               smooth=smooth)
            # Find outliers
            residuals[m] = matches[m]['mz'] - m - shift_preds
            mad_resid = 1.4826 * np.median(np.abs(residuals[m]))
            inliers[m] = np.abs(residuals[m]) <= 2 * mad_resid
            inliers_px[m] = matches[m]['pixel'][inliers[m]]
            disp_ppm[m] = np.max(2 * mad_resid / (shift_preds + m) * 1e6)
            inliers_pct[m] = len(
                np.unique(inliers_px[m])) / len(msiobj.pixels_indices)

    return residuals, inliers, inliers_px, inliers_pct, disp_ppm


def fit_time_series(x, y):
    model = pygam.LinearGAM(pygam.s(0))
    model.gridsearch(X=x.reshape(-1, 1), y=y, progress=False)
    return model


def main():
    parser = __parse_arg()
    args = parser.parse_args()
    params = set_params_dict(args)

    # Load MSI
    msi = MSI(imzml=args.input, meta=params)
    # Load ROI
    if args.roi != 'full':
        roi = np.loadtxt(args.roi, delimiter=',')
        if not np.all(roi.shape == msi.dim_xy[::-1]):
            raise ValueError('ROI has incompatible dimensions.')
        print('Num. ROI pixels = {}'.format(int(np.sum(roi))))
        # Remove non-ROI pixels
        outpx = np.where(roi.ravel() == 0)[0]
        delpx = np.where(np.isin(msi.pixels_indices, outpx))[0]
        delpx = np.sort(delpx)
        msi.del_pixel(list(delpx))

    plots_dir = os.path.join(os.path.dirname(args.input), '_calib')
    print('Plots will be saved in {}'.format(plots_dir))
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)

    ref_masses = gen_ref_list(ion_mode=params['ion_mode'], verbose=True)

    print('Searching lock masses within {} ppm ...'.format(
        np.round(params['max_tol'], 2)))
    matches = search_ref_masses(msiobj=msi, ref_masses=ref_masses,
                                max_tolerance=params['max_tol'],
                                coverage=params['min_cov'])
    print('Removing hits found in less than {} % of ROI pixels ...'.format(
        params['min_cov']))
    matches = {m: matches[m] for m in matches.keys() if
               len(np.unique(matches[m]['pixel'])) / len(
                   msi.pixels_indices) * 100.0 >= params['min_cov']}

    # Select the reference with the smallest median absolute residual
    mad = \
        np.asarray([np.median(np.abs(matches[m]['mz'] - m))
                    for m in matches.keys()], dtype=float)
    sel_ref = [list(matches.keys())[i]
               for i in np.argsort(mad)[:np.min([args.num_peaks, len(mad)])]]

    for mass in sel_ref:
        sel_matches = {m: matches[m] for m in [mass]}
        print('Selected reference: {} m/z'.format(np.round(mass, 4)))

        # Test various bandwidth
        TEST_BW = np.logspace(-3, -1, 3)
        SMOOTH = np.logspace(-3, -1, 3) * 2

        fig, ax = plt.subplots(4, 3, figsize=(8, 6), dpi=150, sharex='all',
                               sharey='all')
        ax = ax.flatten()
        for ax_ in ax:
            ax_.set_visible(False)
        k = 0
        for bw in TEST_BW:
            for smooth in SMOOTH:
                print('Testing bw = {}, smooth = {} ...'.format(bw, smooth))
                results = kde_regress(msi, sel_matches, 'gam', bw, smooth)
                ax[k].scatter(sel_matches[mass]['pixel'][~results[1][mass]],
                              sel_matches[mass]['mz'][~results[1][mass]] -
                              mass,
                              label='Outliers', c='red', s=1)
                ax[k].scatter(
                    sel_matches[mass]['pixel'][results[1][mass]],
                    sel_matches[mass]['mz'][results[1][mass]] - mass,
                    label='Inliers', c='blue', s=1)
                ax[k].set_xlabel('Pixel order')
                ax[k].set_ylabel(r'$M^{\#} - M$')
                ax[k].set_title('bw = {}, smooth = {} \n disp = {} ppm'.format(
                    bw, smooth, np.round(results[4][mass], 2)))
                ax[k].set_visible(True)
                k += 1

        results = kde_regress(msi, matches=sel_matches, model='gam', smooth='cv', kde_bw='silverman')
        ax[k].scatter(sel_matches[mass]['pixel'][~results[1][mass]],
                      sel_matches[mass]['mz'][~results[1][mass]] -
                      mass,
                      label='Outliers', c='red', s=1)
        ax[k].scatter(
            sel_matches[mass]['pixel'][results[1][mass]],
            sel_matches[mass]['mz'][results[1][mass]] - mass,
            label='Inliers', c='blue', s=1)
        ax[k].set_xlabel('Pixel order')
        ax[k].set_ylabel(r'$M^{\#} - M$')
        ax[k].set_title('bw = {}, smooth = {} \n disp = {} ppm'.format(
            'silverman', 'cv', np.round(results[4][mass], 2)))
        ax[k].set_visible(True)

        plt.tight_layout()
        plt.savefig(
            os.path.join(plots_dir, 'test_params_' + str(np.round(mass, 4))
                         + '.png'))
        plt.close()


if __name__ == '__main__':
    main()
