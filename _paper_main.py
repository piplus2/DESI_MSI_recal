#   Copyright 2018 by Paolo Inglese, National Phenome Centre, Imperial College
#   London
#   All rights reserved.
#   This file is part of DESI-MSI recalibration, and is released under the
#   "MIT License Agreement".
#   Please see the LICENSE file that should have been included as part of this
#   package.


import os
import os.path

import numpy as np
import pandas as pd
from tqdm import tqdm

from tools.functions import gen_ref_list, search_ref_masses, fit_spline_kde, \
    fit_shift_model, recal_pixel, del_all_files_dir
from tools.msi import MSI
from tools.plot_style import set_mpl_params_mod

import matplotlib.pyplot as plt


def make_results_dir(root_dir: str, method: str, use_delta: bool) -> str:
    _dir = os.path.join(root_dir, '_RESULTS',
                        method + ('_delta_mz' if use_delta else '_obs_mz'))
    if not os.path.isdir(_dir):
        os.makedirs(_dir)
    return _dir


set_mpl_params_mod()

ROOT_DIR = os.path.join('E:', 'CALIB_PAPER', 'DATA')
DATASET = 'TOF'
METHOD = 'poly'

if DATASET == 'TOF':
    MIN_PERC = 75.0
    MAX_POLY_DEGREE = 5
    MAX_TOL = 100.0
    TRANSFORM = 'sqrt'
    MAX_DISPERSION = 10.0
elif DATASET == 'ORBITRAP':
    MIN_PERC = 75.0
    MAX_POLY_DEGREE = 1
    MAX_TOL = 20.0
    TRANSFORM = None
    MAX_DISPERSION = 10.0
else:
    raise ValueError('Invalid dataset type.')

msi_datasets = pd.read_csv(os.path.join(ROOT_DIR, DATASET, 'meta.csv'),
                           index_col=0)
msi_datasets = msi_datasets[msi_datasets['process'] == 'yes']

str_method = {'spline': 's', 'poly': 'p', 'gam': 'g'}
str_method = str_method[METHOD]

for index in msi_datasets.index:
    run = msi_datasets.loc[index, :]

    results_dir = make_results_dir(run['dir'], method=METHOD, use_delta=False)

    print('MSI {}/{}: {}'.format(index + 1, msi_datasets.shape[0], run['dir']))

    meta = {'ion_mode': run['ion_mode']}
    msi = MSI(imzml=os.path.join(run['dir'], run['fname']), meta=None)
    msi._MSI__meta = meta
    roi = np.loadtxt(os.path.join(run['dir'], 'roi.csv'), delimiter=',')

    if not np.all(roi.shape == msi.dim_xy[::-1]):
        raise ValueError('ROI has incompatible dimensions.')
    print('Num. ROI pixels = {}'.format(int(np.sum(roi))))

    # Remove non-ROI pixels
    outpx = np.where(roi.ravel() == 0)[0]
    delpx = np.where(np.isin(msi.pixels_indices, outpx))[0]
    delpx = np.sort(delpx)
    msi.del_pixel(list(delpx))
    msi.to_imzml(os.path.join(run['dir'], run['tissue'] + '_' + run[
        'ion_mode'] + '_0step.imzML'))

    # RECALIBRATION ---------------------------

    ref_masses = gen_ref_list(ion_mode=run['ion_mode'], verbose=True)

    print('Searching lock masses within {} ppm ...'.format(
        np.round(MAX_TOL, 2)))
    matches = search_ref_masses(msiobj=msi, ref_masses=ref_masses,
                                max_tolerance=MAX_TOL, top_n=-1)

    print('Removing hits found in less than {} % of ROI pixels ...'.format(
        MIN_PERC))
    matches = {m: matches[m] for m in matches.keys() if
               len(np.unique(matches[m]['pixel'])) / len(
                   msi.pixels_indices) * 100.0 >= MIN_PERC}

    print('Num. lock masses with coverage >= {} % = {}'.format(
        np.round(MIN_PERC, 2), len(matches)))

    res_matches_dir = os.path.join(results_dir, 'matches')
    if not os.path.isdir(res_matches_dir):
        os.makedirs(res_matches_dir)
    else:
        print('Deleting old results in {} ...'.format(res_matches_dir))
        del_all_files_dir(res_matches_dir)
    for m in matches.keys():
        df = pd.DataFrame.from_dict(matches[m])
        df.to_csv(os.path.join(res_matches_dir,
                               'matches_orig_{}.csv'.format(np.round(m, 4))))

    print('Fitting KDE ...')
    inliers = {m: [] for m in matches.keys()}
    inliers_px = {m: [] for m in matches.keys()}
    inliers_pct = {m: [] for m in matches.keys()}
    disp_ppm = {m: [] for m in matches.keys()}
    residuals = {m: [] for m in matches.keys()}

    kde_dir = os.path.join(run['dir'], 'kde')
    if not os.path.isdir(kde_dir):
        os.makedirs(kde_dir)
    else:
        del_all_files_dir(kde_dir)

    for m in tqdm(matches.keys()):
        shift_preds, use_kde = fit_spline_kde(
            pixels=matches[m]['pixel'], match_masses=matches[m]['mz'],
            ref_mass=m)
        # Find outliers
        residuals[m] = matches[m]['mz'] - m - shift_preds
        mad_resid = 1.4826 * np.median(np.abs(residuals[m]))
        inliers[m] = np.abs(residuals[m]) <= 2 * mad_resid
        inliers_px[m] = matches[m]['pixel'][inliers[m]]
        disp_ppm[m] = np.max(2 * mad_resid / (shift_preds + m) * 1e6)
        inliers_pct[m] = len(np.unique(inliers_px[m])) / len(msi.pixels_indices)

    # Select by coverage percentage and dispersion
    sel_refs = np.asarray([m for m in inliers_pct.keys() if
                           (inliers_pct[m] * 100.0 >= MIN_PERC) & (
                                   np.abs(disp_ppm[m]) <=
                                   MAX_DISPERSION)], dtype=float)
    sel_refs = np.unique(sel_refs)
    print('Num. candidate references = {}'.format(len(sel_refs)))
    np.savetxt(fname=os.path.join(results_dir, 'references.csv'), X=sel_refs,
               fmt='%f')

    shift_models = {m: [] for m in sel_refs}
    for m in sel_refs:
        shift_models[m] = fit_shift_model(matches[m]['pixel'][inliers[m]],
                                          matches[m]['mz'][inliers[m]],
                                          max_degree=5,  # THIS IS FIXED
                                          model='ols', error='mse')

    print('Recalibrating pixels ...')

    # Predict reference shift in all pixels
    mz_pred = np.full((len(msi.pixels_indices), len(sel_refs)), np.nan)
    for i, m in enumerate(sel_refs):
        mz_pred[:, i] = shift_models[m]['best_model'].predict(
            msi.pixels_indices.reshape(-1, 1)).ravel()
    mass_theor = np.asarray(sel_refs)
    arg_sort = np.argsort(mass_theor)
    mass_theor = mass_theor[arg_sort]

    plot_px = np.random.choice(len(msi.pixels_indices), size=50, replace=False)
    plot_px_dir = os.path.join(run['dir'], 'plot_recal_px')
    if not os.path.isdir(plot_px_dir):
        os.makedirs(plot_px_dir)
    else:
        del_all_files_dir(plot_px_dir)

    model_params = np.full((len(msi.pixels_indices), MAX_POLY_DEGREE + 1),
                           np.nan, dtype=float)

    recal_masses = np.zeros((len(sel_refs), 2))
    recal_masses[:, 0] = sel_refs
    for i in tqdm(range(len(msi.pixels_indices))):
        x_fit = mz_pred[i, arg_sort]
        y_fit = mass_theor.copy()
        x_pred = msi.msdata[i][:, 0]
        mz_corrected, mdl, in_mask = recal_pixel(
            x_fit=x_fit.copy(), y_fit=y_fit.copy(), x_pred=x_pred.copy(),
            transform=TRANSFORM, max_degree=MAX_POLY_DEGREE)
        msi.msdata[i][:, 0] = mz_corrected

        # Add +1 to used masses for fit
        recal_masses[in_mask, 1] += 1

        # Save model coefficients
        model_params[i, :len(mdl[1].results_.params)] = mdl[
            1].results_.params.ravel()

        # Plot delta m/z vs m/z
        if i in plot_px:
            if TRANSFORM == 'sqrt':
                x_pred_ = np.sqrt(x_pred.copy())
            else:
                x_pred_ = x_pred.copy()
            pred = mdl[1].results_.get_prediction(
                mdl[0].transform(x_pred_.reshape(-1, 1)))
            pred_df = pred.summary_frame()

            fig = plt.figure(figsize=(4, 3), dpi=300)
            ax = fig.add_subplot(111)
            if TRANSFORM == 'sqrt':
                ax.plot(x_pred, pred_df['mean'] ** 2 - x_pred, c='black',
                        label='fit')
                ax.fill_between(
                    x=x_pred, y1=pred_df['mean_ci_upper'] ** 2 - x_pred,
                    y2=pred_df['mean_ci_lower'] ** 2 - x_pred,
                    color='lightgrey')
            else:
                ax.plot(x_pred, pred_df['mean'] - x_pred, c='black',
                        label='fit')
                ax.fill_between(
                    x=x_pred, y1=pred_df['mean_ci_upper'] - x_pred,
                    y2=pred_df['mean_ci_lower'] - x_pred, color='lightgrey')
            ax.scatter(x_fit[~in_mask], y_fit[~in_mask] - x_fit[~in_mask],
                       facecolors='none', edgecolors='red', marker='s',
                       label='outlier', zorder=np.inf, s=3, linewidths=0.5)
            ax.scatter(x_fit[in_mask], y_fit[in_mask] - x_fit[in_mask],
                       facecolors='none', edgecolors='royalblue', marker='o',
                       label='inlier', zorder=np.inf, s=3, linewidths=0.5)
            ax.set_xlabel(r'$m^{(obs)}$' + ' (m/z)',
                          fontdict={'weight': 'bold'})
            ax.set_ylabel(r'$\Delta$' + ' (m/z)', fontdict={'weight': 'bold'})
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
                      fancybox=True, shadow=True, ncol=5)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_px_dir, 'px_' + str(i + 1) + '.pdf'),
                        format='pdf')
            plt.close()

    print('Saving models details ...')
    df_coefs = pd.DataFrame(data=model_params,
                            columns=['beta_' + str(i) for i in
                                     range(model_params.shape[1])])
    df_coefs.to_csv(os.path.join(results_dir, 'recal_coefs.csv'))
    recal_masses[:, 1] /= len(msi.pixels_indices)
    df_recal_masses = pd.DataFrame(data=recal_masses,
                                   columns=['mass', 'pct'])
    df_recal_masses.to_csv(os.path.join(results_dir, 'inlier_masses.csv'))

    print('Saving recalibrated ROI imzML ...')
    msi.to_imzml(output_path=os.path.join(run['dir'], 'recal_peaks.imzML'))
