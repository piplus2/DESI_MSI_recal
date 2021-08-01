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
import warnings

from tools.functions import gen_ref_list, search_ref_masses, kde_regress, \
    fit_shift_model, recal_pixel, del_all_files_dir, filter_roi_px
from tools.msi import MSI
from tools.plot_style import set_mpl_params_mod
from scipy.stats import median_abs_deviation

import matplotlib.pyplot as plt


def make_results_dir(root_dir: str, min_pct: float) -> str:
    _dir = os.path.join(root_dir, '_RESULTS', 'recal_' + str(min_pct))
    if not os.path.isdir(_dir):
        os.makedirs(_dir)
    return _dir


set_mpl_params_mod()

ROOT_DIR = os.path.join('E:', 'CALIB_PAPER', 'DATA')
DATASET = 'TOF'

if DATASET == 'TOF':
    MAX_POLY_DEGREE = 5
    MAX_TOL = 100.0
    TRANSFORM = 'sqrt'
    MAX_DISPERSION = 10.0
elif DATASET == 'ORBITRAP':
    MAX_POLY_DEGREE = 1
    MAX_TOL = 20.0
    TRANSFORM = None
    MAX_DISPERSION = 10.0
else:
    raise ValueError('Invalid dataset type.')
MIN_N_PX = 10


msi_datasets = pd.read_csv(os.path.join(ROOT_DIR, DATASET, 'meta.csv'),
                           index_col=0)
msi_datasets = msi_datasets[msi_datasets['process'] == 'yes']

for index in msi_datasets.index:
    run = msi_datasets.loc[index, :]

    results_dir = os.path.join(run['dir'], '_RESULTS', 'analysis_pct_refs')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    else:
        del_all_files_dir(results_dir)

    print('Loading DESI-MSI and search ref. masses ...')
    print('MSI {}/{}: {}'.format(index + 1, msi_datasets.shape[0],
                                 run['dir']))
    meta = {'ion_mode': run['ion_mode']}
    msi = MSI(imzml=os.path.join(run['dir'], run['fname']), meta=None)
    msi._MSI__meta = meta
    # Keep ROI pixels
    roi = np.loadtxt(os.path.join(run['dir'], 'roi.csv'), delimiter=',')
    msi = filter_roi_px(msi, roi)
    print('Num. ROI pixels = {}'.format(int(np.sum(roi))))
    # Do the full search
    print('Searching lock masses within {} ppm ...'.format(
        np.round(MAX_TOL, 2)))
    ref_masses = gen_ref_list(ion_mode=run['ion_mode'], verbose=True)
    matches = search_ref_masses(msiobj=msi, ref_masses=ref_masses,
                                max_tolerance=MAX_TOL, top_n=-1)
    matches = {m: matches[m] for m in matches.keys()
               if len(matches[m]['pixel'] > MIN_N_PX)}

    # RECALIBRATION ----------------------------------------------------

    print('Fitting KDE ...')

    kde_results = kde_regress(msiobj=msi, search_results=matches)

    # Select by coverage percentage and dispersion
    sel_refs = \
        np.asarray([m for m in kde_results.keys() if
                    np.abs(kde_results[m]['dispersion']) <= MAX_DISPERSION],
                   dtype=float)
    sel_refs = np.unique(sel_refs)
    print('Num. candidate references = {}'.format(len(sel_refs)))

    shift_models = {m: [] for m in sel_refs}
    for m in sel_refs:
        shift_models[m] = fit_shift_model(
            matches[m]['pixel'][kde_results[m]['is_inlier']],
            matches[m]['mz'][kde_results[m]['is_inlier']],
            max_degree=5,  # THIS IS FIXED
            model='ols', error='mse')

    print('Recalibrating pixels ...')

    # Predict reference shift in all pixels
    outdir = os.path.join(results_dir, 'trend_preds')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    else:
        del_all_files_dir(outdir)
    mz_pred = np.full((len(msi.pixels_indices), len(sel_refs)), np.nan)
    for i, m in enumerate(sel_refs):
        preds = \
            shift_models[m]['best_model'][1].results_.get_prediction(
                shift_models[m]['best_model'][0].transform(
                    msi.pixels_indices.reshape(-1, 1)))
        preds = preds.summary_frame()
        mz_pred[:, i] = preds['mean'].values
        pred_err_ppm = \
            (preds['mean_ci_upper'] - preds['mean_ci_lower']) / \
            preds['mean'] * 1e6
        if np.any(np.abs(pred_err_ppm) > 1):
            warnings.warn(
                'Ref. mass {}. Some predictions have error > '
                '1 ppm.'.format(np.round(m, 4)))
        preds['err_ppm'] = pred_err_ppm
        preds['greater_than_1ppm'] = np.abs(pred_err_ppm) > 1
        # Save predictions
        preds.to_csv(os.path.join(outdir, str(np.round(m, 4)) + '.csv'))
        del preds
    del outdir

    mass_theor = np.asarray(sel_refs)
    arg_sort = np.argsort(mass_theor)
    mass_theor = mass_theor[arg_sort]
    mz_pred = mz_pred[:, arg_sort]

    plot_px = np.random.choice(len(msi.pixels_indices), size=50, replace=False)
    plot_px_dir = os.path.join(results_dir, 'plot_recal_px')
    if not os.path.isdir(plot_px_dir):
        os.makedirs(plot_px_dir)
    else:
        del_all_files_dir(plot_px_dir)

    model_params = np.full((len(msi.pixels_indices), MAX_POLY_DEGREE + 1),
                           np.nan, dtype=float)

    recal_masses = np.zeros((len(sel_refs), 2))
    recal_masses[:, 0] = sel_refs
    for i in tqdm(range(len(msi.pixels_indices))):
        x_fit = mz_pred[i, :].copy()
        y_fit = mass_theor.copy()
        x_pred = msi.msdata[i][:, 0].copy()
        mz_corrected, mdl, in_mask = recal_pixel(
            x_fit=x_fit, y_fit=y_fit, x_pred=x_pred,
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
                ax.plot(x_pred, pred_df['mean'] ** 2 - x_pred,
                        c='black', label='fit')
                ax.fill_between(
                    x=x_pred,
                    y1=pred_df['mean_ci_upper'] ** 2 - x_pred,
                    y2=pred_df['mean_ci_lower'] ** 2 - x_pred,
                    color='lightgrey')
            else:
                ax.plot(x_pred, pred_df['mean'] - x_pred, c='black',
                        label='fit')
                ax.fill_between(
                    x=x_pred, y1=pred_df['mean_ci_upper'] - x_pred,
                    y2=pred_df['mean_ci_lower'] - x_pred,
                    color='lightgrey')
            ax.scatter(x_fit[~in_mask],
                       y_fit[~in_mask] - x_fit[~in_mask],
                       facecolors='none', edgecolors='red',
                       marker='s', label='outlier', zorder=np.inf,
                       s=3, linewidths=0.5)
            ax.scatter(x_fit[in_mask],
                       y_fit[in_mask] - x_fit[in_mask],
                       facecolors='none', edgecolors='royalblue',
                       marker='o', label='inlier', zorder=np.inf,
                       s=3, linewidths=0.5)
            ax.set_xlabel(r'$m^{(obs)}$' + ' (m/z)',
                          fontdict={'weight': 'bold'})
            ax.set_ylabel(r'$\Delta$' + ' (m/z)',
                          fontdict={'weight': 'bold'})
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
                      fancybox=True, shadow=True, ncol=5)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_px_dir, 'px_' + str(i + 1) +
                                     '.pdf'), format='pdf')
            plt.close()
    del plot_px_dir

    pd.DataFrame.from_dict({
        'mass': recal_masses[:, 0],
        'in_model_pct': recal_masses[:, 1] / len(msi.pixels_indices),
        'tseries_pct': [kde_results[m]['inlier_pct'] for m in sel_refs]
    }).to_csv(os.path.join(results_dir, 'references.csv'))

    print('Saving models details ...')
    pd.DataFrame(data=model_params,
                 columns=['beta_' + str(i) for i in
                          range(model_params.shape[1])]).to_csv(
        os.path.join(results_dir, 'recal_coefs.csv'))

    print('Saving recalibrated ROI imzML ...')
    msi.to_imzml(output_path=os.path.join(results_dir, 'recal_peaks.imzML'))
