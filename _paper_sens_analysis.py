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

from tools.functions import gen_ref_list, search_ref_masses, fit_spline_kde, \
    fit_shift_model, recal_pixel, del_all_files_dir
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

N_REPS = 10
MAX_PCTS = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 75.0, 80.0, 85.0, 90.0]

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

msi_datasets = pd.read_csv(os.path.join(ROOT_DIR, DATASET, 'meta.csv'),
                           index_col=0)
msi_datasets = msi_datasets[msi_datasets['process'] == 'yes']

for index in msi_datasets.index:
    run = msi_datasets.loc[index, :]
    roi = np.loadtxt(os.path.join(run['dir'], 'roi.csv'), delimiter=',')

    print('Loading DESI-MSI and search ref. masses ...')
    ref_masses = gen_ref_list(ion_mode=run['ion_mode'], verbose=True)
    meta = {'ion_mode': run['ion_mode']}
    msi = MSI(imzml=os.path.join(run['dir'], run['fname']), meta=None)
    msi._MSI__meta = meta
    print('MSI {}/{}: {}'.format(index + 1, msi_datasets.shape[0],
                                 run['dir']))
    outpx = np.where(roi.ravel() == 0)[0]
    delpx = np.where(np.isin(msi.pixels_indices, outpx))[0]
    delpx = np.sort(delpx)
    msi.del_pixel(list(delpx))
    # Do the full search
    print('Searching lock masses within {} ppm ...'.format(
        np.round(MAX_TOL, 2)))
    full_matches = search_ref_masses(msiobj=msi, ref_masses=ref_masses,
                                     max_tolerance=MAX_TOL, top_n=-1)

    for MAX_PERC in MAX_PCTS:

        print('Max. coverage = {}%'.format(MAX_PERC))
        TRAIN_SIZE = int(np.floor(MAX_PERC * len(msi.msdata) / 100))
        TEST_SIZE = int(np.floor(0.05 * len(msi.msdata)))
        MIN_SIZE = int(np.ceil(0.95 * len(msi.msdata)))

        results_dir = os.path.join(run['dir'], '_RESULTS',
                                   'max_pct_' + str(MAX_PERC))
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        else:
            del_all_files_dir(results_dir)

        meta = {'ion_mode': run['ion_mode']}
        msi = MSI(imzml=os.path.join(run['dir'], run['fname']), meta=None)
        msi._MSI__meta = meta

        if not np.all(roi.shape == msi.dim_xy[::-1]):
            raise ValueError('ROI has incompatible dimensions.')
        print('Num. ROI pixels = {}'.format(int(np.sum(roi))))

        # Remove non-ROI pixels
        outpx = np.where(roi.ravel() == 0)[0]
        delpx = np.where(np.isin(msi.pixels_indices, outpx))[0]
        delpx = np.sort(delpx)
        msi.del_pixel(list(delpx))

        # Sample matched pixels --------------------------------------------

        # Randomly select train and test pixels. Test pixels always represent
        # 5% of the ROI
        def sample_matches(mt, train_size, test_size):
            unique_px = np.unique(mt['pixel'])
            train_px = np.random.choice(unique_px, size=int(train_size),
                                        replace=False)
            test_px = np.random.choice(unique_px[~np.isin(unique_px, train_px)],
                                       size=int(test_size), replace=False)

            train_mask = np.isin(mt['pixel'], train_px)
            test_mask = np.isin(mt['pixel'], test_px)
            mtrain = {x: [] for x in mt.keys()}
            mtest = {x: [] for x in mt.keys()}
            for m_ in mt.keys():
                mtrain[m_] = mt[m_][train_mask]
                mtest[m_] = mt[m_][test_mask]
            return {'train': mtrain, 'test': mtest}

        pred_err_rep_med = np.full(N_REPS, np.nan)
        pred_err_rep_mad = np.full(N_REPS, np.nan)
        for rep in range(N_REPS):

            print('Replicate: {}'.format(rep + 1))

            matches = {
                m: sample_matches(full_matches[m], train_size=TRAIN_SIZE,
                                  test_size=TEST_SIZE)
                for m in full_matches.keys()
                if len(np.unique(full_matches[m]['pixel'])) >= MIN_SIZE}
    
            # RECALIBRATION ----------------------------------------------------
    
            print('Num. lock masses = {}'.format(len(matches)))
            if len(matches) < 10:
                print('Too few references. Skipping.')
                continue
    
            print('Fitting KDE ...')
            inliers = {m: [] for m in matches.keys()}
            inliers_px = {m: [] for m in matches.keys()}
            inliers_pct = {m: [] for m in matches.keys()}
            disp_ppm = {m: [] for m in matches.keys()}
            residuals = {m: [] for m in matches.keys()}
    
            for m in tqdm(matches.keys()):
                shift_preds, use_kde = fit_spline_kde(
                    pixels=matches[m]['train']['pixel'],
                    match_masses=matches[m]['train']['mz'],
                    ref_mass=m)
                # Find outliers
                residuals[m] = matches[m]['train']['mz'] - m - shift_preds
                mad_resid = 1.4826 * np.median(np.abs(residuals[m]))
                inliers[m] = np.abs(residuals[m]) <= 2 * mad_resid
                inliers_px[m] = matches[m]['train']['pixel'][inliers[m]]
                disp_ppm[m] = np.max(2 * mad_resid / (shift_preds + m) * 1e6)
                inliers_pct[m] = len(np.unique(inliers_px[m])) / \
                    len(msi.pixels_indices)
    
            # Select by coverage percentage and dispersion
            sel_refs = np.asarray([m for m in inliers_pct.keys() if
                                   np.abs(disp_ppm[m]) <= MAX_DISPERSION],
                                  dtype=float)
            sel_refs = np.unique(sel_refs)
            print('Num. candidate references = {}'.format(len(sel_refs)))
            # noinspection PyTypeChecker
            np.savetxt(fname=os.path.join(results_dir,
                                          'references_' + str(rep) + '.csv'),
                       X=sel_refs, fmt='%f')
    
            shift_models = {m: [] for m in sel_refs}
            for m in sel_refs:
                shift_models[m] = fit_shift_model(
                    matches[m]['train']['pixel'][inliers[m]],
                    matches[m]['train']['mz'][inliers[m]],
                    max_degree=5,  # THIS IS FIXED
                    model='ols', error='mse')

            print('Predicting masses time series ...')
            pred_err = np.full(len(sel_refs), np.nan, dtype=float)
            for j, m in enumerate(sel_refs):
                pred_mass = shift_models[m]['best_model'].predict(
                    np.unique(matches[m]['test']['pixel']).reshape(-1, 1))
                u, s = np.unique(matches[m]['test']['pixel'],
                                 return_counts=True)
    
                pred_err_px = np.full(len(pred_mass), np.nan)
                for k, (px, kounts) in enumerate(zip(u, s)):
                    true_val = matches[m]['test']['mz'][
                        matches[m]['test']['pixel'] == px]
                    if kounts > 1:
                        pred_err_px[k] = np.mean(
                            np.abs(true_val - pred_mass[k]))
                    else:
                        pred_err_px[k] = np.abs(true_val - pred_mass[k])
                pred_err[j] = np.mean(pred_err_px)
                del pred_err_px
                
            df_pred_err = pd.DataFrame(data=np.c_[sel_refs, pred_err],
                                       columns=['mz', 'abs_err'])
            df_pred_err.to_csv(
                os.path.join(results_dir, 'abs_err_time_series_rep_' + 
                             str(rep) + '.csv'))
                
            pred_err_rep_med[rep] = np.median(pred_err)
            pred_err_rep_mad[rep] = median_abs_deviation(pred_err,
                                                         scale='normal')

        pd.DataFrame.from_dict({
            'repeat': np.arange(N_REPS),
            'med_abs_err': pred_err_rep_med,
            'mad_err': pred_err_rep_mad
        }).to_csv(os.path.join(results_dir, 'summary_time_series_err.csv'))

            # print('Recalibrating pixels ...')
            # 
            # # Predict reference shift in all pixels
            # outdir = os.path.join(results_dir, 'trend_preds')
            # if not os.path.isdir(outdir):
            #     os.makedirs(outdir)
            # else:
            #     del_all_files_dir(outdir)
            # mz_pred = np.full((len(msi.pixels_indices), len(sel_refs)), np.nan)
            # for i, m in enumerate(sel_refs):
            #     preds = shift_models[m]['best_model'][
            #         1].results_.get_prediction(
            #         shift_models[m]['best_model'][0].transform(
            #             msi.pixels_indices.reshape(-1, 1)))
            #     preds = preds.summary_frame()
            #     mz_pred[:, i] = preds['mean'].values
            #     pred_err_ppm = \
            #         (preds['mean_ci_upper'] - preds['mean_ci_lower']) / \
            #         preds['mean'] * 1e6
            #     if np.any(np.abs(pred_err_ppm) > 1):
            #         warnings.warn(
            #             'Ref. mass {}. Some predictions have error > '
            #             '1 ppm.'.format(np.round(m, 4)))
            #     preds['err_ppm'] = pred_err_ppm
            #     preds['greater_than_1ppm'] = np.abs(pred_err_ppm) > 1
            #     # Save predictions
            #     preds.to_csv(os.path.join(outdir, str(np.round(m, 4)) + '.csv'))
            #     del preds
            # del outdir
            # 
            # mass_theor = np.asarray(sel_refs)
            # arg_sort = np.argsort(mass_theor)
            # mass_theor = mass_theor[arg_sort]
            # mz_pred = mz_pred[:, arg_sort]
            # 
            # plot_px = np.random.choice(len(msi.pixels_indices), size=50,
            #                            replace=False)
            # plot_px_dir = os.path.join(
            #     results_dir, 'plot_recal_px_rep_' + str(rep))
            # if not os.path.isdir(plot_px_dir):
            #     os.makedirs(plot_px_dir)
            # else:
            #     del_all_files_dir(plot_px_dir)
            # 
            # model_params = np.full(
            #     (len(msi.pixels_indices), MAX_POLY_DEGREE + 1),
            #     np.nan, dtype=float)
            # 
            # recal_masses = np.zeros((len(sel_refs), 2))
            # recal_masses[:, 0] = sel_refs
            # for i in tqdm(range(len(msi.pixels_indices))):
            #     x_fit = mz_pred[i, :].copy()
            #     y_fit = mass_theor.copy()
            #     x_pred = msi.msdata[i][:, 0].copy()
            #     mz_corrected, mdl, in_mask = recal_pixel(
            #         x_fit=x_fit, y_fit=y_fit, x_pred=x_pred,
            #         transform=TRANSFORM, max_degree=MAX_POLY_DEGREE)
            #     msi.msdata[i][:, 0] = mz_corrected
            # 
            #     # Add +1 to used masses for fit
            #     recal_masses[in_mask, 1] += 1
            # 
            #     # Save model coefficients
            #     model_params[i, :len(mdl[1].results_.params)] = mdl[
            #         1].results_.params.ravel()
            # 
            #     # Plot delta m/z vs m/z
            #     if i in plot_px:
            #         if TRANSFORM == 'sqrt':
            #             x_pred_ = np.sqrt(x_pred.copy())
            #         else:
            #             x_pred_ = x_pred.copy()
            # 
            #         pred = mdl[1].results_.get_prediction(
            #             mdl[0].transform(x_pred_.reshape(-1, 1)))
            #         pred_df = pred.summary_frame()
            # 
            #         fig = plt.figure(figsize=(4, 3), dpi=300)
            #         ax = fig.add_subplot(111)
            #         if TRANSFORM == 'sqrt':
            #             ax.plot(x_pred, pred_df['mean'] ** 2 - x_pred,
            #                     c='black', label='fit')
            #             ax.fill_between(
            #                 x=x_pred,
            #                 y1=pred_df['mean_ci_upper'] ** 2 - x_pred,
            #                 y2=pred_df['mean_ci_lower'] ** 2 - x_pred,
            #                 color='lightgrey')
            #         else:
            #             ax.plot(x_pred, pred_df['mean'] - x_pred, c='black',
            #                     label='fit')
            #             ax.fill_between(
            #                 x=x_pred, y1=pred_df['mean_ci_upper'] - x_pred,
            #                 y2=pred_df['mean_ci_lower'] - x_pred,
            #                 color='lightgrey')
            #         ax.scatter(x_fit[~in_mask],
            #                    y_fit[~in_mask] - x_fit[~in_mask],
            #                    facecolors='none', edgecolors='red',
            #                    marker='s', label='outlier', zorder=np.inf,
            #                    s=3, linewidths=0.5)
            #         ax.scatter(x_fit[in_mask],
            #                    y_fit[in_mask] - x_fit[in_mask],
            #                    facecolors='none', edgecolors='royalblue',
            #                    marker='o', label='inlier', zorder=np.inf,
            #                    s=3, linewidths=0.5)
            #         ax.set_xlabel(r'$m^{(obs)}$' + ' (m/z)',
            #                       fontdict={'weight': 'bold'})
            #         ax.set_ylabel(r'$\Delta$' + ' (m/z)',
            #                       fontdict={'weight': 'bold'})
            #         ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
            #                   fancybox=True, shadow=True, ncol=5)
            #         plt.tight_layout()
            #         plt.savefig(os.path.join(plot_px_dir, 'px_' + str(i + 1) +
            #                                  '.pdf'), format='pdf')
            #         plt.close()
            # del plot_px_dir
            # 
            # print('Saving models details ...')
            # df_coefs = pd.DataFrame(data=model_params,
            #                         columns=['beta_' + str(i) for i in
            #                                  range(model_params.shape[1])])
            # df_coefs.to_csv(os.path.join(
            #     results_dir, 'recal_coefs_' + str(rep) + '.csv'))
            # recal_masses[:, 1] /= len(msi.pixels_indices)
            # df_recal_masses = pd.DataFrame(data=recal_masses,
            #                                columns=['mass', 'pct'])
            # df_recal_masses.to_csv(
            #     os.path.join(results_dir,
            #                  'inlier_masses_rep_' + str(rep) + '.csv'))
            # 
            # print('Saving recalibrated ROI imzML ...')
            # msi.to_imzml(output_path=os.path.join(
            #     results_dir, 'recal_peaks_rep_' + str(rep) + '.imzML'))
