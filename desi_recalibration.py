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

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from tools.functions import gen_ref_list, search_ref_masses, fit_spline_kde, \
    fit_shift_model, recal_pixel, del_all_files_dir
from tools.msi import MSI


def __parse_arg():
    parser_ = argparse.ArgumentParser(description='DESI-MSI recalibration tool')
    parser_.add_argument('input', type=str, help='Input imzML file.')
    parser_.add_argument('output', type=str, help='Output imzML file.')
    parser_.add_argument('roi', type=str,
                         help='Sample ROI mask CSV file. If set equal to '
                              '\'full\', the entire image is analyzed.')
    parser_.add_argument('--analyzer', choices=['tof', 'orbitrap'],
                         help='MS analyzer.')
    parser_.add_argument('--ion-mode', choices=['pos', 'neg'],
                         help='ES Polarization mode.')
    parser_.add_argument('--search-tol', default='auto',
                         help='Search tolerance expressed in ppm. If \'auto\', '
                              'default value for MS analyzer is used.')
    parser_.add_argument('--min-coverage', default=75.0, type=float,
                         help='Min. coverage percentage for hits filtering '
                              '(default=75.0).')
    return parser_


def set_params_dict(args_) -> Dict:
    default_max_tol = {'orbitrap': 20.0, 'tof': 100.0}
    params_ = {
        'input': args_.input,
        'output': args_.output,
        'roi': args_.roi,
        'analyzer': args_.analyzer,
        'ion_mode': 'ES-' if args_.ion_mode == 'neg' else 'ES+',
        'max_tol': args_.search_tol if args_.search_tol != 'auto' else
        default_max_tol[args_.analyzer],
        'min_cov': args_.min_coverage,
        'max_disp': 5.0 if args_.analyzer == 'orbitrap' else 10.0,
        'max_degree': 1 if args_.analyzer == 'orbitrap' else 5,
        'transform': 'none' if args_.analyzer == 'orbitrap' else 'sqrt'
    }
    return params_


def main():
    parser = __parse_arg()
    args = parser.parse_args()

    params = set_params_dict(args)
    # Load MSI
    msi = MSI(imzml=params['input'], meta=params)
    # Load ROI
    if params['roi'] != 'full':
        roi = np.loadtxt(params['roi'], delimiter=',')
        if not np.all(roi.shape == msi.dim_xy[::-1]):
            raise ValueError('ROI has incompatible dimensions.')
        print('Num. ROI pixels = {}'.format(int(np.sum(roi))))
        # Remove non-ROI pixels
        outpx = np.where(roi.ravel() == 0)[0]
        delpx = np.where(np.isin(msi.pixels_indices, outpx))[0]
        delpx = np.sort(delpx)
        msi.del_pixel(list(delpx))

    # Creating match images dir
    plots_dir = os.path.join(
        os.path.dirname(params['output']), msi.ID + '_recal_imgs')
    if not os.path.isdir(plots_dir):
        print('Creating plots dir {} ...'.format(plots_dir))
        os.makedirs(plots_dir)
    else:
        del_all_files_dir(plots_dir)

    # RECALIBRATION ---------------------------

    ref_masses = gen_ref_list(ion_mode=params['ion_mode'], verbose=True)

    print('Searching lock masses within {} ppm ...'.format(
        np.round(params['max_tol'], 2)))
    matches = search_ref_masses(msiobj=msi, ref_masses=ref_masses,
                                max_tolerance=params['max_tol'], top_n=-1)

    print('Removing hits found in less than {} % of ROI pixels ...'.format(
        params['min_cov']))
    matches = {m: matches[m] for m in matches.keys() if
               len(np.unique(matches[m]['pixel'])) / len(
                   msi.pixels_indices) * 100.0 >= params['min_cov']}

    print('Num. lock masses with coverage >= {} % = {}'.format(
        np.round(params['min_cov'], 2), len(matches)))

    inliers = {m: [] for m in matches.keys()}
    inliers_px = {m: [] for m in matches.keys()}
    inliers_pct = {m: [] for m in matches.keys()}
    disp_ppm = {m: [] for m in matches.keys()}
    residuals = {m: [] for m in matches.keys()}

    for m in tqdm(matches.keys()):
        shift_preds, use_kde = fit_spline_kde(pixels=matches[m]['pixel'],
                                              match_masses=matches[m]['mz'],
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
                           (inliers_pct[m] * 100.0 >= params['min_cov']) & (
                                   np.abs(disp_ppm[m]) <=
                                   params['max_disp'])], dtype=float)
    sel_refs = np.unique(sel_refs)

    # Plot image of selected references
    for m in sel_refs:
        fig = plt.figure(dpi=300, figsize=(4, 3))
        ax = fig.add_subplot(111)
        img = np.zeros(np.prod(msi.dim_xy))
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
                    # If more than one hit per pixel, take the one with the
                    # smallest residual
                    sel_px.append(p[np.argmin(np.abs(residuals[m][idx]))])
                    sel_peaks[matches[m]['pixel'] == p] = True
        else:
            sel_px = inliers_px[m].copy()
            sel_peaks = inliers[m].copy()

        img[np.asarray(sel_px)] = matches[m]['intensity'][sel_peaks]
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        im = ax.imshow(img.reshape(msi.dim_xy[::-1]), interpolation='none',
                       cmap='inferno')
        ax.set_title('{} m/z'.format(m), fontsize=6)
        ax.set_xlabel('X', fontdict={'size': 6})
        ax.set_ylabel('Y', fontdict={'size': 6})
        ax.tick_params(labelsize=4)
        plt.colorbar(im, cax=cax)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, '{}.png'.format(m)), format='png')
        plt.close()

    shift_models = {m: [] for m in sel_refs}
    for m in sel_refs:
        shift_models[m] = fit_shift_model(matches[m]['pixel'][inliers[m]],
                                          matches[m]['mz'][inliers[m]],
                                          max_degree=5,  # THIS IS FIXED
                                          model='ols', error='mse')

    # TODO: warning for extrapolation size

    print('Recalibrating pixels ...')

    # Predict reference shift in all pixels
    mz_pred = np.full((len(msi.pixels_indices), len(sel_refs)), np.nan)
    for i, m in enumerate(sel_refs):
        mz_pred[:, i] = shift_models[m]['best_model'].predict(
            msi.pixels_indices.reshape(-1, 1)).ravel()
    mass_theor = np.asarray(sel_refs)
    arg_sort = np.argsort(mass_theor)
    mass_theor = mass_theor[arg_sort]

    for i in tqdm(range(len(msi.pixels_indices))):
        x_fit = mz_pred[i, arg_sort]
        y_fit = mass_theor.copy()
        x_pred = msi.msdata[i][:, 0].copy()
        mz_corrected, mdl = recal_pixel(x_fit=x_fit, y_fit=y_fit, x_pred=x_pred,
                                        transform=params['transform'],
                                        max_degree=params['max_degree'])
        msi.msdata[i][:, 0] = mz_corrected

    print('Saving recalibrated ROI imzML ...')
    msi.to_imzml(output_path=params['output'])


if __name__ == '__main__':
    main()
