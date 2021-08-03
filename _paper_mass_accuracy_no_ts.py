#   Copyright 2018 by Paolo Inglese, National Phenome Centre, Imperial College
#   London
#   All rights reserved.
#   This file is part of DESI-MSI recalibration, and is released under the
#   "MIT License Agreement".
#   Please see the LICENSE file that should have been included as part of this
#   package.


import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, median_abs_deviation
from scipy.signal import find_peaks

from tools.functions import search_ref_masses
from tools.msi import MSI
from tools.plot_style import set_mpl_params


def dmz_to_dppm(dmz: np.ndarray, m: float) -> float:
    return dmz / m * 1e6


def dppm_to_dmz(dppm: np.ndarray, m: float) -> float:
    return dppm * m / 1e6


def del_all_files_dir(dirname: str) -> None:
    for filename in os.listdir(dirname):
        file_path = os.path.join(dirname, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)


set_mpl_params()

REF_DIR = os.path.join('C:/Users', 'pingl', 'Desktop',
                       'MSI_recalibration-master', 'recalibration')
ROOT_DIR = os.path.join('E:', 'CALIB_PAPER', 'DATA')
MAX_TOL = {'ORBITRAP': 20, 'TOF': 100}
MIN_PCT = 75.0

for dataset in ['TOF']:  # ['TOF', 'ORBITRAP']:
    msi_datasets = pd.read_csv(os.path.join(ROOT_DIR, dataset, 'meta.csv'),
                               index_col=0)
    msi_datasets = msi_datasets[msi_datasets['process'] == 'yes']
    for index in [0]:  # msi_datasets.index:
        run = msi_datasets.loc[index, :]

        print('MSI {}/{}: {}'.format(index + 1, msi_datasets.shape[0],
                                     run['dir']))

        outdir = os.path.join(run['dir'], '_RESULTS', 'analysis_no_ts',
                              'test_mass_accuracy')
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        else:
            del_all_files_dir(outdir)

        # Find test ions -------------------------------------------------------

        # # Load reference masses used for recalibrating the data
        # refdir = os.path.join(run['dir'], '_RESULTS', 'poly_obs_mz')
        # train_masses = np.loadtxt(os.path.join(refdir, 'references.csv'))
        # del refdir

        # Load the METASPACE reference and remove the recalibration masses
        test_masses_fname = run['tissue'] + '_' + (
            'pos' if run['ion_mode'] == 'ES+' else 'neg') + '.txt'
        print('Loading METASPACE masses from {} ...'.format(test_masses_fname))
        test_masses = np.loadtxt(os.path.join(REF_DIR, test_masses_fname))
        test_masses = np.unique(np.round(test_masses, 4))
        # rm_idx = []
        # for i in range(len(train_masses)):
        #     rm_idx.append(
        #         np.where(np.abs(test_masses - train_masses[i]) <= 2e-4)[0])
        # if len(rm_idx) > 0:
        #     rm_idx = np.concatenate(rm_idx)
        #     test_masses = np.delete(test_masses, rm_idx)
        # Remove test masses that differ less than 2*1e-4
        test_masses = np.sort(test_masses)
        d_ = np.where(np.diff(test_masses) <= 2e-4)[0]
        test_masses = np.delete(test_masses, d_ + 1)
        print('Num. test lock masses = {}'.format(len(test_masses)))

        # Original data --------------------------------------------------------

        print('Searching lock masses ...')
        input_imzml_orig = '{}_{}_0step.imzML'.format(run['tissue'],
                                                      run['ion_mode'])
        print('Loading original data ...')
        meta = {'ion_mode': run['ion_mode']}
        msi = MSI(imzml=os.path.join(run['dir'], input_imzml_orig), meta=None)
        msi._MSI__meta = meta
        matches = search_ref_masses(msiobj=msi, ref_masses=test_masses,
                                    max_tolerance=MAX_TOL[dataset], top_n=-1)
        matches = {m: matches[m] for m in matches.keys() if
                   len(np.unique(matches[m]['pixel'])) / len(
                       msi.pixels_indices) * 100.0 >= MIN_PCT}

        # Take only references with similar errors in ppm

        err_tilde = np.asarray([np.median(dmz_to_dppm(matches[m]['mz'] - m, m))
                                for m in matches.keys()], dtype=float)
        if np.any(np.isnan(err_tilde)):
            raise RuntimeError(
                '{} median errors are nan.'.format(np.sum(np.isnan(err_tilde))))

        density = gaussian_kde(err_tilde)
        x_density = np.linspace(np.min(err_tilde), np.max(err_tilde),
                                int(np.ceil(np.ptp(err_tilde)) * 2))
        y_density = density(x_density)
        # Find density highest peak interval
        peaks, properties = find_peaks(
            x=y_density)  # , rel_height=0.25, width=0)
        max_peak = np.argmax(y_density[peaks])
        xmax_density = x_density[peaks[max_peak]]
        xleft = np.max([xmax_density - 1, x_density[0]])
        xright = np.min([xmax_density + 1, x_density[-1]])

        # Plot density
        fig = plt.figure(figsize=(4, 3), dpi=300)
        ax = fig.add_subplot(111)
        ax.plot(x_density, y_density, c='black')
        ax.axvline(x=xmax_density, color='red')
        idx_left = np.argmin(np.abs(x_density - xleft))
        idx_right = np.argmin(np.abs(x_density - xright))
        ax.axvline(x=xleft, c='black', ls='dashed')
        ax.axvline(x=xright, c='black', ls='dashed')
        ax.set_xlabel(r'$\tilde{\Delta}$' + ' (ppm)')
        ax.set_ylabel(r'$f(\tilde{\Delta})$')
        ax.fill_between(x_density[idx_left:idx_right + 1],
                        y_density[idx_left:idx_right + 1],
                        color='gray')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'error_density.pdf'), format='pdf')
        plt.close()

        sel_test_masses = []
        for i, m in enumerate(matches.keys()):
            if (err_tilde[i] >= xleft) & (err_tilde[i] <= xright):
                sel_test_masses.append(m)
        # Save list of test masses
        np.savetxt(fname=os.path.join(outdir, 'test_masses.txt'),
                   X=sel_test_masses, fmt='%f')

        print('Num. test masses = {}'.format(len(sel_test_masses)))
        matches_test_orig = {m: matches[m] for m in sel_test_masses}
        del msi, matches

        # Recal data -----------------------------------------------------------

        print('Loading recal. data ...')
        meta = {'ion_mode': run['ion_mode']}
        msi = MSI(imzml=os.path.join(run['dir'], '_RESULTS', 'analysis_no_ts',
                                     'recal_peaks.imzML'),
                  meta=None)
        msi._MSI__meta = meta
        matches_test_recal = {x: [] for x in matches_test_orig.keys()}
        for m in matches_test_orig.keys():
            mass_ = []
            for px, match_idx in zip(matches_test_orig[m]['pixel'],
                                     matches_test_orig[m]['peak']):
                idx = int(np.where(msi.pixels_indices == px)[0])
                mass_.append(msi.msdata[idx][match_idx, 0])
            matches_test_recal[m] = {'mz': np.asarray(mass_, dtype=float)}
            del mass_
        del msi

        # Gen tables and save --------------------------------------------------

        mae = np.zeros(len(matches_test_orig))
        med_errors = np.full((len(matches_test_orig), 2), np.nan, dtype=float)
        mad_errors = np.full((len(matches_test_orig), 2), np.nan, dtype=float)

        for i, m in enumerate(matches_test_orig.keys()):
            assert (len(matches_test_orig[m]['mz']) == len(
                matches_test_recal[m]['mz']))
            ppm_orig = dmz_to_dppm(
                np.asarray(matches_test_orig[m]['mz']) - m, m)
            ppm_recal = dmz_to_dppm(
                np.asarray(matches_test_recal[m]['mz'] - m), m)
            mae[i] = np.median(np.abs(ppm_orig) - np.abs(ppm_recal))
            med_errors[i, :] = np.asarray(
                [np.median(ppm_orig), np.median(ppm_recal)])
            mad_errors[i, :] = np.asarray(
                [median_abs_deviation(ppm_orig),
                 median_abs_deviation(ppm_recal)])

        # Save median and MAD errors

        df = pd.DataFrame(data=np.c_[np.abs(med_errors), mae],
                          columns=['Orig.', 'Recal.', 'MAE'])  # , 'LaRocca'])
        csv_fname = 'abs_med_errors_ppm_TEST_new.csv'
        df.to_csv(os.path.join(outdir, csv_fname))

        df = pd.DataFrame(data=mad_errors,
                          columns=['Orig.', 'Recal.'])  # , 'LaRocca'])
        csv_fname = 'mad_errors_ppm_TEST_new.csv'
        df.to_csv(os.path.join(outdir, csv_fname))
