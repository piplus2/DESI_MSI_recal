#   Copyright 2018 by Paolo Inglese, National Phenome Centre, Imperial College
#   London
#   All rights reserved.
#   This file is part of DESI-MSI recalibration, and is released under the
#   "MIT License Agreement".
#   Please see the LICENSE file that should have been included as part of this
#   package.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tools._paper_funcs import make_run_labels
from tools.plot_style import set_mpl_params_mod
from statsmodels.stats.multitest import multipletests


# def permtest(x, y, nperm):
#     t0 = np.median(x - y)
#     t1 = np.zeros(nperm)
#     for i in range(nperm):
#         sign = np.random.binomial(1, 0.5, len(x)) * 2 - 1
#         t1[i] = np.median(sign * (x - y))
#     pval = (np.sum(np.abs(t1) >= np.abs(t0)) + 1) / (nperm + 1)
#     return pval, t1


def permtest_onesample(x, nperm):
    t0 = np.median(x)
    t1 = np.zeros(nperm)
    for i in range(nperm):
        sign = np.random.binomial(1, 0.5, len(x)) * 2 - 1
        t1[i] = np.mean(sign * x)
    pval = (np.sum(np.abs(t1) >= np.abs(t0)) + 1) / (nperm + 1)
    return pval, t1


set_mpl_params_mod()

ROOT_DIR = 'E:\\CALIB_PAPER\\DATA'
pvals = []
delta_tilde = []
lo_delta = []
hi_delta = []
mu0 = []
mu1 = []
run_labels = []

for dataset in ['ORBITRAP', 'TOF']:

    data = pd.read_csv(os.path.join(ROOT_DIR, dataset, 'meta.csv'))
    data = data[data['process'] == 'yes']
    run_labels += list(make_run_labels(data))

    for index in data.index:
        d = data.loc[index, 'dir']
        print(d)
        abs_err = pd.read_csv(
            os.path.join(d, 'test_mz', 'abs_med_errors_ppm_TEST_new.csv'),
            index_col=0)

        # W, pvals_ = wilcoxon(abs_err['Orig.'], abs_err['Recal.'])
        p_, t1_ = permtest_onesample(abs_err['MAE'].values, 9999)
        pvals.append(p_)
        # Bootstrap median difference
        med_fc = np.full(9999, np.nan, dtype=float)
        for n in range(9999):
            idx = np.random.choice(abs_err.shape[0], size=abs_err.shape[0],
                                   replace=True)
            med_fc[n] = np.median(abs_err['Orig.'].to_numpy()[idx] -
                                  abs_err['Recal.'].to_numpy()[idx])

        mu0.append(np.median(abs_err['Orig.']))
        mu1.append(np.median(abs_err['Recal.']))
        delta_tilde.append(np.mean(abs_err['MAE'].values))
        lo_delta.append(np.quantile(med_fc, q=0.025))
        hi_delta.append(np.quantile(med_fc, q=0.975))

reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05,
                                              method='fdr_bh')

df = pd.DataFrame(
    data=np.c_[delta_tilde, mu0, mu1, lo_delta, hi_delta, pvals_corrected],
    columns=['delta_tilde', 'med_orig', 'med_recal', 'lo_delta', 'hi_delta',
             'pBH'])

df['reject'] = reject
df['signif'] = -np.log10(df['pBH'])
df['analyzer'] = np.r_[np.repeat('Orbitrap', 20), np.repeat('TOF', 10)]
df['label'] = run_labels

df.to_csv(os.path.join(ROOT_DIR, 'perm_test_masses.csv'))
