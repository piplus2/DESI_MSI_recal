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


def boot(x, nb, statistics=np.median):
    t0 = statistics(x)
    x1 = x - t0
    t1 = np.zeros(nb)
    for i in range(nb):
        xb = np.random.choice(x1, size=len(x1), replace=True)
        t1[i] = statistics(xb)
    pval = (np.sum(np.abs(t1) >= np.abs(t0)) + 1) / (nb + 1)
    return pval


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
        abs_err = \
            pd.read_csv(
                os.path.join(
                    d,
                    # 'test_mz', 'abs_med_errors_ppm_TEST_new.csv'),
                    '_RESULTS', 'new_inmask', 'test_masses',
                    'abs_med_errors_ppm_TEST_new.csv'), index_col=0)

        p_ = boot(abs_err['MAE recal.'].values, 9999)
        pvals.append(p_)
        # Bootstrap median difference
        med_fc = np.full(9999, np.nan, dtype=float)
        for n in range(9999):
            idx = np.random.choice(abs_err.shape[0], size=abs_err.shape[0],
                                   replace=True)
            med_fc[n] = np.median(abs_err['Recal.'].to_numpy()[idx] -
                                  abs_err['noTS'].to_numpy()[idx])

        mu0.append(np.median(abs_err['Recal.']))
        mu1.append(np.median(abs_err['noTS']))
        delta_tilde.append(np.mean(abs_err['MAE recal.'].values))
        lo_delta.append(np.quantile(med_fc, q=0.025))
        hi_delta.append(np.quantile(med_fc, q=0.975))

reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05,
                                              method='fdr_bh')

df = pd.DataFrame(
    data=np.c_[delta_tilde, mu0, mu1, lo_delta, hi_delta, pvals_corrected],
    columns=['delta_tilde', 'med_recal', 'med_nots', 'lo_delta', 'hi_delta',
             'pBH'])

df['reject'] = reject
df['signif'] = -np.log10(df['pBH'])
df['analyzer'] = np.r_[np.repeat('Orbitrap', 20), np.repeat('TOF', 10)]
# df['analyzer'] = np.repeat('TOF', 10)
df['label'] = run_labels

df.to_csv(os.path.join(ROOT_DIR, 'perm_test_masses_recal_vs_nots.csv'))
