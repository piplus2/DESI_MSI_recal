#   Copyright 2018 by Paolo Inglese, National Phenome Centre, Imperial College
#   London
#   All rights reserved.
#   This file is part of DESI-MSI recalibration, and is released under the
#   "MIT License Agreement".
#   Please see the LICENSE file that should have been included as part of this
#   package.


import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tools.plot_style import set_mpl_params_mod
from tools.msi import MSI
from tools._paper_funcs import make_run_labels

set_mpl_params_mod()

ROOT_DIR = 'E:\\CALIB_PAPER\\DATA'
DATASET = 'TOF'

meta = pd.read_csv(os.path.join(ROOT_DIR, DATASET, 'meta.csv'), index_col=0)
meta = meta[meta['process'] == 'yes']
run_labels = make_run_labels(meta)

inlier_masses_all = {x: [] for x in run_labels}
test_masses_all = {x: [] for x in run_labels}

mzmin = np.zeros(meta.shape[0])
mzmax = np.zeros(meta.shape[0])
for i, index in enumerate(meta.index):
    run = meta.loc[index, :]
    print(run['dir'])

    meta_ = {'ion_mode': run['ion_mode']}
    msi = MSI(imzml=os.path.join(run['dir'], run['fname']), meta=None)
    msi._MSI__meta = meta_
    for j in range(len(msi.msdata)):
        mzvec = msi.msdata[j][:, 0]
        if j == 0:
            mzmin[i] = np.min(mzvec)
            mzmax[i] = np.max(mzvec)
        else:
            mzmin[i] = np.min(np.r_[mzmin[i], mzvec])
            mzmax[i] = np.max(np.r_[mzmax[i], mzvec])

    del msi

    inlier_masses = pd.read_csv(
        os.path.join(run['dir'], '_RESULTS', 'poly_obs_mz',
                     'inlier_masses.csv'), index_col=0)

    inlier_masses_all[run_labels[i]] = \
        inlier_masses[inlier_masses['pct'] >= 0.95]['mass'].values

    test_masses = np.loadtxt(
        os.path.join(run['dir'], 'test_mz', 'test_masses.txt'))
    test_masses_all[run_labels[i]] = test_masses

for p in ['ES-', 'ES+']:
    ncols = 4
    nrows = int(np.ceil(np.sum(meta['ion_mode'] == p) / ncols))

    df = pd.DataFrame()
    for j, k in enumerate(run_labels[meta['ion_mode'] == p]):
        jj = np.where(run_labels == k)[0]

        df = df.append(
            pd.DataFrame.from_dict({
                'dset': np.repeat(k, len(inlier_masses_all[k]) + len(
                    test_masses_all[k])),
                'mz': np.r_[inlier_masses_all[k], test_masses_all[k]],
                'set': np.r_[
                    np.repeat('fit', len(inlier_masses_all[k])),
                    np.repeat('test', len(test_masses_all[k]))]
            }))

    g = sns.FacetGrid(df, col='dset', hue='set', palette='Set1',
                      col_wrap=4, sharex=False)
    g = g.map(sns.distplot, 'mz', hist=False, rug=True).add_legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(ROOT_DIR), 'New folder',
                             'ref_masses_' + DATASET + '_' + p + '.pdf'),
                format='pdf', dpi=300)
    plt.close()
