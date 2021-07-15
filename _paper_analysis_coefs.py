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
import matplotlib
from tools._paper_funcs import make_run_labels

set_mpl_params_mod()
params = {
        'image.origin': 'lower',
        'image.interpolation': 'none',
        'image.cmap': 'viridis',
        'axes.linewidth': 0.1,
        'axes.grid': False,
        'savefig.dpi': 300,  # to adjust notebook inline plot size
        'axes.labelsize': 10,  # fontsize for x and y labels (was 10)
        'axes.titlesize': 10,
        'font.size': 10,  # was 10
        'legend.fontsize': 10,  # was 10
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.usetex': False,
        'figure.figsize': [5, 4],
        'font.family': 'sans-serif',
        'font.sans-serif': 'Arial',
        'font.weight': 'bold',
        'xtick.major.width': 0.1,
        'xtick.minor.width': 0.1,
        'ytick.major.width': 0.1,
        'ytick.minor.width': 0.1
    }
matplotlib.rcParams.update(params)


ROOT_DIR = 'E:\\CALIB_PAPER\\DATA'
DATASET = 'TOF'

meta = pd.read_csv(os.path.join(ROOT_DIR, DATASET, 'meta.csv'), index_col=0)
meta = meta[meta['process'] == 'yes']
run_labels = make_run_labels(meta)

all_coefs = pd.DataFrame()
for i, index in enumerate(meta.index):
    run = meta.loc[index, :]

    coefs = pd.read_csv(
        os.path.join(run['dir'], '_RESULTS', 'poly_obs_mz', 'recal_coefs.csv'),
        index_col=0)

    coefs['dataset'] = run_labels[i]
    all_coefs = all_coefs.append(coefs)
    del coefs

if DATASET == 'TOF':
    for k in range(5):
        print(np.sum(np.isnan(all_coefs['beta_' + str(k)].values)) /
              all_coefs.shape[0])

fig = plt.figure(figsize=(8, 6), dpi=300)
ax = fig.add_subplot(111)
sns.boxplot(x='dataset', y='beta_0', data=all_coefs, ax=ax)
ax.set_ylabel(r'$\beta_{0}$' + ' (m/z)')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(ROOT_DIR, DATASET, 'beta0.pdf'), dpi=300, format='pdf')
plt.close()

fig = plt.figure(figsize=(8, 6), dpi=300)
ax = fig.add_subplot(111)
sns.boxplot(x='dataset', y='beta_1', data=all_coefs, ax=ax)
ax.set_ylabel(r'$\beta_{1}$')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(ROOT_DIR, DATASET, 'beta1.pdf'), dpi=300, format='pdf')
plt.close()
