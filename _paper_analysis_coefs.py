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
        os.path.join(run['dir'], '_RESULTS', 'new_inmask', 'recal_coefs.csv'),
        index_col=0)

    coefs['dataset'] = run_labels[i]
    all_coefs = all_coefs.append(coefs)
    del coefs

for i in range(6):
    all_coefs[['beta_' + str(i), 'dataset']].to_csv(os.path.join(ROOT_DIR, DATASET, 'all_coefs_beta_' + str(i) + '.csv'))

if DATASET == 'TOF':
    for k in range(6):
        print(np.sum(np.isnan(all_coefs['beta_' + str(k)].values)) /
              all_coefs.shape[0])

fig = plt.figure(figsize=(8, 6), dpi=300)
ax = fig.add_subplot(111)
sns.boxplot(x='dataset', y='beta_0', data=all_coefs, ax=ax)
if DATASET == 'TOF':
    ax.set_ylabel(r'$\beta_{0} ((m/z)^{1/2})$')
else:
    ax.set_ylabel(r'$\beta_{0} (m/z)$')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
plt.yscale('symlog', linthresh=1e-5)
plt.tight_layout()
plt.savefig(os.path.join(ROOT_DIR, DATASET, 'beta0.pdf'), dpi=300, format='pdf')
plt.close()

fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all')
# plot the same data on both axes
sns.boxplot(x='dataset', y='beta_1', data=all_coefs, ax=ax1)
sns.boxplot(x='dataset', y='beta_1', data=all_coefs, ax=ax2)

# zoom-in / limit the view to different portions of the data
ax1.set_yscale('symlog')
ax1.set_ylim(1-2e-4, 1+3e-4)  # outliers only
ax2.set_ylim(0.9975, 0.9977)  # most of the data

# hide the spines between ax and ax2
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

ax1.set_xlabel('')
ax1.set_ylabel('')
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.set_ylabel(r'$\beta_{1}$')
plt.tight_layout()
plt.savefig(os.path.join(ROOT_DIR, DATASET, 'beta1.pdf'), dpi=300, format='pdf')
plt.close()

fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all')
# plot the same data on both axes
sns.boxplot(x='dataset', y='beta_2', data=all_coefs, ax=ax1)
sns.boxplot(x='dataset', y='beta_2', data=all_coefs, ax=ax2)

# zoom-in / limit the view to different portions of the data
ax2.set_yscale('symlog', linthresh=1e-5)
ax1.set_ylim(0.000265, 0.000275)  # outliers only
ax2.set_ylim(-0.00017, 1e-5)  # most of the data

# hide the spines between ax and ax2
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

ax1.set_xlabel('')
ax1.set_ylabel('')
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.set_ylabel(r'$\beta_{2}$')
ax1.set_title(r'$\beta_{2}$')
plt.tight_layout()
plt.savefig(os.path.join(ROOT_DIR, DATASET, 'beta2.pdf'), dpi=300, format='pdf')
plt.close()

fig = plt.figure(figsize=(8, 6), dpi=300)
ax = fig.add_subplot(111)
sns.boxplot(x='dataset', y='beta_3', data=all_coefs, ax=ax)
ax.set_ylabel(r'$\beta_{3}$')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
plt.yscale('symlog', linthresh=1e-7)
plt.tight_layout()
plt.savefig(os.path.join(ROOT_DIR, DATASET, 'beta3.pdf'), dpi=300, format='pdf')
plt.close()

fig = plt.figure(figsize=(8, 6), dpi=300)
ax = fig.add_subplot(111)
sns.boxplot(x='dataset', y='beta_4', data=all_coefs, ax=ax)
ax.set_ylabel(r'$\beta_{4}$')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(ROOT_DIR, DATASET, 'beta4.pdf'), dpi=300, format='pdf')
plt.close()

fig = plt.figure(figsize=(8, 6), dpi=300)
ax = fig.add_subplot(111)
sns.boxplot(x='dataset', y='beta_5', data=all_coefs, ax=ax)
ax.set_ylabel(r'$\beta_{5}$')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(ROOT_DIR, DATASET, 'beta5.pdf'), dpi=300, format='pdf')
plt.close()