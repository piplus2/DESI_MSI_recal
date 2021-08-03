#   Copyright 2018 by Paolo Inglese, National Phenome Centre, Imperial College
#   London
#   All rights reserved.
#   This file is part of DESI-MSI recalibration, and is released under the
#   "MIT License Agreement".
#   Please see the LICENSE file that should have been included as part of this
#   package.


import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from metaspace import SMInstance
from tools.plot_style import set_mpl_params_mod
from tools._paper_funcs import make_run_labels


def get_sminstance(api_key):
    sm_ = SMInstance()
    if not sm_.logged_in():
        sm_.login(api_key=api_key)
    return sm_


def permtest(x, y, nperm):
    t0 = np.median(x - y)
    t1 = np.zeros(nperm)
    for i in range(nperm):
        sign = np.random.binomial(1, 0.5, len(x)) * 2 - 1
        t1[i] = np.median(sign * (x - y))
    pval = (np.sum(np.abs(t1) >= np.abs(t0)) + 1) / (nperm + 1)
    return pval, t1


set_mpl_params_mod()
sm = get_sminstance(api_key='4d3pw03ewJF4')

dbs = [['ChEBI', '2018-01'], ['CoreMetabolome', 'v3'], ['HMDB', 'v4'],
       ['LipidMaps', '2017-12-12']]

ROOT_DIR = 'E:\\CALIB_PAPER\\DATA'

anno_formulas = {'ORBITRAP': [], 'TOF': []}
run_labels = []
anno_db = np.empty((0, 2))
set_size = {'ORBITRAP': 0, 'TOF': 0}

Moran = {'ORBITRAP': [], 'TOF': []}

for dataset in ['ORBITRAP', 'TOF']:
    meta = pd.read_csv(os.path.join(ROOT_DIR, dataset, 'meta.csv'), index_col=0)
    meta = meta[meta['process'] == 'yes']
    run_labels += list(make_run_labels(meta))
    set_size[dataset] = meta.shape[0]

    for index in meta.index:
        run = meta.loc[index, :]

        print(run['dir'])
        roi = np.loadtxt(os.path.join(run['dir'], 'roi.csv'), delimiter=',')
        roi = roi[~np.all(roi == 0, axis=1), :]
        roi = roi[:, ~np.all(roi == 0, axis=0)]

        out_dir = run['dir']
        orig_id = run['orig_id']
        recal_id = run['recal_id']

        ds_orig = sm.dataset(id=orig_id)
        ds_recal = sm.dataset(id=recal_id)

        print('Reading annotated molecules ...')
        anno_orig = []
        anno_recal = []
        for i, db in enumerate(dbs):
            results_orig = ds_orig.results(database=(db[0], db[1]))
            results_recal = ds_recal.results(database=(db[0], db[1]))

            results_orig = results_orig[results_orig['fdr'] == 0.05]
            results_recal = results_recal[results_recal['fdr'] == 0.05]

            formula_orig = [x[0] for x in results_orig.index]
            adduct_orig = [x[1] for x in results_orig.index]

            formula_recal = [x[0] for x in results_recal.index]
            adduct_recal = [x[1] for x in results_recal.index]

            anno_orig += \
                [x + '' + y for x, y in zip(formula_orig, adduct_orig)]
            anno_recal += \
                [x + '' + y for x, y in zip(formula_recal, adduct_recal)]

        anno_orig = np.unique(anno_orig)
        anno_recal = np.unique(anno_recal)

        anno_db = np.vstack((anno_db, [len(anno_orig), len(anno_recal)]))
        anno_formulas[dataset].append({'orig': anno_orig, 'recal': anno_recal})

df_ = pd.DataFrame.from_dict(
    {
        'run_label': run_labels,
        'diff': (anno_db[:, 1] - anno_db[:, 0]),
        'rel_diff': (anno_db[:, 1] - anno_db[:, 0]) / anno_db[:, 0],
        'analyzer': np.r_[
            np.repeat('Orbitrap', 20), np.repeat('TOF', 10)]
    }
)
df_ = df_.sort_values(by='rel_diff', ascending=False)
df_.to_csv(
    os.path.join(os.path.dirname(ROOT_DIR), 'New folder', 'annotations.csv'))


def align_yaxis_np(ax1, ax2):
    """Align zeros of the two axes, zooming them out by same ratio"""
    axes = np.array([ax1, ax2])
    extrema = np.array([ax.get_ylim() for ax in axes])
    tops = extrema[:,1] / (extrema[:,1] - extrema[:,0])
    # Ensure that plots (intervals) are ordered bottom to top:
    if tops[0] > tops[1]:
        axes, extrema, tops = [a[::-1] for a in (axes, extrema, tops)]

    # How much would the plot overflow if we kept current zoom levels?
    tot_span = tops[1] + 1 - tops[0]

    extrema[0,1] = extrema[0,0] + tot_span * (extrema[0,1] - extrema[0,0])
    extrema[1,0] = extrema[1,1] + tot_span * (extrema[1,0] - extrema[1,1])
    [axes[i].set_ylim(*extrema[i]) for i in range(2)]


fig = plt.figure(figsize=(4, 3), dpi=300)
ax = fig.add_subplot(111)
sns.barplot(x='run_label', y='rel_diff', data=df_, fill='analyzer',
            hue='analyzer', dodge=False,
            palette={'Orbitrap': 'cornflowerblue', 'TOF': 'salmon'},
            ax=ax)
ax.set_ylabel(r'$d_{r}$')
ax.set_yscale('symlog', linthresh=0.1)
ax.set_ylim((-50, 50))

ax.yaxis.set_minor_locator(MultipleLocator(10))

ax.set_xlabel('')

ax2 = ax.twinx()
ax2.set_ylabel('Delta num. annot.')
ax2.plot(df_['run_label'], df_['diff'], c='black', lw=0.5, ls='dashed')
ax2.scatter(df_[df_['diff'] > 0]['run_label'], df_[df_['diff'] > 0]['diff'],
            s=7, c='#1B9E77', marker='^', edgecolors='black', linewidth=0.1)
ax2.scatter(df_[df_['diff'] <= 0]['run_label'], df_[df_['diff'] <= 0]['diff'],
            s=7, c='#D95F02', marker='v', edgecolors='black', linewidth=0.1)
ax2.set_yscale('symlog')
ax2.yaxis.set_minor_locator(MultipleLocator(10))

plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()

ax2.get_ylim()
ax2.set_ylim((-300, 300))

plt.savefig(os.path.join(os.path.dirname(ROOT_DIR), 'New folder',
                         'num_annotations.tif'), format='tif')


# Venn diagrams

