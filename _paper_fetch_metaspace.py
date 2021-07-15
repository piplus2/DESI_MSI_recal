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
import libpysal
import esda
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

anno_mz = {'ORBITRAP': [], 'TOF': []}
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
        anno_mz_orig = []
        anno_mz_recal = []
        for i, db in enumerate(dbs):
            results_orig = ds_orig.results(database=(db[0], db[1]))
            results_recal = ds_recal.results(database=(db[0], db[1]))
            anno_mz_orig += list(
                results_orig.ionFormula[results_orig.fdr == 0.05].values)
            anno_mz_recal += list(
                results_recal.ionFormula[results_recal.fdr == 0.05].values)

        anno_mz_orig = np.unique(anno_mz_orig)
        anno_mz_recal = np.unique(anno_mz_recal)

        anno_db = np.vstack((anno_db, [len(anno_mz_orig), len(anno_mz_recal)]))
        anno_mz[dataset].append({'orig': anno_mz_orig, 'recal': anno_mz_recal})

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
ax.set_yscale('symlog')
ax.yaxis.set_minor_locator(MultipleLocator(10))

ax.set_xlabel('')
ax2 = ax.twinx()
ax2.set_ylabel('Delta num. annot.')
ax2.plot(df_['run_label'], df_['diff'], c='black', lw=0.5, ls='dashed')
ax2.scatter(df_[df_['diff'] > 0]['run_label'], df_[df_['diff'] > 0]['diff'],
            s=7, c='#1B9E77', marker='^', edgecolors='black', linewidth=0.1)
ax2.scatter(df_[df_['diff'] <= 0]['run_label'], df_[df_['diff'] <= 0]['diff'],
            s=7, c='#D95F02', marker='v', edgecolors='black', linewidth=0.1)
ax2.yaxis.set_minor_locator(MultipleLocator(10))

plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()

ax2.get_ylim()
ax2.set_ylim((-148, 309))

plt.savefig(os.path.join(os.path.dirname(ROOT_DIR), 'New folder',
                         'num_annotations.tif'), format='tif')

# fig, ax = plt.subplots(1, 3, dpi=300, figsize=(9, 3))
# ax = ax.flatten()

# anno_db = np.zeros((3, 2))
# for i, db in enumerate(dbs):
#     results_orig = ds_orig.results(database=(db[0], db[1]))
#     results_recal = ds_recal.results(database=(db[0], db[1]))
#     anno_db[i, 0] = len(
#         np.unique(results_orig.ionFormula[results_orig.fdr <= 0.05]))
#     anno_db[i, 1] = len(
#         np.unique(results_recal.ionFormula[results_recal.fdr <= 0.05]))

# anno_mz_orig = {x: [] for x in fdr}
# anno_mz_recal = {x: [] for x in fdr}
# for i, db in enumerate(dbs):
#     results_orig = ds_orig.results(database=(db[0], db[1]))
#     results_recal = ds_recal.results(database=(db[0], db[1]))
#
#     for f in fdr:
#         # orig_formulas = np.asarray([x[0] for x in results_orig.index])
#         # recal_formulas = np.asarray([x[0] for x in results_recal.index])
#         anno_mz_orig[f] += list(
#             results_orig.ionFormula[results_orig.fdr <= f])
#         anno_mz_recal[f] += list(
#             results_recal.ionFormula[results_recal.fdr <= f])
#
#     counts_orig = results_orig.fdr.value_counts().sort_index()
#     counts_recal = results_recal.fdr.value_counts().sort_index()
#
#     cumsum_orig = counts_orig.cumsum()
#     cumsum_recal = counts_recal.cumsum()
#
#     df = cumsum_orig.to_frame()
#     df.columns = ['Orig.']
#     df['FDR'] = df.index
#     df['Recal.'] = cumsum_recal.to_frame()['fdr']
#     df = df[['FDR', 'Orig.', 'Recal.']]
#     df.index = list(range(df.shape[0]))
#
#     df.to_csv(
#         os.path.join(out_dir, 'test_mz', 'annotations_' + db[0] + '.csv'),
#         index=False)

# # Plot annotations cumsum
# ax[i].plot(cumsum_orig.index, cumsum_orig.values, marker='o',
#            color='red', markeredgecolor='red', markerfacecolor='none',
#            label='Orig.')
# ax[i].plot(cumsum_recal.index, cumsum_recal.values, marker='o',
#            color='#00C000', markeredgecolor='#00C000',
#            markerfacecolor='none', label='Recal.')
# ax[i].legend(loc='lower right', fontsize=6)
# # ax[i].legend(loc='upper center', bbox_to_anchor=(0.5, 0.1),
# #              fancybox=True, shadow=True, ncol=2)
# ax[i].set_xlabel('FDR')
# ax[i].set_ylabel(r'$\Sigma$')
# ax[i].set_xticks([0.05, 0.1, 0.2, 0.5])
# ax[i].set_xticklabels(['0.05', '0.10', '0.20', '0.50'], rotation=45,
#                       ha='right')
# ax[i].set_title(db[0])

# plt.suptitle(run['tissue'] + ' ' + run['ion_mode'])
# plt.tight_layout()
# plt.savefig(os.path.join(out_dir, 'test_mz', 'annotations.pdf'),
#             format='pdf')
# plt.close()

# Num unique ions
# tot_anno_orig = [len(np.unique(anno_mz_orig[x])) for x in fdr]
# tot_anno_recal = [len(np.unique(anno_mz_recal[x])) for x in fdr]
# tot_anno = pd.DataFrame(np.c_[tot_anno_orig, tot_anno_recal],
#                         index=fdr,
#                         columns=['Orig.', 'Recal.'])
# tot_anno = tot_anno.cumsum()

anno_db = pd.DataFrame.from_dict({
    'db': ['CM', 'HMDB', 'LM'] * 2,
    'n': np.r_[anno_db[:, 0], anno_db[:, 1]].astype(int),
    'set': (['Orig.'] * 3) + (['Recal.'] * 3)
})
anno_db['n'].astype(int)
g = sns.barplot(x='db', y='n', hue='set', data=anno_db,
                ax=ax[index],
                palette={'Orig.': '#FF0000', 'Recal.': '#00C000'})
ax[index].legend([], [], frameon=False)

# ax[index].plot(fdr, tot_anno['Orig.'], marker='o', label='Orig.',
#                color='#FF0000',
#                markerfacecolor='none', markeredgecolor='#FF0000')
# ax[index].plot(fdr, tot_anno['Recal.'], marker='*', label='Recal.',
#                color='#00C000')
ax[index].set_xticks(fdr)
ax[index].set_xticklabels(fdr, rotation=45)
ax2 = ax[index].twinx()
ax2.set_ylabel(r'$d_{r}$')
ax2.plot(['CM', 'HMDB', 'LM'],
         (anno_db[anno_db['set'] == 'Recal.']['n'].to_numpy() -
          anno_db[anno_db['set'] == 'Orig.']['n'].to_numpy()) /
         anno_db[anno_db['set'] == 'Orig.']['n'].to_numpy(),
         ls='dashed', color='navy', lw=0.5, marker='.', markersize=1)
ax2.tick_params(axis='y', labelcolor='navy')
ax2.axhline(y=0, ls='dashed', color='gray', lw=0.5)

ax[index].set_xlabel('db', fontdict={'weight': 'bold'})
ax[index].set_ylabel(r'$\Sigma$')
ax[index].yaxis.set_minor_locator(AutoMinorLocator())
ax[index].tick_params(which='minor', length=5)
# ax[index].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax[index].set_title(run['tissue'] + ' ' + run['ion_mode'],
                    fontweight='bold')
ax[index].set_visible(True)

plt.tight_layout()
plt.savefig(os.path.join(ROOT_DIR, 'unique_anno.pdf'), format='pdf')
plt.close()

# fig = plt.figure(figsize=(4, 3), dpi=300)
# ax = fig.add_subplot(111)
# ax = sns.violinplot(x='db', y='msm', hue='set', data=msm,
#                     palette={'Orig.': '#FF0000', 'Recal.': '#00C000'},
#                     ax=ax, trim=True)
# ax.set_xticklabels([db[0] for db in dbs], fontsize=6)
# ax.set_xlabel('db')
# ax.set_ylabel('MSM [FDR = 0.05]')
# ax.set_title(run['tissue'] + ' ' + run['ion_mode'])
# plt.tight_layout()
# plt.savefig(os.path.join(out_dir, 'test_mz', 'msm_fdr_0.05.pdf'),
#             format='pdf')
# plt.close()
