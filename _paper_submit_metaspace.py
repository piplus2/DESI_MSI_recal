#   Copyright 2018 by Paolo Inglese, National Phenome Centre, Imperial College
#   London
#   All rights reserved.
#   This file is part of DESI-MSI recalibration, and is released under the
#   "MIT License Agreement".
#   Please see the LICENSE file that should have been included as part of this
#   package.

import os
import pandas as pd
from metaspace import SMInstance


def get_sminstance(api_key):
    sm_ = SMInstance()
    if not sm_.logged_in():
        sm_.login(api_key=api_key)
    return sm_


def fill_metadata(organism, part, ion_mode, analyzer, res_power):
    meta = {
        'Data_Type': 'Imaging MS',  # shouldn't be changed
        'Sample_Information': {
            'Organism': organism,
            'Organism_Part': part,
            'Condition': 'N/A',
            'Sample_Growth_Conditions': 'N/A'
            # this is an extra field
        },
        'Sample_Preparation': {
            'Sample_Stabilisation': 'Fresh frozen',
            'Tissue_Modification': 'none',
            'MALDI_Matrix': 'none',
            'MALDI_Matrix_Application': 'none',
            'Solvent': 'none'  # this is an extra field
        },
        'MS_Analysis': {
            'Polarity': ion_mode,
            'Ionisation_Source': 'DESI',
            'Analyzer': analyzer,
            'Detector_Resolving_Power': {
                'mz': float(res_power[1]),
                'Resolving_Power': float(res_power[0])
            },
            'Pixel_Size': {
                'Xaxis': 100,
                'Yaxis': 100
            }
        }
    }
    return meta


sm = get_sminstance(api_key='4d3pw03ewJF4')

ROOT_DIR = os.path.join('E:', 'CALIB_PAPER', 'DATA')
DATASET = 'TOF'
msi_datasets = pd.read_csv(os.path.join(ROOT_DIR, DATASET, 'meta.csv'),
                           index_col=0)
msi_datasets = msi_datasets[msi_datasets['process'] == 'yes']

for index in [9, 10, 11]:  # msi_datasets.index:
    run = msi_datasets.loc[index, :]

    print(run['dir'])

    dset_name = run['organism'] + ' ' + run['tissue'] + ' ' + run[
        'ion_mode'] + ' ' + DATASET + ' orig'

    fname = run['tissue'] + '_' + run['ion_mode'] + '_0step'
    # fname = 'larocca'
    # fname = 'recal_peaks'
    imzml_fn = os.path.join(run['dir'], fname + '.imzML')
    ibd_fn = os.path.join(run['dir'], fname + '.ibd')
    metadata = fill_metadata(
        organism=run['organism'], part=run['tissue'],
        ion_mode='Negative' if run['ion_mode'] == 'ES-' else 'Positive',
        analyzer=DATASET, res_power=eval(run['res_power']))

    sm.submit_dataset(imzml_fn=imzml_fn, ibd_fn=ibd_fn, name=dset_name,
                      metadata=metadata, is_public=False,
                      databases=[19, 22, 24, 38])
