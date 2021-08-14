import numpy as np
import pandas as pd
import os

root_dir = 'C:/Users/pingl/Documents/GitHub/Peakolo-DESI-MSI/peakolo/' \
           'calibration'

csv_fnames = []
for f in os.listdir(os.path.join(root_dir, 'orig_lipidmaps')):
    if os.path.isfile(os.path.join(root_dir, 'orig_lipidmaps', f)) and \
            os.path.splitext(f)[1] == '.csv':
        csv_fnames.append(os.path.join(root_dir, 'orig_lipidmaps', f))

for f in csv_fnames:

    lipid_class = os.path.splitext(os.path.basename(f))[0]

    tab = pd.read_csv(f)
    tab = tab[tab['MASS'] != '-']

    tab['MASS'] = np.asarray(tab['MASS'], dtype=float)
    unique_mz = np.asarray(np.unique(tab['MASS']), dtype=float)

    merge_dict = {'MASS': [], 'COMMON_NAME': [], 'FORMULA': []}
    for m in unique_mz:
        merge_dict['MASS'].append(m)
        merge_dict['COMMON_NAME'].append(
            ';'.join(np.unique(tab[tab['MASS'] == m]['COMMON_NAME'])))
        merge_dict['FORMULA'].append(
            ';'.join(np.unique(tab[tab['MASS'] == m]['FORMULA'])))

    df = pd.DataFrame.from_dict(merge_dict)
    df.to_csv(os.path.join(root_dir, '{}_curated.csv'.format(lipid_class)))
