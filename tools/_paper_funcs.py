#   Copyright 2018 by Paolo Inglese, National Phenome Centre, Imperial College
#   London
#   All rights reserved.
#   This file is part of DESI-MSI recalibration, and is released under the
#   "MIT License Agreement".
#   Please see the LICENSE file that should have been included as part of this
#   package.


import numpy as np


def make_run_labels(meta):
    run_labels = (meta['tissue'] + ' ' + meta['ion_mode']).values
    u, s = np.unique(run_labels, return_counts=True)
    for lbl in u[s > 1]:
        for i, j in enumerate(np.where(run_labels == lbl)[0]):
            run_labels[j] = run_labels[j] + ' ' + str(i + 1)
    return run_labels
