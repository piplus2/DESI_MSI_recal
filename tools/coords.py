#   Copyright 2018 by Paolo Inglese, National Phenome Centre, Imperial College
#   London
#   All rights reserved.
#   This file is part of DESI-MSI recalibration, and is released under the
#   "MIT License Agreement".
#   Please see the LICENSE file that should have been included as part of this
#   package.

import numpy as np


def coord2idx(x_array, y_array, order='C') -> np.ndarray:

    unique_x = np.sort(np.unique(x_array))
    unique_y = np.sort(np.unique(y_array))

    nrows = len(unique_y)
    ncols = len(unique_x)
    mat_idx = np.arange(nrows * ncols).reshape((nrows, ncols), order=order)

    idx = np.full(len(x_array), -1, dtype=int)
    for i in range(len(x_array)):
        row = np.where(unique_y == y_array[i])[0]
        col = np.where(unique_x == x_array[i])[0]
        idx[i] = mat_idx[row, col]

    return idx
