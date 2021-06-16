#   Copyright 2018 by Paolo Inglese, National Phenome Centre, Imperial College
#   London
#   All rights reserved.
#   This file is part of DESI-MSI recalibration, and is released under the
#   "MIT License Agreement".
#   Please see the LICENSE file that should have been included as part of this
#   package.


import numpy as np
from .msi import MSI
from joblib import Parallel, delayed


class MSBinner:

    def __init__(self, decimals: int):
        self.__decimals = decimals
        self.__bin_cmz = None

    def bin(self, msobj: MSI) -> np.ndarray:
        list_msx = msobj.msdata
        # Extract the full m/z vector and bin it at the digit level
        print("Binning M/Z values with bin size = {} M/Z ...".format(
            10 ** self.__decimals))

        _idx = msobj.pixels_indices

        binned_mz = np.empty(0)
        for msp in list_msx:
            mz_ = np.round(msp[:, 0], decimals=self.__decimals)
            binned_mz = np.append(binned_mz, mz_)
            binned_mz = np.unique(binned_mz)
            del mz_

        # binned_mz = np.round(all_mz, decimals=self.__decimals)
        self.__bin_cmz = np.sort(np.unique(binned_mz))
        print("Num. M/Z bins: {}".format(len(self.__bin_cmz)))

        # Bin the spectra intensities: skip empty objects
        print("Binning intensities ...")
        bin_yi_list = Parallel(n_jobs=-1, require='sharedmem')(
            delayed(self.thread)(msp)
            for msp in list_msx)

        binned_intensities = np.zeros(
            (np.prod(msobj.dim_xy), len(self.__bin_cmz)))
        for i, idx in enumerate(_idx):
            binned_intensities[idx, :] = bin_yi_list[i]

        del bin_yi_list

        return binned_intensities

    def thread(self, msp: np.ndarray) -> np.ndarray:
        bin_yi = np.zeros(len(self.__bin_cmz))

        mz_ = np.round(msp[:, 0], decimals=self.__decimals)
        umz, s = np.unique(mz_, return_index=True)
        yi_ = np.split(msp[:, 1], s[1:])
        # Sum intensities same M/Z
        yi_ = [np.sum(x) for x in yi_]
        bin_yi[np.isin(self.__bin_cmz, umz)] = np.asarray(yi_)

        return bin_yi
