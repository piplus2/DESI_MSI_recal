#   Copyright 2018 by Paolo Inglese, National Phenome Centre, Imperial College
#   London
#   All rights reserved.
#   This file is part of DESI-MSI recalibration, and is released under the
#   "MIT License Agreement".
#   Please see the LICENSE file that should have been included as part of this
#   package.


import os
from dataclasses import dataclass
from typing import List, Union, Dict

import numpy as np
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
from tqdm import tqdm

from .coords import coord2idx


@dataclass
class MSI:
    __px: Union[np.ndarray, None]
    __xy: Union[np.ndarray, None]
    __dim_xy: np.ndarray
    __msp: List[np.ndarray]
    __meta: Dict
    __is_binned: bool
    __cmz: np.ndarray
    __id: str

    def __init__(self, imzml, meta):
        self.__msp = []
        self.__px = None
        self.__xy = None
        self.__dim_xy = np.array([0, 0], dtype=int)
        self.__meta = meta
        self.__fname = imzml
        self.__is_binned = False
        self.__id = os.path.splitext(os.path.basename(imzml))[0]

        self.__load_msdata()

    def __load_msdata(self):
        _x = []
        _y = []
        self.__msp = []
        with ImzMLParser(self.__fname) as p:
            for i, (x, y, z) in enumerate(tqdm(p.coordinates)):
                _x.append(x)
                _y.append(y)
                mz_, yi_ = p.getspectrum(index=i)
                self.__msp.append(np.c_[mz_, yi_])

        self.__xy = np.c_[_x, _y]
        self.__dim_xy = np.asarray([len(np.unique(_x)), len(np.unique(_y))],
                                   dtype=int)
        self.__px = coord2idx(x_array=self.__xy[:, 0], y_array=self.__xy[:, 1])

    def del_pixel(self, idx: Union[int, List[int]]) -> None:
        if isinstance(idx, list) or isinstance(idx, np.ndarray):
            rem_idx = idx
        else:
            rem_idx = [idx]

        self.__xy = np.delete(self.__xy, rem_idx, axis=0)
        self.__px = np.delete(self.__px, rem_idx)
        for i in sorted(rem_idx, reverse=True):
            del self.__msp[i]
        # Check that everything is fine
        if self.__xy.shape[0] != len(self.__px):
            raise RuntimeError('Wrong dimensions after deleting.')

    @property
    def ID(self):
        return self.__id

    @property
    def cmz(self):
        return self.__cmz

    @property
    def dim_xy(self):
        return self.__dim_xy

    @property
    def pixels_indices(self):
        return self.__px

    @property
    def pixels_coords(self):
        return self.__xy

    @property
    def msdata(self):
        return self.__msp

    def to_imzml(self, output_path):
        ux = np.sort(np.unique(self.__xy[:, 0]))
        uy = np.sort(np.unique(self.__xy[:, 1]))
        icoords = np.zeros((0, 2), dtype=int)
        for xy_ in self.__xy:
            icoords = np.vstack([icoords, np.asarray(
                [np.where(ux == xy_[0])[0][0], np.where(uy == xy_[1])[0][0]])])

        if self.__meta['ion_mode'] == 'ES+':
            polarity = 'positive'
        elif self.__meta['ion_mode'] == 'ES-':
            polarity = 'negative'
        else:
            raise ValueError('Invalid ion mode.')

        with ImzMLWriter(output_path, polarity=polarity) as w:
            for i in tqdm(range(len(self.__msp))):
                w.addSpectrum(mzs=self.__msp[i][:, 0],
                              intensities=self.__msp[i][:, 1],
                              coords=icoords[i, :] + 1)

    def gen_intensity_matrix(self) -> np.ndarray:
        if not self.__is_binned:
            raise RuntimeError('MSI peaks are not binned.')

        def _assign_val():
            pidx = np.isin(self.cmz, msp[:, 0])
            X[self.__px[i], pidx] = msp[:, 1]

        X = np.zeros((np.prod(self.__dim_xy), len(self.__cmz)), dtype=float)

        print('Generating intensity matrix ...')
        for i, msp in enumerate(tqdm(self.__msp)):
            _assign_val()

        if np.any(np.sum(X != 0, axis=0) == 0):
            raise RuntimeError

        return X
