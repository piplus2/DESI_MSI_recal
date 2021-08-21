#   Copyright 2018 by Paolo Inglese, National Phenome Centre, Imperial College
#   London
#   All rights reserved.
#   This file is part of DESI-MSI recalibration, and is released under the
#   "MIT License Agreement".
#   Please see the LICENSE file that should have been included as part of this
#   package.

import argparse
import os
from typing import Dict

import numpy as np

from tools.functions import gen_ref_list, search_ref_masses, KDEMassRecal, \
    del_all_files_dir
from tools.msi import MSI


def __parse_arg():
    parser_ = argparse.ArgumentParser(description='DESI-MSI recalibration tool')
    parser_.add_argument('input', type=str,
                         help='Input imzML file (centroided).')
    parser_.add_argument('output', type=str, help='Output imzML file.')
    parser_.add_argument('roi', type=str,
                         help='Sample ROI mask CSV file. If set equal to '
                              '\'full\', the entire image is analyzed.')
    parser_.add_argument('--analyzer', choices=['tof', 'orbitrap'],
                         help='MS analyzer.')
    parser_.add_argument('--ion-mode', choices=['pos', 'neg'],
                         help='ES Polarization mode.')
    parser_.add_argument('--search-tol', default='auto',
                         help='Search tolerance expressed in ppm. If \'auto\', '
                              'default value for MS analyzer is used.')
    parser_.add_argument('--kde-bw', default='silverman',
                         help='KDE bandwidth. It can be numeric or '
                              '\'silverman\' (default=\'silverman\').')
    parser_.add_argument('--max-res-smooth', default='cv', dest='smooth',
                         help='Smoothing parameter for spline. It represents '
                              'the maximum sum of squared errors. If set to '
                              '\'cv\', it is determined by cross-validation '
                              '(default = \'cv\').')
    parser_.add_argument('--max-dispersion', default=10.0, type=float,
                         help='Max dispersion in ppm for outlier detection '
                              '(default=10.0).', dest='max_disp')
    parser_.add_argument('--min-coverage', default=75.0, type=float,
                         help='Min. coverage percentage for hits filtering '
                              '(default=75.0).')
    parser_.add_argument('--plot-ref-imgs', default=False, action='store_true',
                         dest='plot',
                         help='Save the intensity images of the reference '
                              'masses. It can slow down the process '
                              '(default=False).')
    parser_.add_argument('--parallel', action='store_true', dest='parallel',
                         default=False, help='Use multithreading.')
    return parser_


def set_params_dict(args_) -> Dict:
    default_max_tol = {'orbitrap': 20.0, 'tof': 100.0}
    params_ = {
        'input': args_.input,
        'output': args_.output,
        'roi': args_.roi,
        'analyzer': args_.analyzer,
        'bw': args_.kde_bw,
        'ion_mode': 'ES-' if args_.ion_mode == 'neg' else 'ES+',
        'max_tol': args_.search_tol if args_.search_tol != 'auto' else
        default_max_tol[args_.analyzer],
        'min_cov': args_.min_coverage,
        'max_disp': args_.max_disp,
        'max_degree': 1 if args_.analyzer == 'orbitrap' else 5,
        'parallel': args_.parallel,
        'plot': args_.plot,
        'smooth': args_.smooth,
        'transform': 'none' if args_.analyzer == 'orbitrap' else 'sqrt'
    }
    return params_


def main():
    parser = __parse_arg()
    args = parser.parse_args()

    params = set_params_dict(args)
    # Load MSI
    msi = MSI(imzml=params['input'], meta=params)
    # Load ROI
    if params['roi'] != 'full':
        roi = np.loadtxt(params['roi'], delimiter=',')
        if not np.all(roi.shape == msi.dim_xy[::-1]):
            raise ValueError('ROI has incompatible dimensions.')
        print('Num. ROI pixels = {}'.format(int(np.sum(roi))))
        # Remove non-ROI pixels
        outpx = np.where(roi.ravel() == 0)[0]
        delpx = np.where(np.isin(msi.pixels_indices, outpx))[0]
        delpx = np.sort(delpx)
        msi.del_pixel(list(delpx))

    # Creating match images dir
    if params['plot']:
        plots_dir = os.path.join(
            os.path.dirname(params['output']), msi.ID + '_recal_imgs')
        print('Intensity images will be saved in {}'.format(plots_dir))
        if not os.path.isdir(plots_dir):
            print('Creating plots dir {} ...'.format(plots_dir))
            os.makedirs(plots_dir)
        else:
            del_all_files_dir(plots_dir)
    else:
        plots_dir = None

    # RECALIBRATION ---------------------------

    ref_masses = gen_ref_list(ion_mode=params['ion_mode'], verbose=True)

    print('Searching lock masses within {} ppm ...'.format(
        np.round(params['max_tol'], 2)))
    matches = search_ref_masses(msiobj=msi, ref_masses=ref_masses,
                                max_tolerance=params['max_tol'], top_n=-1)

    print('Removing hits found in less than {} % of ROI pixels ...'.format(
        params['min_cov']))
    matches = {m: matches[m] for m in matches.keys() if
               len(np.unique(matches[m]['pixel'])) / len(
                   msi.pixels_indices) * 100.0 >= params['min_cov']}

    print('Num. lock masses with coverage >= {} % = {}'.format(
        np.round(params['min_cov'], 2), len(matches)))

    recal = \
        KDEMassRecal(min_pct=params['min_cov'],
                     transform=params['transform'],
                     max_poly_degree=params['max_degree'],
                     max_disp_ppm=params['max_disp'],
                     kde_bw=params['bw'],
                     grid_size=2**10, smooth=params['smooth'],
                     parallel=params['parallel'], plot=params['plot'],
                     plot_dir=plots_dir, plot_dim_xy=msi.dim_xy)
    msi = recal.recalibrate(msi, matches)

    print('Saving recalibrated ROI imzML ...')
    msi.to_imzml(output_path=params['output'])


if __name__ == '__main__':
    main()
