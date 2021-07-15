#   Copyright 2018 by Paolo Inglese, National Phenome Centre, Imperial College
#   London
#   All rights reserved.
#   This file is part of DESI-MSI recalibration, and is released under the
#   "MIT License Agreement".
#   Please see the LICENSE file that should have been included as part of this
#   package.

import matplotlib
import matplotlib.style


def set_mpl_params():
    params = {
        'image.origin': 'lower',
        'image.interpolation': 'none',
        'image.cmap': 'viridis',
        'axes.grid': False,
        'savefig.dpi': 150,  # to adjust notebook inline plot size
        'axes.labelsize': 12,  # fontsize for x and y labels (was 10)
        'axes.titlesize': 12,
        'font.size': 12,  # was 10
        'legend.fontsize': 6,  # was 10
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'text.usetex': False,
        'figure.figsize': [5, 4],
        'font.family': 'serif',
    }
    matplotlib.rcParams.update(params)
    matplotlib.rcParams['pdf.fonttype'] = 42


def set_mpl_params_mod():
    params = {
        'image.origin': 'lower',
        'image.interpolation': 'none',
        'image.cmap': 'viridis',
        'axes.linewidth': 0.1,
        'axes.grid': False,
        'savefig.dpi': 300,  # to adjust notebook inline plot size
        'axes.labelsize': 5,  # fontsize for x and y labels (was 10)
        'axes.titlesize': 5,
        'font.size': 5,  # was 10
        'legend.fontsize': 6,  # was 10
        'xtick.labelsize': 5,
        'ytick.labelsize': 5,
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
    matplotlib.rcParams['pdf.fonttype'] = 42


def set_mpl_defaults():
    matplotlib.style.use('default')
