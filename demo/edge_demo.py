import os
import sys
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

import gptools
from gptools import PACKAGE_ROOT

plt.ion()

# Load the test data:
# core_data contains density data as a function of r/a from the core of an
# Alcator C-Mod H-mode.
# These data are expected to be fit properly with a stationary covariance
# kernel.
pkl_kwargs = {'encoding': 'latin1'}
with open(os.path.join(PACKAGE_ROOT, '..', 'demo', 'sample_data_core.pkl'), 'rb') as f:
    core_data = pkl.load(f, **pkl_kwargs)
# edge_data contains density data as a function of r/a from the edge of the
# same Alcator C-Mod H-mode as core_data.
# When these datasets are combined, a non-stationary covariance kernel becomes
# necessary.
with open(os.path.join(PACKAGE_ROOT, '..', 'demo', 'sample_data_edge.pkl'), 'rb') as f:
    edge_data = pkl.load(f, **pkl_kwargs)

# Make some figures to hold our results:
f = plt.figure()
a_val = f.add_subplot(2, 1, 1)
a_val.errorbar(core_data['X'], core_data['y'], yerr=core_data['err_y'], label='core data', fmt='.')
a_val.errorbar(edge_data['X'], edge_data['y'], yerr=edge_data['err_y'], label='edge data', fmt='.')
a_grad = f.add_subplot(2, 1, 2)

X_star = np.linspace(0, 1.1, 400)

# TODO(ZanderKeith) for right now I'm only interested in the core and edge together, can come back to fill in the rest of the notebook later

# When the edge data are incorporated, a stationary kernel such as the SE is
# no longer appropriate:
# Either the region of rapid change will drive the fit to short values, or the
# gradual region will cause the rapid change to be oversmoothed.
# :py:class:`GibbsKernel1dTanh` was designed to fit nonstationary data like
# this where there is a smooth region and a rough region. An arbitrary length
# scale function can be selected by following this template. In addition, more
# powerful input warpings are provided in the :py:mod:`warping` submodule. See
# the manual for more details.
hp = (
    gptools.UniformJointPrior([[0.0, 20.0]]) *
    gptools.GammaJointPriorAlt([1.0, 0.5, 0.0, 1.0], [0.3, 0.25, 0.1, 0.1])
)
k_gibbs = gptools.GibbsKernel1dTanh(hyperprior=hp)
gp = gptools.GaussianProcess(k_gibbs)
gp.add_data(core_data['X'], core_data['y'], err_y=core_data['err_y'])
gp.add_data(edge_data['X'], edge_data['y'], err_y=edge_data['err_y'])
gp.add_data(0, 0, n=1)
gp.optimize_hyperparameters(verbose=True)
y_star, std_y_star = gp.predict(X_star)
gptools.univariate_envelope_plot(
    X_star,
    y_star,
    std_y_star,
    label='whole profile, Gibbs+tanh kernel',
    ax=a_val
)
grad_y_star, std_grad_y_star = gp.predict(X_star, n=1)
gptools.univariate_envelope_plot(
    X_star,
    grad_y_star,
    std_grad_y_star,
    label='whole profile, Gibbs+tanh kernel',
    ax=a_grad
)
# This could (and usually should!) be done with MCMC sampling exactly as shown
# above.

f.savefig(os.path.join(PACKAGE_ROOT, '..', 'demo', 'edge_demo_output.png'))