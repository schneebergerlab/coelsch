import logging

import numpy as np
import pandas as pd
from scipy.special import erf

import torch
from pomegranate.gmm import GeneralMixtureModel
from pomegranate import distributions as pmd

from .rhmm import RigidHMM
from .crossovers import detect_crossovers
from ..records import NestedData
from ..sim import simulate_doublets
from .. import stats
from ..defaults import DEFAULT_RANDOM_SEED
from ..main.logger import progress_bar

log = logging.getLogger('coelsch')
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)
DEFAULT_DEVICE = torch.device('cpu')


def _calc_d_prime(mu1, var1, mu2, var2):
    return abs(mu1 - mu2) / ((var1 + var2) / 2) ** 0.5


def _d_prime_to_bayes_error(d_prime):
    return 0.5 * (1 - erf(d_prime / (2 * np.sqrt(2))))


def detect_doublets(co_markers, co_preds, rhmm, n_doublets=1000,
                    freeze_doublet_dist=False, inertia=0.95,
                    batch_size=128, processes=1, rng=DEFAULT_RNG):

    if n_doublets > 1:
        n_sim = int(n_doublets)
    else:
        if not n_doublets:
            raise ValueError('n-doublets must be >0 for doublet detection')
        n_sim = int(len(co_markers) * n_doublets)

    log.info(f'Simulating {n_sim} doublets')
    sim_co_markers = simulate_doublets(co_markers, n_sim, rng=rng)
    sim_co_preds = detect_crossovers(
        sim_co_markers, rhmm,
        sample_paths=False,
        batch_size=batch_size,
        processes=processes,
    )

    barcodes = co_preds.barcodes
    log_acc_score_real = np.log(np.array([
        stats.accuracy_score(co_markers[cb], co_preds[cb])
        for cb in barcodes
    ]))

    log_acc_score_sim = np.log(np.array([
        stats.accuracy_score(sim_co_markers[cb], sim_co_preds[cb])
        for cb in sim_co_markers.barcodes
    ]))

    if log_acc_score_real.mean() < log_acc_score_sim.mean():
        log.warning('Real data looks more doublet-y than simulated data! '
                    'Doublet predictions may not be accurate')

    doublet_dist = pmd.Normal(
        [log_acc_score_sim.mean()],
        [log_acc_score_sim.var()],
        covariance_type='diag',
        frozen=freeze_doublet_dist
    )

    nondoublet_dist = pmd.Normal(
        [log_acc_score_real.mean()],
        [log_acc_score_real.var()],
        covariance_type='diag',
    )

    gmm = GeneralMixtureModel([nondoublet_dist, doublet_dist], inertia=inertia)
    gmm = gmm.fit(log_acc_score_real.reshape(-1, 1))

    mu_nondoublet = gmm.distributions[0].means.numpy()[0]
    var_nondoublet = gmm.distributions[0].covs.numpy()[0]
    mu_doublet = gmm.distributions[1].means.numpy()[0]
    var_doublet = gmm.distributions[1].covs.numpy()[0]

    d_prime_sim_real = _calc_d_prime(
        log_acc_score_sim.mean(), log_acc_score_sim.var(),
        mu_doublet, var_doublet
    )
    if d_prime_sim_real > 1.5:
        log.debug(f'Simulated doublets do not model true doublets well (d_prime = {d_prime_sim_real:.2f})')

    d_prime_dblt_nondblt = _calc_d_prime(
        mu_nondoublet, var_nondoublet,
        mu_doublet, var_doublet
    )
    bayes_error = _d_prime_to_bayes_error(d_prime_dblt_nondblt)
    if mu_nondoublet < mu_doublet:
        log.warning('Non-doublet class is unexpectedly noiser than doublet class. '
                    'Something may have gone wrong.')
    if d_prime_dblt_nondblt < 1.5:
        log.warning('Doublets and non-doublets do not separate well. Predictions may be poor')

    log.info('Classifying doublets and non-doublets using GMM with '
             f'Bayes error rate: {bayes_error:.2f}')

    doublet_probs = gmm.predict_proba(log_acc_score_real.reshape(-1, 1)).numpy()[:, 1]
    co_preds.add_metadata(
        doublet_probability=NestedData(
            levels=('cb',), dtype=float, data=dict(zip(barcodes, doublet_probs.tolist()))
        )
    )
    return co_preds
