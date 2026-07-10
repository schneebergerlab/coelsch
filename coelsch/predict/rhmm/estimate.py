import logging

import numpy as np

import torch
from pomegranate import distributions as pmd
from pomegranate.gmm import GeneralMixtureModel
from pomegranate._utils import _update_parameter


from coelsch.signal import align_foreground_column, detect_heterozygous_bins
from .dists import NegativeBinomial, ZeroInflated
from .utils import numpy_to_torch


def _estimate_alpha(m, v):
    if m <= 0: 
        return 0.1  # fallback
    return max((v - m) / (m**2), 1e-6)


@torch.no_grad()
def estimate_haploid_emissions(X, window=40, dist_type="poisson"):
    """
    Estimate emission parameters for haploid data.

    Returns
    -------
    tuple(dict, dict, float)
        fg_params, bg_params, empty_fraction
    """
    X_ordered = align_foreground_column(X, window)
    if any(isinstance(x, np.ma.MaskedArray) for x in X_ordered):
        X_flattened = np.ma.concatenate(X_ordered)
    else:
        X_flattened = np.concatenate(X_ordered)

    init_fg_mean, init_bg_mean = np.mean(X_flattened, axis=0)

    if dist_type == "poisson":
        zid = ZeroInflated(pmd.Poisson([init_fg_mean, init_bg_mean])).fit(numpy_to_torch(X_flattened))
        fg_lambda, bg_lambda = zid.distribution.lambdas.numpy()
        fg_params = {"lambda": float(fg_lambda)}
        bg_params = {"lambda": float(bg_lambda)}

    elif dist_type == "nb":
        init_fg_var, init_bg_var = np.var(X_flattened, axis=0, ddof=1)
        init_fg_alpha = _estimate_alpha(init_fg_mean, init_fg_var)
        init_bg_alpha = _estimate_alpha(init_bg_mean, init_bg_var)
        zid = ZeroInflated(
            NegativeBinomial(means=[init_fg_mean, init_bg_mean],
                             alphas=[init_fg_alpha, init_bg_alpha], frozen=False)
        ).fit(numpy_to_torch(X_flattened))
        fg_mean, bg_mean = zid.distribution.means.numpy()
        fg_alpha, bg_alpha = zid.distribution.alphas.numpy()
        fg_params = {"mean": float(fg_mean), "alpha": float(fg_alpha)}
        bg_params = {"mean": float(bg_mean), "alpha": float(bg_alpha)}

    else:
        raise ValueError(f"Unknown dist_type: {dist_type}")

    fg_params['empty_fraction'] = float(zid.priors.numpy()[0])
    bg_params['empty_fraction'] = float(zid.priors.numpy()[1])
    return fg_params, bg_params


@torch.no_grad()
def _fit_constrainted_diploid(X, init_fg_mean, init_bg_mean, *, priors=None,
                              dist_type='poisson', init_fg_alpha=None, init_bg_alpha=None,
                              max_iter=1000, tol=0.1):

    if dist_type == 'poisson':
        gmm = GeneralMixtureModel([
            ZeroInflated(pmd.Poisson([init_fg_mean * 2, init_bg_mean])),
            ZeroInflated(pmd.Poisson([init_fg_mean, init_fg_mean])),
        ])
        mean_attr = 'lambdas'
    else:
        gmm = GeneralMixtureModel([
            ZeroInflated(NegativeBinomial([init_fg_mean * 2, init_bg_mean], [init_fg_alpha, init_bg_alpha])),
            ZeroInflated(NegativeBinomial([init_fg_mean, init_fg_mean], [init_fg_alpha, init_fg_alpha])),
        ])
        mean_attr = 'means'

    logp = None
    for i in range(max_iter):
        last_logp = logp
        logp = gmm.summarize(X, priors=priors)

        if i > 0:
            improvement = logp - last_logp
            if improvement < tol:
                break

        gmm.from_summaries()

        hom_means = getattr(gmm.distributions[0].distribution, mean_attr)
        het_means = getattr(gmm.distributions[1].distribution, mean_attr)

        # average across the means to get a single foreground result
        fg_mean = float(np.mean([hom_means.numpy()[0] / 2, *het_means.numpy()]))
        bg_mean = float(hom_means.numpy()[1])
        _update_parameter(hom_means, [fg_mean * 2, bg_mean])
        _update_parameter(het_means, [fg_mean, fg_mean])

        if dist_type == 'nb':
            hom_alphas = gmm.distributions[0].distribution.alphas
            het_alphas = gmm.distributions[1].distribution.alphas
            fg_alpha = float(np.mean([hom_alphas.detach().numpy()[0], *het_alphas.detach().numpy()]))
            bg_alpha = float(hom_alphas.detach().numpy()[1])
            _update_parameter(hom_alphas, [fg_alpha, bg_alpha])
            _update_parameter(het_alphas, [fg_alpha, fg_alpha])

        # enforce equivalent zero inflation across distributions
        hom_empty = gmm.distributions[0].priors
        het_empty = gmm.distributions[1].priors
        fg_empty = float(np.mean([hom_empty.numpy()[0], *het_empty.numpy()]))
        bg_empty = float(hom_empty.numpy()[1])
        _update_parameter(hom_empty, [fg_empty, bg_empty])
        _update_parameter(het_empty, [fg_empty, fg_empty])

    gmm._reset_cache()
    if dist_type == 'poisson':
        return fg_mean, bg_mean, fg_empty, bg_empty
    return fg_mean, bg_mean, fg_alpha, bg_alpha, fg_empty, bg_empty


@torch.no_grad()
def estimate_diploid_emissions_ordered(X_ordered, window=40, dist_type="poisson"):
    """
    Estimate emission parameters for diploid data with known foreground column ordering.

    Parameters
    ----------
    X_ordered : list of np.ndarray
        List of marker count arrays with shape (bins, 2), foreground must be column 0.
    window : int, optional
        Width of the smoothing window (default is 40).
    dist_type : str, optional
        Either "poisson" or "nb".

    Returns
    -------
    tuple(dict, dict, float)
        fg_params, bg_params, empty_fraction
    """
    mask = detect_heterozygous_bins(X_ordered, window)
    if any(isinstance(x, np.ma.MaskedArray) for x in X_ordered):
        X_flattened = np.ma.concatenate(X_ordered)
    else:
        X_flattened = np.concatenate(X_ordered)
    mask = np.concatenate(mask)

    init_fg_mean, init_bg_mean = X_flattened[mask, 1].mean(), X_flattened[~mask, 1].mean()
    priors = np.stack([1 - mask.astype(float), mask.astype(float)], axis=-1)

    if dist_type == "poisson":

        fg_mean, bg_mean, fg_empty, bg_empty = _fit_constrainted_diploid(
            numpy_to_torch(X_flattened), init_fg_mean, init_bg_mean, priors=priors,
        )

        fg_params = {'lambda': float(fg_mean), 'empty_fraction': fg_empty}
        bg_params = {'lambda': float(bg_mean), 'empty_fraction': bg_empty}

    elif dist_type == "nb":
        init_fg_var, init_bg_var = X_flattened[mask, 1].var(ddof=1), X_flattened[~mask, 1].var(ddof=1)
        init_fg_alpha = _estimate_alpha(init_fg_mean, init_fg_var)
        init_bg_alpha = _estimate_alpha(init_bg_mean, init_bg_var)

        fg_mean, bg_mean, fg_alpha, bg_alpha, fg_empty, bg_empty = _fit_constrainted_diploid(
            numpy_to_torch(X_flattened), init_fg_mean, init_bg_mean, priors=priors,
            dist_type='nb', init_fg_alpha=init_fg_alpha, init_bg_alpha=init_bg_alpha
        )

        fg_params = {"mean": float(fg_mean), "alpha": float(fg_alpha), "empty_fraction": fg_empty}
        bg_params = {"mean": float(bg_mean), "alpha": float(bg_alpha), "empty_fraction": bg_empty}

    else:
        raise ValueError(f"Unknown dist_type: {dist_type}")

    return fg_params, bg_params



def estimate_diploid_emissions_f2(X, window=40, dist_type="poisson"):
    """
    Estimate Poisson parameters for diploid F2 data by reordering and calling the ordered estimator.

    Parameters
    ----------
    X : list of np.ndarray
        List of marker count arrays with shape (bins, 2).
    window : int, optional
        Width of the smoothing window (default is 40).
    dist_type : str, optional
        Either "poisson" or "nb".

    Returns
    -------
    tuple(dict, dict, float)
        fg_params, bg_params, empty_fraction
    """
    X_ordered = align_foreground_column(X, window)
    return estimate_diploid_emissions_ordered(X_ordered, window, dist_type=dist_type)
