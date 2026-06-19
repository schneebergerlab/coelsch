import numpy as np

import torch
from pomegranate import distributions as pmd
from pomegranate.gmm import GeneralMixtureModel
from pomegranate._utils import _update_parameter

from coelsch.signal import align_foreground_column, detect_heterozygous_bins, smooth_counts_sum
from .dists import NegativeBinomial, ZeroInflated
from .utils import numpy_to_torch


def _concat_arrays(X):
    if any(isinstance(x, np.ma.MaskedArray) for x in X):
        arr = np.ma.concatenate(X)
        if np.asarray(arr.mask).shape == ():
            arr.mask = np.zeros_like(arr.data, dtype=bool)
        return arr
    return np.concatenate(X)


def _estimate_alpha(m, v):
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)

    alpha = np.maximum((v - m) / np.maximum(m**2, 1e-12), 1e-6)
    alpha = np.where(m <= 0, 0.1, alpha)

    return float(alpha) if alpha.ndim == 0 else alpha


def _estimate_init_params(X, dosages):
    dosages = np.array(dosages)

    colmeans = np.mean(X, axis=0)
    fg_mask = dosages > 0
    fg_mean = np.mean(colmeans[fg_mask] / dosages[fg_mask])
    bg_mean = np.mean(colmeans[~fg_mask])

    colvars = np.var(X, axis=0, ddof=1)
    fg_alpha = np.mean(_estimate_alpha(colmeans[fg_mask], colvars[fg_mask]))
    bg_alpha = np.mean(_estimate_alpha(colmeans[~fg_mask], colvars[~fg_mask]))

    return fg_mean, bg_mean, fg_alpha, bg_alpha


def _estimate_init_params_multicomp(X, y, component_dosages):
    fg_means, bg_means = [], []
    fg_alphas, bg_alphas = [], []
    weight = []

    for comp in np.unique(y):
        mask = y == comp
        fg_mean, bg_mean, fg_alpha, bg_alpha = _estimate_init_params(
            X[mask], component_dosages[comp]
        )
        fg_means.append(fg_mean)
        bg_means.append(bg_mean)
        fg_alphas.append(fg_alpha)
        bg_alphas.append(bg_alpha)
        weight.append(mask.mean())

    return (
        np.average(fg_means, weights=weight),
        np.average(bg_means, weights=weight),
        np.average(fg_alphas, weights=weight),
        np.average(bg_alphas, weights=weight),
    )


def _calculate_model_params(component_dosages, fg_mean, bg_mean,
                            fg_alpha=None, bg_alpha=None,
                            fg_prior=None, bg_prior=None):
    component_dosages = np.array(component_dosages)
    means = component_dosages * fg_mean # multiply dosage by foreground
    means[component_dosages == 0] = bg_mean # zero dosages are replaced with background

    alphas = None
    if fg_alpha is not None and bg_alpha is not None:
        alphas = np.full_like(component_dosages, fill_value=bg_alpha)
        alphas[component_dosages > 0] = fg_alpha

    priors = None
    if fg_prior is not None and bg_prior is not None:
        priors = np.full_like(component_dosages, fill_value=bg_prior)
        priors[component_dosages > 0] = fg_prior

    return means, alphas, priors


def _format_params(params, dist_type):
    fg_mean, bg_mean, fg_alpha, bg_alpha, fg_prior, bg_prior = params

    if dist_type == 'poisson':
        return (
            {'lambda': float(fg_mean), 'empty_fraction': float(fg_prior)},
            {'lambda': float(bg_mean), 'empty_fraction': float(bg_prior)},
        )

    if dist_type == 'nb':
        return (
            {'mean': float(fg_mean), 'alpha': float(fg_alpha), 'empty_fraction': float(fg_prior)},
            {'mean': float(bg_mean), 'alpha': float(bg_alpha), 'empty_fraction': float(bg_prior)},
        )

    raise ValueError(f'Unknown dist_type: {dist_type}')

def _constrain_params(model, component_dosages, dist_type, update=True):

    mean_attr = 'lambdas' if dist_type == 'poisson' else 'means'

    # for instances where model is actually just a distribution
    if not isinstance(model, GeneralMixtureModel):
        model_dists = (model, )
        component_dosages = (component_dosages, )
    else:
        model_dists = model.distributions

    fg_means, bg_means = [], []
    fg_alphas, bg_alphas = [], []
    fg_priors, bg_priors = [], []
    for dist, dosage in zip(model_dists, component_dosages):
        means = getattr(dist.distribution, mean_attr).numpy()
        if dist_type == 'poisson':
            alphas = np.full_like(means, fill_value=np.nan)
        elif dist_type == 'nb':
            alphas = getattr(dist.distribution, "alphas").numpy()
        else:
            raise NotImplementedError()
        priors = getattr(dist, 'priors').numpy()

        for d, m, a, p in zip(dosage, means, alphas, priors):
            if d > 0:
                fg_means.append(m / d)
                fg_alphas.append(a)
                fg_priors.append(p)
            else:
                bg_means.append(m)
                bg_alphas.append(a)
                bg_priors.append(p)

    fg_mean = np.mean(fg_means)
    bg_mean = np.mean(bg_means)

    if dist_type == 'nb':
        fg_alpha = np.mean(fg_alphas)
        bg_alpha = np.mean(bg_alphas)
    else:
        fg_alpha, bg_alpha = None, None

    fg_prior = np.mean(fg_priors)
    bg_prior = np.mean(bg_priors)

    if update:
        # reapply the parameters to the model
        means, alphas, priors = _calculate_model_params(
            component_dosages, fg_mean, bg_mean, fg_alpha, bg_alpha, fg_prior, bg_prior
        )
        if alphas is None:
            alphas = [None] * len(model_dists)
        if priors is None:
            priors = [None] * len(model_dists)

        for dist_mean, dist_alpha, dist_prior, dist in zip(means, alphas, priors, model_dists):
            _update_parameter(
                getattr(dist.distribution, mean_attr), dist_mean
            )
            if dist_type == 'nb':
                _update_parameter(
                    getattr(dist.distribution, 'alphas'), dist_alpha
                )
            if dist_prior is not None:
                _update_parameter(
                    getattr(dist, 'priors'), dist_prior
                )

    return fg_mean, bg_mean, fg_alpha, bg_alpha, fg_prior, bg_prior


def _build_zid(dosages, fg_mean, bg_mean, fg_alpha=None, bg_alpha=None, dist_type='poisson'):

    means, alphas, _ = _calculate_model_params(dosages, fg_mean, bg_mean, fg_alpha, bg_alpha)

    if dist_type == 'poisson':
        dist = pmd.Poisson(means)
    elif dist_type == 'nb':
        dist = NegativeBinomial(means=means, alphas=alphas)
    else:
        raise NotImplementedError()

    return ZeroInflated(dist)


def _build_gmm(component_dosages, *, fg_mean, bg_mean, priors=None,
               fg_alpha=None, bg_alpha=None, dist_type="poisson"):
    dists = []
    for dosage in component_dosages:
        dists.append(_build_zid(dosage, fg_mean, bg_mean, fg_alpha, bg_alpha, dist_type))

    return GeneralMixtureModel(dists, priors=priors)


@torch.no_grad()
def _fit_constrained_model(model, X, dosages, priors=None,
                           dist_type="poisson", max_iter=1000, tol=0.1):
    X = numpy_to_torch(X)
    is_gmm = isinstance(model, GeneralMixtureModel)
    logp = None
    for i in range(max_iter):
        last_logp = logp
        if is_gmm:
            logp = model.summarize(X, priors=priors)
        else:
            # dists don't have priors
            logp = model.summarize(X)

        if i > 0:
            improvement = logp - last_logp
            if improvement < tol:
                break

        model.from_summaries()
        new_params = _constrain_params(
            model, dosages, dist_type
        )

    model._reset_cache()
    return new_params


@torch.no_grad()
def estimate_haploid_emissions(X, window=40, dist_type="poisson"):

    dosages = [1.0, 0.0] # fg bg
    
    X_ordered = align_foreground_column(X, window)
    X_flattened = _concat_arrays(X_ordered)
    init_fg_mean, init_bg_mean, init_fg_alpha, init_bg_alpha = _estimate_init_params(
        X_flattened, dosages
    )

    zid = _build_zid(
        dosages=dosages,
        fg_mean=init_fg_mean,
        bg_mean=init_bg_mean,
        fg_alpha=init_fg_alpha,
        bg_alpha=init_bg_alpha,
        dist_type=dist_type
    )
    # model doesn't need constraining since theres only one param for fg and bg
    zid = zid.fit(numpy_to_torch(X_flattened))

    params = _constrain_params(zid, dosages, dist_type=dist_type, update=False)
    return _format_params(params, dist_type)


def estimate_diploid_emissions_backcross(X, window=40, dist_type="poisson"):

    # data is already ordered with states 2fg, bg and fg, fg
    X_ordered = X
    component_dosages = [
        [2.0, 0.0], # hom: 2*fg, bg
        [1.0, 1.0], # het: fg, fg
    ]

    init_comp_pred = detect_heterozygous_bins(X_ordered, window)
    X_flattened = _concat_arrays(X_ordered)
    init_comp_pred = np.concatenate(init_comp_pred).astype(int)
    init_fg_mean, init_bg_mean, init_fg_alpha, init_bg_alpha = _estimate_init_params_multicomp(
        X_flattened, init_comp_pred, component_dosages,
    )
    label_priors = np.stack([1 - init_comp_pred.astype(float), init_comp_pred.astype(float)], axis=-1)

    gmm = _build_gmm(
        component_dosages,
        fg_mean=init_fg_mean,
        bg_mean=init_bg_mean,
        priors=label_priors.mean(axis=0),
        fg_alpha=init_fg_alpha,
        bg_alpha=init_bg_alpha,
        dist_type=dist_type,
    )
    params = _fit_constrained_model(
        gmm, X_flattened, component_dosages,
        priors=label_priors,
        dist_type=dist_type
    )
    return _format_params(params, dist_type)


def estimate_diploid_emissions_f2(X, window=40, dist_type="poisson"):
    # reorder columns and run as backcross
    return estimate_diploid_emissions_backcross(
        align_foreground_column(X, window),
        window,
        dist_type=dist_type,
    )


def estimate_diploid_emissions_testcross(X, window=40, dist_type='poisson'):

    dosages = [1.0, 1.0, 0.0] # fg fg bg

    # reordering columns 1 and 2 gives us a haploid-like problem
    X_ordered = align_foreground_column(X, window, columns=(1, 2))    
    X_flattened = _concat_arrays(X_ordered)
    init_fg_mean, init_bg_mean, init_fg_alpha, init_bg_alpha = _estimate_init_params(
        X_flattened, dosages
    )
    zid = _build_zid(
        dosages=dosages,
        fg_mean=init_fg_mean,
        bg_mean=init_bg_mean,
        fg_alpha=init_fg_alpha,
        bg_alpha=init_bg_alpha,
        dist_type=dist_type
    )
    params = _fit_constrained_model(
        zid, X_flattened, dosages,
        dist_type=dist_type
    )
    return _format_params(params, dist_type)


def _align_three_way_columns(X, window=40):
    X_ordered = align_foreground_column(X, window=window, columns=(1, 2))

    for x in X_ordered:
        smoothed = smooth_counts_sum(x, window)
        roll_idx = 2 * smoothed[:, 0] < smoothed[:, (1, 2)].sum(axis=1)
        x[roll_idx] = np.roll(x[roll_idx], shift=-1, axis=1)

    return X_ordered


def estimate_diploid_emissions_three_way(X, window=40, dist_type='poisson'):
    # reorder A, B, C columns from a ((A*B)*(A*C)) cross
    X_ordered = _align_three_way_columns(X, window)
    # after reordering the columns should represent:
    # 2:0:0 (bins with AA haplotype)
    # 1:1:0 (bins with AB, AC, or BC haplotype)
    component_dosages = [
        [2.0, 0.0, 0.0], # hom: 2*fg, bg, bg
        [1.0, 1.0, 0.0], # het: fg, fg, bg
    ]

    # ignore last channel and make init_comp_pred as if a normal f2
    init_comp_pred = detect_heterozygous_bins(
        [x[:, (0, 1)] for x in X_ordered],
        window
    )
    X_flattened = _concat_arrays(X_ordered)
    init_comp_pred = np.concatenate(init_comp_pred).astype(int)
    init_fg_mean, init_bg_mean, init_fg_alpha, init_bg_alpha = _estimate_init_params_multicomp(
        X_flattened, init_comp_pred, component_dosages,
    )
    label_priors = np.stack([1 - init_comp_pred.astype(float), init_comp_pred.astype(float)], axis=-1)

    gmm = _build_gmm(
        component_dosages,
        fg_mean=init_fg_mean,
        bg_mean=init_bg_mean,
        priors=label_priors.mean(axis=0),
        fg_alpha=init_fg_alpha,
        bg_alpha=init_bg_alpha,
        dist_type=dist_type,
    )
    params = _fit_constrained_model(
        gmm, X_flattened, component_dosages,
        priors=label_priors,
        dist_type=dist_type
    )
    return _format_params(params, dist_type)


def estimate_diploid_emissions_four_way(X, window=40, dist_type='poisson'):
    # convert to a haploid problem
    X_pairs = [x[:, (0, 1)] for x in X] + [x[:, (2, 3)] for x in X]
    return estimate_haploid_emissions(X_pairs, window, dist_type=dist_type)


def estimate_emissions(X, experiment_params, window=40, dist_type='poisson'):
    if experiment_params.genotyping_strategy == 'recombinant':
        return estimate_haploid_emissions(X, window, dist_type=dist_type)

    if experiment_params.ploidy == 1 or experiment_params.crossing_strategy == 'f1':
        return estimate_haploid_emissions(X, window, dist_type=dist_type)

    strategy = experiment_params.crossing_strategy
    if strategy == 'backcross':
        return estimate_diploid_emissions_backcross(X, window, dist_type=dist_type)
    if strategy == 'f2':
        return estimate_diploid_emissions_f2(X, window, dist_type=dist_type)
    if strategy == 'testcross':
        return estimate_diploid_emissions_testcross(X, window, dist_type=dist_type)
    if strategy == 'three_way':
        return estimate_diploid_emissions_three_way(X, window, dist_type=dist_type)
    if strategy == 'four_way':
        return estimate_diploid_emissions_four_way(X, window, dist_type=dist_type)

    raise ValueError(f'Unsupported crossing_strategy: {strategy}')
