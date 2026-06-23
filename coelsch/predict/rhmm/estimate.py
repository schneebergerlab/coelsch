import logging
import numpy as np
import torch

from pomegranate import distributions as pmd
from pomegranate.gmm import GeneralMixtureModel
from pomegranate._utils import _update_parameter

from coelsch.signal import smooth_counts_sum
from .dists import NegativeBinomial, ZeroInflated
from . import utils

log = logging.getLogger('coelsch')


def _require_n_haplotypes(X, expected, strategy):
    if len(X) == 0:
        raise ValueError("X is empty")
    observed = X[0].shape[1]
    if observed != expected:
        raise ValueError(
            f"{strategy} expects {expected} haplotype channels, "
            f"but X has {observed}"
        )


def _estimate_alpha(m, v):
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)

    alpha = np.maximum((v - m) / np.maximum(m**2, 1e-12), 1e-6)
    alpha = np.where(m <= 0, 0.1, alpha)

    return float(alpha) if alpha.ndim == 0 else alpha


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=axis, keepdims=True)


def _shrink_to_shared(x, strength=1.0, fallback=1.0):
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)

    if finite.any():
        shared = np.nanmean(x)
    else:
        shared = fallback

    x = np.where(finite, x, shared)
    return shared + strength * (x - shared)


def _broadcast_param(x, n_haplotypes):
    if x is None:
        return None

    x = np.asarray(x, dtype=float)

    if x.ndim == 0:
        return np.full(n_haplotypes, float(x), dtype=float)

    if x.shape != (n_haplotypes,):
        raise ValueError(
            f"parameter has shape {x.shape}, expected {(n_haplotypes,)}"
        )

    return x


def _calculate_model_params(
    component_dosages,
    fg_mean,
    bg_mean,
    fg_alpha=None,
    bg_alpha=None,
    fg_empty_fraction=None,
    bg_empty_fraction=None,
):
    component_dosages = np.asarray(component_dosages, dtype=float)

    if component_dosages.ndim == 1:
        n_haplotypes = component_dosages.shape[0]
    elif component_dosages.ndim == 2:
        n_haplotypes = component_dosages.shape[1]
    else:
        raise ValueError("component_dosages must be 1D or 2D")

    fg_mean = _broadcast_param(fg_mean, n_haplotypes)
    bg_mean = _broadcast_param(bg_mean, n_haplotypes)

    means = np.where(
        component_dosages > 0,
        component_dosages * fg_mean,
        bg_mean,
    )

    alphas = None
    if fg_alpha is not None and bg_alpha is not None:
        fg_alpha = _broadcast_param(fg_alpha, n_haplotypes)
        bg_alpha = _broadcast_param(bg_alpha, n_haplotypes)

        alphas = np.where(
            component_dosages > 0,
            fg_alpha,
            bg_alpha,
        )

    empty_fractions = None
    if fg_empty_fraction is not None and bg_empty_fraction is not None:
        fg_empty_fraction = _broadcast_param(fg_empty_fraction, n_haplotypes)
        bg_empty_fraction = _broadcast_param(bg_empty_fraction, n_haplotypes)

        empty_fractions = np.where(
            component_dosages > 0,
            fg_empty_fraction,
            bg_empty_fraction,
        )

    return means, alphas, empty_fractions


def _format_params(params, dist_type):
    fg_mean, bg_mean, fg_alpha, bg_alpha, fg_empty_fraction, bg_empty_fraction = params

    if dist_type == "poisson":
        return (
            {
                "lambda": np.asarray(fg_mean, dtype=float),
                "empty_fraction": np.asarray(fg_empty_fraction, dtype=float),
            },
            {
                "lambda": np.asarray(bg_mean, dtype=float),
                "empty_fraction": np.asarray(bg_empty_fraction, dtype=float),
            },
        )

    if dist_type == "nb":
        return (
            {
                "mean": np.asarray(fg_mean, dtype=float),
                "alpha": np.asarray(fg_alpha, dtype=float),
                "empty_fraction": np.asarray(fg_empty_fraction, dtype=float),
            },
            {
                "mean": np.asarray(bg_mean, dtype=float),
                "alpha": np.asarray(bg_alpha, dtype=float),
                "empty_fraction": np.asarray(bg_empty_fraction, dtype=float),
            },
        )

    raise ValueError(f"Unknown dist_type: {dist_type}")


def _build_zid(
    dosages,
    fg_mean,
    bg_mean,
    fg_alpha=None,
    bg_alpha=None,
    dist_type="poisson",
):
    means, alphas, _ = _calculate_model_params(
        dosages,
        fg_mean,
        bg_mean,
        fg_alpha,
        bg_alpha,
    )

    if dist_type == "poisson":
        dist = pmd.Poisson(means)
    elif dist_type == "nb":
        dist = NegativeBinomial(means=means, alphas=alphas)
    else:
        raise NotImplementedError()

    return ZeroInflated(dist)


def _build_gmm(
    component_dosages,
    *,
    fg_mean,
    bg_mean,
    label_priors=None,
    fg_alpha=None,
    bg_alpha=None,
    dist_type="poisson",
):
    dists = []

    for dosage in component_dosages:
        dists.append(
            _build_zid(
                dosage,
                fg_mean,
                bg_mean,
                fg_alpha,
                bg_alpha,
                dist_type,
            )
        )

    return GeneralMixtureModel(dists, priors=label_priors)


def _estimate_init_params_soft(
    X,
    label_priors,
    component_dosages,
    pseudocount=1.0,
    channel_strength=1.0,
):
    data, valid = utils.data_and_mask(X)

    label_priors = np.asarray(label_priors, dtype=float)
    component_dosages = np.asarray(component_dosages, dtype=float)

    n_bins, n_haplotypes = data.shape

    if label_priors.shape[0] != n_bins:
        raise ValueError("label_priors and X have different numbers of rows")

    if component_dosages.shape[1] != n_haplotypes:
        raise ValueError("component_dosages and X have different channel counts")

    fg_mean = np.full(n_haplotypes, np.nan, dtype=float)
    bg_mean = np.full(n_haplotypes, np.nan, dtype=float)
    fg_alpha = np.full(n_haplotypes, np.nan, dtype=float)
    bg_alpha = np.full(n_haplotypes, np.nan, dtype=float)

    for h in range(n_haplotypes):
        fg_wsum = 0.0
        fg_sum = 0.0
        fg_sumsq = 0.0

        bg_wsum = 0.0
        bg_sum = 0.0
        bg_sumsq = 0.0

        for comp, dosage in enumerate(component_dosages[:, h]):
            w = label_priors[:, comp] * valid[:, h]

            if dosage > 0:
                values = data[:, h] / dosage
                fg_wsum += w.sum()
                fg_sum += np.sum(w * values)
                fg_sumsq += np.sum(w * values**2)
            else:
                values = data[:, h]
                bg_wsum += w.sum()
                bg_sum += np.sum(w * values)
                bg_sumsq += np.sum(w * values**2)

        if fg_wsum > 0:
            mean = (fg_sum + pseudocount) / (fg_wsum + pseudocount)
            raw_mean = fg_sum / fg_wsum
            var = max(fg_sumsq / fg_wsum - raw_mean**2, 0.0)

            fg_mean[h] = mean
            fg_alpha[h] = _estimate_alpha(mean, var)

        if bg_wsum > 0:
            mean = (bg_sum + pseudocount) / (bg_wsum + pseudocount)
            raw_mean = bg_sum / bg_wsum
            var = max(bg_sumsq / bg_wsum - raw_mean**2, 0.0)

            bg_mean[h] = mean
            bg_alpha[h] = _estimate_alpha(mean, var)

    fg_mean = _shrink_to_shared(fg_mean, channel_strength, fallback=1.0)
    bg_mean = _shrink_to_shared(bg_mean, channel_strength, fallback=0.1)
    fg_alpha = _shrink_to_shared(fg_alpha, channel_strength, fallback=0.1)
    bg_alpha = _shrink_to_shared(bg_alpha, channel_strength, fallback=0.1)

    return fg_mean, bg_mean, fg_alpha, bg_alpha


def _haplotype_pattern_priors(
    X,
    component_dosages,
    window=40,
    temperature=0.05,
    floor=0.02,
):
    component_dosages = np.asarray(component_dosages, dtype=float)

    dosage_totals = component_dosages.sum(axis=1, keepdims=True)
    if np.any(dosage_totals <= 0):
        raise ValueError("all component dosage vectors must have non-zero dosage")

    dosage_props = component_dosages / dosage_totals
    priors = []

    temperature = max(float(temperature), 1e-12)

    for x in X:
        x = utils.as_float_array(x)
        smoothed = utils.as_float_array(smooth_counts_sum(x, window))

        row_sum = smoothed.sum(axis=1, keepdims=True)
        zero_rows = row_sum[:, 0] <= 0

        obs_props = np.divide(
            smoothed,
            row_sum,
            out=np.zeros_like(smoothed, dtype=float),
            where=row_sum > 0,
        )

        dist2 = ((obs_props[:, None, :] - dosage_props[None, :, :]) ** 2).sum(axis=2)
        p = _softmax(-dist2 / temperature, axis=1)

        if np.any(zero_rows):
            p[zero_rows] = 1.0 / component_dosages.shape[0]

        p = (1.0 - floor) * p + floor / component_dosages.shape[0]
        priors.append(p)

    return np.concatenate(priors, axis=0)


def _constrain_params(
    model,
    component_dosages,
    dist_type,
    update=True,
    channel_strength=1.0,
):
    mean_attr = "lambdas" if dist_type == "poisson" else "means"

    model_dists = model.distributions
    component_dosages = np.asarray(component_dosages, dtype=float)

    n_haplotypes = component_dosages.shape[1]

    fg_means = [[] for _ in range(n_haplotypes)]
    bg_means = [[] for _ in range(n_haplotypes)]
    fg_alphas = [[] for _ in range(n_haplotypes)]
    bg_alphas = [[] for _ in range(n_haplotypes)]
    fg_priors = [[] for _ in range(n_haplotypes)]
    bg_priors = [[] for _ in range(n_haplotypes)]

    for dist, dosage in zip(model_dists, component_dosages):
        means = utils.torch_to_numpy(getattr(dist.distribution, mean_attr))

        if dist_type == "poisson":
            alphas = np.full_like(means, fill_value=np.nan, dtype=float)
        elif dist_type == "nb":
            alphas = utils.torch_to_numpy(getattr(dist.distribution, "alphas"))
        else:
            raise NotImplementedError()

        priors = utils.torch_to_numpy(getattr(dist, "priors"))

        for h, (d, m, a, p) in enumerate(zip(dosage, means, alphas, priors)):
            if d > 0:
                fg_means[h].append(m / d)
                fg_alphas[h].append(a)
                fg_priors[h].append(p)
            else:
                bg_means[h].append(m)
                bg_alphas[h].append(a)
                bg_priors[h].append(p)

    fg_mean = np.array([
        np.mean(values) if values else np.nan
        for values in fg_means
    ])

    bg_mean = np.array([
        np.mean(values) if values else np.nan
        for values in bg_means
    ])

    fg_mean = _shrink_to_shared(fg_mean, channel_strength, fallback=1.0)
    bg_mean = _shrink_to_shared(bg_mean, channel_strength, fallback=0.1)

    if dist_type == "nb":
        fg_alpha = np.array([
            np.nanmean(values) if values else np.nan
            for values in fg_alphas
        ])

        bg_alpha = np.array([
            np.nanmean(values) if values else np.nan
            for values in bg_alphas
        ])

        fg_alpha = _shrink_to_shared(fg_alpha, channel_strength, fallback=0.1)
        bg_alpha = _shrink_to_shared(bg_alpha, channel_strength, fallback=0.1)
    else:
        fg_alpha, bg_alpha = None, None

    fg_prior = np.array([
        np.mean(values) if values else np.nan
        for values in fg_priors
    ])

    bg_prior = np.array([
        np.mean(values) if values else np.nan
        for values in bg_priors
    ])

    fg_prior = _shrink_to_shared(fg_prior, channel_strength, fallback=0.01)
    bg_prior = _shrink_to_shared(bg_prior, channel_strength, fallback=0.01)

    fg_prior = np.clip(fg_prior, 1e-6, 1.0 - 1e-6)
    bg_prior = np.clip(bg_prior, 1e-6, 1.0 - 1e-6)

    if update:
        means, alphas, priors = _calculate_model_params(
            component_dosages,
            fg_mean,
            bg_mean,
            fg_alpha,
            bg_alpha,
            fg_prior,
            bg_prior,
        )

        if alphas is None:
            alphas = [None] * len(model_dists)

        if priors is None:
            priors = [None] * len(model_dists)

        for dist_mean, dist_alpha, dist_prior, dist in zip(
            means,
            alphas,
            priors,
            model_dists,
        ):
            _update_parameter(
                getattr(dist.distribution, mean_attr),
                dist_mean,
            )

            if dist_type == "nb":
                _update_parameter(
                    getattr(dist.distribution, "alphas"),
                    dist_alpha,
                )

            if dist_prior is not None:
                _update_parameter(
                    getattr(dist, "priors"),
                    dist_prior,
                )

    return fg_mean, bg_mean, fg_alpha, bg_alpha, fg_prior, bg_prior


@torch.no_grad()
def _fit_constrained_model(
    model,
    X,
    component_dosages,
    priors=None,
    dist_type="poisson",
    max_iter=1000,
    tol=0.1,
    channel_strength=1.0,
):
    X = utils.numpy_to_torch(X)
    logp = None

    for i in range(max_iter):

        last_logp = logp
        logp = model.summarize(X, priors=priors)
        if i > 0:
            improvement = logp - last_logp
            if improvement < tol:
                break

        model.from_summaries()

        _constrain_params(
            model,
            component_dosages,
            dist_type,
            update=True,
            channel_strength=channel_strength,
        )

    model._reset_cache()

    return _constrain_params(
        model,
        component_dosages,
        dist_type,
        update=False,
        channel_strength=channel_strength,
    )


def _expected_component_dosages(X, experiment_params):
    DOSAGES = {
        'recombinant': np.array([
            [1, 0],
            [0, 1]
        ]),
        'f1': np.array([
            [1, 0],
            [0, 1]
        ]),
        'f2': np.array([
            [2.0, 0.0],
            [1.0, 1.0],
            [0.0, 2.0],
        ]),
        'backcross': np.array([
            [2.0, 0.0],
            [1.0, 1.0],
        ]),
        'testcross': np.array([
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ]),
        'three_way': np.array([
            [2.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]),
        'four_way': np.array([
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ])
    }

    if experiment_params.genotyping_strategy == "recombinant":
        component_dosages = DOSAGES['recombinant']
        _require_n_haplotypes(X, component_dosages.shape[1], 'recombinant')
    else:
        component_dosages = DOSAGES[experiment_params.crossing_strategy]
        _require_n_haplotypes(X, component_dosages.shape[1], experiment_params.crossing_strategy)
    return component_dosages


def estimate_emissions(
    X,
    experiment_params,
    window=40,
    dist_type="poisson",
    prior_temperature=0.05,
    prior_floor=0.02,
    channel_strength=0.75,
    max_iter=1000,
    tol=0.1,
):
    component_dosages = _expected_component_dosages(X, experiment_params)
    X_flattened = utils.concat_arrays(X)

    label_priors = _haplotype_pattern_priors(
        X,
        component_dosages,
        window=window,
        temperature=prior_temperature,
        floor=prior_floor,
    )

    init_fg_mean, init_bg_mean, init_fg_alpha, init_bg_alpha = (
        _estimate_init_params_soft(
            X_flattened,
            label_priors,
            component_dosages,
            channel_strength=channel_strength,
        )
    )

    gmm = _build_gmm(
        component_dosages,
        fg_mean=init_fg_mean,
        bg_mean=init_bg_mean,
        label_priors=label_priors.mean(axis=0),
        fg_alpha=init_fg_alpha,
        bg_alpha=init_bg_alpha,
        dist_type=dist_type,
    )

    params = _fit_constrained_model(
        gmm,
        X_flattened,
        component_dosages,
        priors=label_priors,
        dist_type=dist_type,
        max_iter=max_iter,
        tol=tol,
        channel_strength=channel_strength,
    )

    return _format_params(params, dist_type)
