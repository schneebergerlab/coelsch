import numpy as np
import torch
from pomegranate._utils import (
    _cast_as_tensor, _cast_as_parameter, _update_parameter,
    _check_parameter, _reshape_weights
)
from pomegranate.distributions._distribution import Distribution
from pomegranate import distributions as pmd


def _unwrap_mt(x):
    if isinstance(x, torch.masked.MaskedTensor):
        return x._masked_data, x._masked_mask
    return x, None


def _rewrap_mt(data, mask):
    if mask is None:
        return data
    data = data.clone()
    data[~mask] = 1e-48
    return torch.masked.MaskedTensor(data, mask)


def _mask_union(*masks):
    mask = None
    for m in masks:
        if m is not None:
            mask = m if mask is None else (mask | m)
    return mask


def mt_where(cond, a, b):
    cond_d, cond_m = _unwrap_mt(cond)
    a_d, a_m = _unwrap_mt(a)
    b_d, b_m = _unwrap_mt(b)
    out_d = torch.where(cond_d, a_d, b_d)
    out_m = _mask_union(cond_m, a_m, b_m)
    return _rewrap_mt(out_d, out_m)


def mt_clamp_min(x, min_val):
    x_d, x_m = _unwrap_mt(x)
    m_d, m_m = _unwrap_mt(min_val)
    m_d = torch.as_tensor(m_d, device=x_d.device, dtype=x_d.dtype)
    out_d = torch.maximum(x_d, m_d)
    out_m = _mask_union(x_m, m_m)
    return _rewrap_mt(out_d, out_m)


def mt_stack(tensors, dim=0):
    datas = []
    masks = []
    masked = 0

    for t in tensors:
        if isinstance(t, torch.masked.MaskedTensor):
            datas.append(t._masked_data)
            masks.append(t._masked_mask)
            masked += 1
        else:
            datas.append(t)

    data_stacked = torch.stack(datas, dim=dim)

    if not masked:
        return data_stacked

    if masked != len(tensors):
        raise ValueError('Either all or none must be masked')

    mask_stacked = torch.stack(masks, dim=dim)
    return torch.masked.MaskedTensor(data_stacked, mask_stacked)


class NegativeBinomial(Distribution):
    """
    A DESeq2-style Negative Binomial distribution parameterized by mean and dispersion.
    Variance is given by:
        Var(X) = mean + alpha * mean^2
    Parameters
    ----------
    means : list, numpy.ndarray, torch.Tensor or None, shape=(d,)
        Mean count for each feature.
    alphas : list, numpy.ndarray, torch.Tensor or None, shape=(d,)
        Dispersion factor for each feature.
    inertia : float
        Interpolation factor for updates (0 = replace, 1 = ignore update).
    frozen : bool
        If True, disables training.
    check_data : bool
        If True, validates input types and values.
    """

    def __init__(self, means=None, alphas=None, inertia=0.0, frozen=False, check_data=True):
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
        self.name = "NegativeBinomial"

        self.means = _check_parameter(_cast_as_parameter(means), "means", min_value=0, ndim=1)
        self.alphas = _check_parameter(_cast_as_parameter(alphas), "alphas", min_value=0, ndim=1)

        self._initialized = self.means is not None and self.alphas is not None
        self.d = self.means.shape[-1] if self._initialized else None
        self._reset_cache()

    def _initialize(self, d):
        self.means = _cast_as_parameter(torch.ones(d, dtype=self.dtype, device=self.device))
        self.alphas = _cast_as_parameter(torch.ones(d, dtype=self.dtype, device=self.device) * 0.1)
        self._initialized = True
        super()._initialize(d)

    def _reset_cache(self):
        if not self._initialized:
            return

        self.register_buffer('_w_sum', torch.zeros(self.d, device=self.device))
        self.register_buffer('_xw_sum', torch.zeros(self.d, device=self.device))
        self.register_buffer('_x2w_sum', torch.zeros(self.d, device=self.device))

    def sample(self, n):
        """
        Sample n values using mean and dispersion.
        Internally maps to (r, p) form:
            r = 1 / alpha
            p = r / (r + mean)
        """
        r = 1.0 / self.alphas
        p = r / (r + self.means)
        dist = torch.distributions.NegativeBinomial(total_count=r, probs=(1 - p))
        return dist.sample((n,))

    def log_probability(self, X):
        """
        Compute log-probabilities for each example under the NB distribution.
        """
        X = _check_parameter(
            _cast_as_tensor(X), "X", min_value=0.0, ndim=2, shape=(-1, self.d),
            check_parameter=self.check_data
        )

        r = 1.0 / self.alphas
        p = r / (r + self.means)

        return torch.sum(
            torch.lgamma(X + r) - torch.lgamma(r) - torch.lgamma(X + 1) +
            r * torch.log(p) + X * torch.log1p(-p),
            dim=-1
        )

    def summarize(self, X, sample_weight=None):
        """
        Accumulate sufficient statistics (x, x^2, weights).
        """
        if self.frozen:
            return

        X, sample_weight = super().summarize(X, sample_weight=sample_weight)
        _check_parameter(X, "X", min_value=0, check_parameter=self.check_data)

        self._w_sum = self._w_sum + torch.sum(sample_weight, dim=0)
        self._xw_sum = self._xw_sum + torch.sum(X * sample_weight, dim=0)
        self._x2w_sum = self._x2w_sum + torch.sum((X ** 2) * sample_weight, dim=0)

    def from_summaries(self):
        """
        Update mean and alpha from accumulated statistics using method-of-moments
        """
        if self.frozen:
            return

        mean = self._xw_sum / self._w_sum
        second_moment = self._x2w_sum / self._w_sum
        var = second_moment - mean ** 2

        # Ensure variance is always > mean for NB validity
        var = mt_clamp_min(var, min_val=mean + 1e-6)
        alpha = (var - mean) / (mean ** 2)
        alpha = mt_clamp_min(alpha, min_val=1e-8)

        _update_parameter(self.means, mean, self.inertia)
        _update_parameter(self.alphas, alpha, self.inertia)

        self._reset_cache()


class ZeroInflated(Distribution):
    """A zero-inflated wrapper for generic distributions, supporting independent zero inflation per feature."""

    def __init__(self, distribution, priors=None, inertia=0.0, frozen=False, check_data=True, eps=1e-8):
        if distribution.name not in ('Poisson', 'NegativeBinomial'):
            raise ValueError('ZeroInflated currently only implemented for Poisson/NB')
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
        self.name = "ZeroInflated({})".format(distribution.name)
        self.distribution = distribution

        d = getattr(distribution, "d", None)
        if d is None:
            raise ValueError("Base distribution must have a 'd' attribute for feature dimension.")

        if priors is None:
            priors = torch.full((d,), 0.5, dtype=self.dtype, device=self.device)
        priors = _check_parameter(_cast_as_parameter(priors), "priors", min_value=0.0, max_value=1.0, ndim=1, shape=(d,))

        self.log_priors = _cast_as_parameter(torch.log(priors))
        self.log_1mpriors = _cast_as_parameter(torch.log(1.0 - priors))
        self.log_eps = _cast_as_parameter(torch.log(torch.tensor(eps, dtype=torch.float64)))
        self.log_1meps = _cast_as_parameter(torch.log1p(-torch.tensor(eps, dtype=torch.float64)))

        self.d = d
        self._initialized = True
        self._reset_cache()

    def _initialize(self, d):
        # Uniform prior for initialization
        log_p = torch.log(torch.full((d,), 0.5, dtype=self.dtype, device=self.device))
        log_1mp = torch.log(1. - torch.full((d,), 0.5, dtype=self.dtype, device=self.device))
        self.log_priors = _cast_as_parameter(log_p)
        self.log_1mpriors = _cast_as_parameter(log_1mp)
        self.d = d
        self._initialized = True
        if hasattr(self.distribution, "_initialize"):
            self.distribution._initialize(d)
        self._reset_cache()

    def _reset_cache(self):
        self.register_buffer("_priors_sum", torch.zeros(self.d, device=self.device))
        self.register_buffer("_zero_sum", torch.zeros(self.d, device=self.device))
        if hasattr(self.distribution, "_reset_cache"):
            self.distribution._reset_cache()

    def log_probability(self, X):
        X = _check_parameter(_cast_as_tensor(X), "X", min_value=0, ndim=2, shape=(-1, self.d), check_parameter=self.check_data)

        log_probs = []
        for j in range(self.d):
            xj = X[:, j]
            log_prior_j = self.log_priors[j]
            log_1mp = self.log_1mpriors[j]
            log_pj = self._base_logprob_feature(X, j)
            log_prob = mt_where(
                xj == 0,
                torch.logaddexp(log_prior_j, log_1mp + log_pj),
                log_1mp + log_pj
            )
            log_probs.append(log_prob)
        logp = mt_stack(log_probs, dim=1).sum(dim=1)
        return torch.logaddexp(logp + self.log_1meps, self.log_eps)

    def _base_logprob_feature(self, X, j):
        xj = X[:, j]
        if self.distribution.name == "Poisson":
            lambdaj = self.distribution.lambdas[j]
            return xj * torch.log(lambdaj) - lambdaj - torch.lgamma(xj + 1)
        elif self.distribution.name == "NegativeBinomial":
            meanj = self.distribution.means[j]
            alphaj = self.distribution.alphas[j]
            pj = alphaj / (alphaj + meanj)
            rj = alphaj
            return (
                torch.lgamma(xj + rj)
                - torch.lgamma(rj)
                - torch.lgamma(xj + 1)
                + rj * torch.log(pj)
                + xj * torch.log(1 - pj)
            )
        raise NotImplementedError(
            f"Per-feature log-probability for zero-inflated {self.distribution.name} is not implemented."
        )

    def sample(self, n):
        X_base = self.distribution.sample(n)
        # Convert log-priors to probability for sampling
        prior_probs = torch.exp(self.log_priors)
        mask = torch.rand((n, self.d), device=X_base.device) < prior_probs
        X_base_masked = X_base.clone()
        X_base_masked[mask] = 0
        return X_base_masked

    def summarize(self, X, sample_weight=None):
        if self.frozen:
            return
        X, sample_weight = super().summarize(X, sample_weight=sample_weight)
        X = _check_parameter(X, "X", min_value=0, check_parameter=self.check_data)
        gamma = torch.zeros_like(_unwrap_mt(X)[0], dtype=self.dtype, device=self.device)

        for j in range(self.d):
            xj = X[:, j]
            log_prior_j = self.log_priors[j]
            log_1mp = self.log_1mpriors[j]
            # Probability of zero for underlying
            if self.distribution.name == "Poisson":
                lambdaj = self.distribution.lambdas[j]
                log_p0 = -lambdaj
            elif self.distribution.name == "NegativeBinomial":
                meanj = self.distribution.means[j]
                alphaj = self.distribution.alphas[j]
                pj = alphaj / (alphaj + meanj)
                rj = alphaj
                log_p0 = rj * torch.log(pj)
            else:
                raise NotImplementedError(
                    f"Per-feature summarization for zero-inflated {self.distribution.name} is not implemented. "
                )
            log_numer = log_prior_j
            log_denom = torch.logaddexp(log_prior_j, log_1mp + log_p0)
            gamma[:, j] = mt_where(xj == 0, torch.exp(log_numer - log_denom), torch.zeros_like(_unwrap_mt(xj)[0], dtype=self.dtype))
        if sample_weight is not None:
            w = _cast_as_tensor(sample_weight)
        else:
            w = torch.ones_like(_unwrap_mt(X)[0], dtype=self.dtype, device=self.device)
        self._priors_sum[:] = self._priors_sum + torch.sum(gamma * w, dim=0)
        self._zero_sum[:] = self._zero_sum + torch.sum(w, dim=0)
        w_base = (1 - gamma) * w
        if hasattr(self.distribution, "summarize"):
            self.distribution.summarize(X, sample_weight=w_base)

    def from_summaries(self):
        if self.frozen:
            return
        priors_new = self._priors_sum / mt_clamp_min(self._zero_sum, min_val=1e-6)
        log_priors_new = torch.log(priors_new + 1e-32)
        log_1mpriors_new = torch.log(1. - priors_new + 1e-32)
        _update_parameter(self.log_priors, log_priors_new, self.inertia)
        _update_parameter(self.log_1mpriors, log_1mpriors_new, self.inertia)
        if hasattr(self.distribution, "from_summaries"):
            self.distribution.from_summaries()
        self._reset_cache()

    @property
    def priors(self):
        return torch.exp(self.log_priors)