import os
import logging
from copy import deepcopy
from collections import Counter
from dataclasses import dataclass
import numpy as np

import torch
from pomegranate import distributions as pmd
from pomegranate.hmm import DenseHMM

from coelsch.defaults import DEFAULT_RANDOM_SEED
from .dists import NegativeBinomial, ZeroInflated
from .utils import sorted_edit_distance, mask_array_zeros, numpy_to_torch


log = logging.getLogger('coelsch')
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)
DEFAULT_DEVICE = torch.device('cpu')
TINY_PROB = 1e-12


@dataclass
class Transition:
    self_loop: float = 0.0
    rigid: float = 0.0
    co: float = TINY_PROB # use a tiny default for COs to prevent -infs when double CO dists << rfactor occur
    start: float = 0.0
    end: float = 0.0


def trans_prob_decay(rfactor, trans_prob, decay_rate=0.25,  floor=TINY_PROB):
    '''
    calculates the decay rate of the crossover probability through the rigid chain.
    Set to zero to make the chain entirely rigid (no crossovers allowed before end of segment)
    '''
    scale = np.exp(-(np.arange(rfactor)[::-1]) / decay_rate)
    scale[-1] = 1.0 # fix prob at the end of the sequence
    return np.maximum(scale * trans_prob, floor)


def term_prob_decay(rfactor, term_rfactor, nstates, decay_rate=0.25, floor=TINY_PROB):
    '''
    calculates the decay rate of the states to be at the end of the sequence.
    Set to zero to make the terminal segments entirely rigid (no crossovers allowed within 
    term_rfactor bins of the end of the chromosome).
    '''
    scale = np.ones(rfactor)
    scale[:term_rfactor] = floor
    if decay_rate:
        scale[term_rfactor:] = 1 - np.exp(-(np.arange(rfactor - term_rfactor) + 1) / decay_rate)
    scale = np.maximum(scale, floor)
    return scale / scale.sum() / nstates


class RigidHMM:
    """
    Implements a rigid Hidden Markov Model (rHMM) for modeling haplotype transitions
    with fixed structure and constrained transition probabilities.

    Parameters
    ----------
    states : list of tuple, list or set
        The haplotype states to produce a model for
    rfactor : int
        Number of bins in a genomic segment - constrains the minimum distance between transitions.
    term_rfactor : int
        Number of bins in a terminal genomic segment - constrains the minimum distance between 
        end of sequence and transitions.
    trans_prob : float
        Probability of transition between rigid states.
    dist_type: str
        Sets the underlying distributions used in the model. Can be either "poisson" or "nb"
    fg_params : dict
        Foreground model params (dict with value "lambda" if dist_type == "poisson"
        or values ["mean", "alpha"] if dist_type == "nb")
    bg_params : float
        Background model params (dict with value "lambda" if dist_type == "poisson"
        or values ["mean", "alpha"] if dist_type == "nb")
    empty_fraction : float
        Fraction of empty bins in zero inflated model.
    trans_prob_decay_rate : float
        Decay rate of trans_prob allowing escape from the rigid state sequence,
        can rescue occasional double crossovers but also increases sensitivity to noise.
    device : torch.device, optional
        Device on which to allocate model (default: CPU).
    """

    def __init__(self, states, rfactor, term_rfactor, trans_prob,
                 fg_params, bg_params, dist_type='poisson', trans_prob_decay_rate=0.25,
                 device=DEFAULT_DEVICE):
        for hap_comb in states:
            if not isinstance(hap_comb, (tuple, list)):
                raise ValueError(
                    'states should be a list/tuple of tuples, which represent haplotype combinations'
                )
            for hap in hap_comb:
                if hap not in (0, 1):
                    raise ValueError('haplotypes can only be 0 or 1')
        self.states = tuple(tuple(s) for s in states)
        self.nstates = len(self.states)
        self._state_haplo = np.mean(self.states, axis=1)
        self.rfactor = int(rfactor)
        self.term_rfactor = int(term_rfactor)
        self.trans_prob = float(trans_prob)
        self.trans_prob_decay_rate = float(trans_prob_decay_rate)
        assert 0.0 < self.trans_prob < 1.0
        self.dist_type = dist_type.lower()
        self.fg_params = fg_params
        self.bg_params = bg_params
        if self.dist_type == "poisson":
            for d in (self.fg_params, self.bg_params):
                if "lambda" not in d:
                    raise ValueError("Poisson requires {'lambda'} in fg_params/bg_params")
        elif self.dist_type == "nb":
            for d in (self.fg_params, self.bg_params):
                if not all(k in d for k in ("mean", "alpha")):
                    raise ValueError("Negative Binomial requires {'mean','alpha'} in fg_params/bg_params")
        else:
            raise ValueError('Unrecognised model type')
        self._device = device
        self.initialise_model()

    def _calculate_transition_probs(self):
        trans_probs = trans_prob_decay(self.rfactor, self.trans_prob, self.trans_prob_decay_rate)
        end_probs = term_prob_decay(
            self.rfactor, self.term_rfactor, self.nstates, self.trans_prob_decay_rate
        )
        start_probs = end_probs[::-1]
        for i in range(self.rfactor):
            # first state of chain
            remaining_prob = 1.0 - trans_probs[i] - end_probs[i]
            assert remaining_prob > 0
            if i == 0:
                self._transition_probs[i] = Transition(
                    self_loop=remaining_prob / 2,
                    rigid=remaining_prob / 2,
                    co=trans_probs[i],
                    start=start_probs[i],
                    end=end_probs[i]
                )
            # last state of chain
            elif (i + 1) == self.rfactor:
                # no rigid transition for last of chain
                self._transition_probs[i] = Transition(
                    self_loop=remaining_prob,
                    co=trans_probs[i],
                    start=start_probs[i],
                    end=end_probs[i]
                )
            # internal point of the chain
            else:
                self._transition_probs[i] = Transition(
                    self_loop=0.0,
                    rigid=remaining_prob,
                    start=start_probs[i],
                    co=trans_probs[i],
                    end=end_probs[i],
                )

    def _create_distribution(self, state):
        priors = [self.bg_params['empty_fraction'], self.bg_params['empty_fraction']]
        if self.dist_type == 'poisson':
            lambdas = [self.bg_params['lambda'], self.bg_params['lambda']]
        else:
            means = [self.bg_params['mean'], self.bg_params['mean']]
            alphas = [self.bg_params['alpha'], self.bg_params['alpha']]
        for hap, count in Counter(state).items():
            priors[hap] = self.fg_params['empty_fraction']
            if self.dist_type == "poisson":
                lambdas[hap] = self.fg_params["lambda"] * count
            else:
                means[hap] = self.fg_params['mean'] * count
                alphas[hap] = self.fg_params['alpha']
        if self.dist_type == 'poisson':
            return ZeroInflated(pmd.Poisson(lambdas), priors=priors)
        else:
            return ZeroInflated(NegativeBinomial(means, alphas), priors=priors)

    def _create_rigid_chain(self, state):
        dist = self._create_distribution(state)
        self._distributions[state] = []
        for _ in range(self.rfactor):
            dist = deepcopy(dist)
            self._model.add_distribution(dist)
            self._distributions[state].append(dist)

    def _add_transitions(self):

        # manually set all edges to -inf first, then update
        # this prevents a bug where edges in large models are occasionally
        # initialised with NaNs by pomegranate
        # see: https://github.com/jmschrei/pomegranate/issues/1078

        n = self._model.n_distributions
        self._model.starts = torch.full(
            (n,), -np.inf, dtype=self._model.dtype, device=self._device
        )
        self._model.ends = torch.full(
            (n,), -np.inf, dtype=self._model.dtype, device=self._device
        )
        self._model.edges = torch.full(
            (n, n), -np.inf, dtype=self._model.dtype, device=self._device
        )

        for state in self.states:
            for i in range(self.rfactor):
                if self._transition_probs[i].self_loop:
                    self._model.add_edge(
                        self._distributions[state][i],
                        self._distributions[state][i],
                        self._transition_probs[i].self_loop
                    )
                if self._transition_probs[i].rigid:
                    self._model.add_edge(
                        self._distributions[state][i],
                        self._distributions[state][i + 1],
                        self._transition_probs[i].rigid
                    )
                if self._transition_probs[i].co:
                    for other in self.states:
                        # only connect states with edit dist 1 with crossovers
                        if sorted_edit_distance(state, other) == 1:
                            self._model.add_edge(
                                self._distributions[state][i],
                                self._distributions[other][0],
                                self._transition_probs[i].co
                            )
                if self._transition_probs[i].start:
                    self._model.add_edge(
                        self._model.start,
                        self._distributions[state][i],
                        self._transition_probs[i].start
                    )
                if self._transition_probs[i].end:
                    self._model.add_edge(
                        self._distributions[state][i],
                        self._model.end,
                        self._transition_probs[i].end
                    )

    def initialise_model(self):
        """
        Initializes the DenseHMM model using estimated parameters.
        """
        self._transition_probs = {}
        self._calculate_transition_probs()
        self._distributions = {}
        self._chains = {}
        self._model = DenseHMM(frozen=True)
        for state in self.states:
            self._create_rigid_chain(state)
        self._add_transitions()
        log.debug(f'moving model to device: {self._device}')
        self._model.to(self._device)
        log.debug(
            f'Finished initialising model with {self._model.n_distributions} distributions'
        )

    torch.no_grad()
    def predict_state_proba(self, X, batch_size=128):
        """
        Predicts state probabilities for input marker arrays. Each state can represent a single
        haplotype (for haploid data) or a mixture of two or more haplotypes (for diploid+ data)

        Parameters
        ----------
        X : list of np.ndarray or np.ndarray
            3D array of shape (N, L, 2) containing haplotype-specific read/variant counts per barcode.
        batch_size : int, optional
            Batch size for model prediction (default: 128).

        Returns
        -------
        np.ndarray
            3D array of predicted probabilities for each state, with shape (N, L, n_states).
        """
        proba = []
        for X_batch in np.array_split(X, int(np.ceil(len(X) / batch_size))):
            batch_size, chrom_size = X_batch.shape[:2]
            X_batch = numpy_to_torch(X_batch)
            if self._device is not None:
                X_batch = X_batch.to(self._device)
            p_batch = self._model.predict_proba(X_batch).cpu().numpy()
            if np.isnan(p_batch).any():
                log.warn(
                    'At least one sample is impossible under the rHMM. '
                    'This is usually caused either by doublets, or low CO interference that is '
                    'much shorter than rfactor. '
                    'Try reducing --segment-size or increasing --interference-half-life'
                )
            # sum over all chain distributions representing the same state
            p_batch = p_batch.reshape(batch_size, chrom_size, self.nstates, self.rfactor).sum(axis=3)
            proba.append(p_batch)
        proba = np.concatenate(proba, axis=0)
        # convert to single value per bin, representing the probability of alt hap

        return proba

    def predict_haplo_proba(self, X, batch_size=128):
        """
        Predicts haplotype probabilities for input marker arrays.

        Parameters
        ----------
        X : list of np.ndarray or np.ndarray
            3D array of shape (N, L, 2) containing haplotype-specific read/variant counts per barcode.
        batch_size : int, optional
            Batch size for model prediction (default: 128).

        Returns
        -------
        np.ndarray
            2D array of predicted probabilities of alternative haplotype (hap 1), with shape (N, L).
        """
        return np.clip(self.predict_state_proba(X, batch_size) @ self._state_haplo, 0, 1)

    def predict(self, X, batch_size=128):
        """
        Predicts haplotype probabilities for input marker arrays.
        Alias to RigidHMM.predict_haplo_proba.

        Parameters
        ----------
        X : list of np.ndarray or np.ndarray
            3D array of shape (N, L, 2) containing haplotype-specific read/variant counts per barcode.
        batch_size : int, optional
            Batch size for model prediction (default: 128).

        Returns
        -------
        np.ndarray
            2D array of predicted probabilities of alternative haplotype (hap 1), with shape (N, L).
        """
        return self.predict_haplo_proba(X, batch_size)

    @torch.no_grad()
    def log_probability(self, X, batch_size=128):
        logp = []
        for X_batch in np.array_split(X, int(np.ceil(len(X) / batch_size))):
            batch_size, chrom_size = X_batch.shape[:2]
            X_batch = numpy_to_torch(X_batch)
            if self._device is not None:
                X_batch = X_batch.to(self._device)
            lp_batch = self._model.log_probability(X_batch).cpu().numpy()
            if np.isnan(lp_batch).any():
                log.warn(
                    'At least one sample is impossible under the rHMM. '
                    'This is usually caused either by doublets, or low CO interference that is '
                    'much shorter than rfactor. '
                    'Try reducing --segment-size or increasing --interference-half-life'
                )
            logp.append(lp_batch)
        return np.concatenate(logp)

    @torch.no_grad()
    def sample(self, X, n=1, batch_size=128, rng=DEFAULT_RNG):
        """
        Posterior sampling of haplotype paths given markers

        Parameters
        ----------
        X : np.ndarray, shape (n_seq, n_bins, 2)
            Marker emission arrays.
        n : int, default 1
            Number of posterior samples per sequence.
        batch_size : int, optional
            Batch size for model prediction (default: 128).
        rng : np.random.Generator, optional
            Random number generator instance.

        Returns
        -------
        np.ndarray, shape (n_seq, n_bins) if n==1 else (n_seq, n, n_bins)
            Sampled haplotype identities (0/1) for each sequence × replicate × bin.
        """

        seed = rng.integers(0, 2**63 - 1, dtype=np.int64).item()
        rng = torch.Generator(device=self._device)
        rng.manual_seed(seed)

        model = self._model
        n_seq, n_bins, n_haps = X.shape
        assert n_haps == 2
        n_states = model.n_distributions
        rfactor = self.rfactor
        log_A = model.edges.to(self._device)

        def _unmask(x):
            if isinstance(x, torch.masked.MaskedTensor):
                return x._masked_data
            return x

        X_samples = np.empty((n_seq, n, n_bins), dtype=np.int16)
        offset = 0

        for X_batch in np.array_split(X, int(np.ceil(len(X) / batch_size))):
            X_batch = numpy_to_torch(X_batch)
            X_batch = X_batch.to(self._device, dtype=torch.float32)
            n_batch = X_batch.shape[0]

            log_alpha = _unmask(model.forward(X_batch))
            log_beta  = _unmask(model.backward(X_batch))
            log_B = torch.stack(
                [_unmask(d.log_probability(X_batch.reshape(-1, n_haps))).reshape(n_batch, n_bins)
                 for d in model.distributions],
                dim=2
            )

            z = torch.empty((n_batch, n, n_bins), dtype=torch.long, device=self._device)
            log_post_T = log_alpha[:, -1, :] + log_beta[:, -1, :]
            pT = torch.log_softmax(log_post_T, dim=1)
            pT_expanded = pT.unsqueeze(1).expand(-1, n, -1)
            flat_probs = pT_expanded.reshape(-1, n_states)
            samples = torch.multinomial(torch.exp(flat_probs.double()), 1, generator=rng).squeeze(1)
            z[:, :, -1] = samples.view(n_batch, n)

            for t in reversed(range(n_bins - 1)):
                log_alpha_t = log_alpha[:, t, :]
                log_beta_t1 = log_beta[:, t + 1, :]
                log_B_t1    = log_B[:, t + 1, :]

                for k in range(n):
                    j = z[:, k, t + 1]
                    A_cols = log_A[:, j]
                    log_p = (
                        log_alpha_t
                        + A_cols.T
                        + log_B_t1[torch.arange(n_batch), j][:, None]
                        + log_beta_t1[torch.arange(n_batch), j][:, None]
                    )
                    dead = ~torch.isfinite(log_p).any(dim=1)
                    if dead.any():
                        # very occasionally sequences are impossible due to the rigidity of the model
                        # when this happens we move the sampler to the end of the current rigid sequence and
                        # then allow it to continue.
                        skip_idx = ((j // self.rfactor) + 1) * self.rfactor - 1
                        log_p_skip = torch.full_like(log_p, float('-inf'), device=log_p.device)
                        log_p_skip[torch.arange(n_batch, device=log_p.device), skip_idx] = 0.0
                        log_p[dead] = log_p_skip[dead]  # jumps the model to the end of the rigid sequence
                        log.warn(
                            'At least one sample is impossible under the rHMM. '
                            'This is usually caused either by doublets, or low CO interference that is '
                            'much shorter than rfactor. '
                            'Try reducing --segment-size or increasing --interference-half-life'
                        )
                    log_p = torch.log_softmax(log_p, dim=1)
                    z[:, k, t] = torch.multinomial(torch.exp(log_p.double()), 1, generator=rng).squeeze(1)
            z = (z // rfactor).to(torch.int16).cpu().numpy()
            X_samples[offset:offset + n_batch] = z
            offset += n_batch

        return X_samples.squeeze()

    @property
    def params(self):
        return {
            'states': [list(s) for s in self.states],
            'rfactor': float(self.rfactor),
            'term_rfactor': float(self.term_rfactor),
            'trans_prob': float(self.trans_prob),
            'trans_prob_decay_rate': float(self.trans_prob_decay_rate),
            'is_poisson': 1.0 if self.dist_type == 'poisson' else 0.0,
            'fg_lambda': self.fg_params['lambda'] if self.dist_type == 'poisson' else np.nan,
            'bg_lambda': self.bg_params['lambda'] if self.dist_type == 'poisson' else np.nan,
            'fg_mean': self.fg_params['mean'] if self.dist_type == 'nb' else np.nan,
            'bg_mean': self.bg_params['mean'] if self.dist_type == 'nb' else np.nan,
            'fg_alpha': self.fg_params['alpha'] if self.dist_type == 'nb' else np.nan,
            'bg_alpha': self.bg_params['alpha'] if self.dist_type == 'nb' else np.nan,
            'fg_empty_fraction': self.fg_params['empty_fraction'],
            'bg_empty_fraction': self.bg_params['empty_fraction']
        }

    @classmethod
    def from_params(cls, params, device=DEFAULT_DEVICE):
        if params['is_poisson']:
            fg_params = {
                'lambda': params['fg_lambda'],
                'empty_fraction': params['fg_empty_fraction']
            }
            bg_params = {
                'lambda': params['bg_lambda'],
                'empty_fraction': params['bg_empty_fraction']
            }
        else:
            fg_params = {
                'mean': params['fg_mean'],
                'alpha': params['fg_alpha'],
                'empty_fraction': params['fg_empty_fraction']
            }
            bg_params = {
                'mean': params['bg_mean'],
                'alpha': params['bg_alpha'],
                'empty_fraction': params['bg_empty_fraction']
            }
        return cls(
            params['states'], params['rfactor'], params['term_rfactor'],
            params['trans_prob'], fg_params, bg_params,
            dist_type='poisson' if params['is_poisson'] else 'nb',
            trans_prob_decay_rate=params['trans_prob_decay_rate'], device=device
        )