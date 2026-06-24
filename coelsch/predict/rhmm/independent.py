"""
Independent-meiosis wrapper models for rigid HMM prediction.
"""
import logging
from functools import reduce
import numpy as np
import torch

from coelsch.defaults import DEFAULT_RANDOM_SEED
from .model import RigidHMM

log = logging.getLogger('coelsch')
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)
DEFAULT_DEVICE = torch.device('cpu')


def _extract_haplotype_params(params, haplotypes):
    return {
        param_name: param_values[list(haplotypes)]
        for param_name, param_values in params.items()
    }


class IndependentMeiosesHMM:
    """
    Run two haploid RigidHMM independently over two meioses.

    This wrapper targets three and four-way crosses, where marker channels
    (0, 1) and (2, 3) or (0, 1) and (0, 2) correspond to independent haploid meioses.
    It exposes the public methods used by crossover prediction while returning
    full marginal haplotype outputs.
    """

    def __init__(self, crossing_strategy,
                 rfactor, term_rfactor, trans_prob,
                 fg_params, bg_params, dist_type='poisson',
                 trans_prob_decay_rate=0.25, device=DEFAULT_DEVICE):

        if crossing_strategy == 'four_way':
            self.meioses = ((0, 1), (2, 3))
            rhmm_states = ((0,), (1,))
            uneven_ploidy = False
            corrected_dosage = None
        elif crossing_strategy == 'three_way':
            self.meioses = ((0, 1), (0, 2))
            rhmm_states = ((0, 0,), (0,), (0, 1), (1,))
            uneven_ploidy = True
            corrected_dosage = np.array([
                [1, 0], # true diploid dosage: [2, 0, 0]
                [1, 0], # true diploid dosage: [1, 0, 1]
                [0, 1], # true diploid dosage: [1, 1, 0]
                [0, 1], # true diploid dosage  [0, 1, 1]
            ])
        self.crossing_strategy = crossing_strategy
        self.rhmms = []
        for i, meiosis_haplotypes in enumerate(self.meioses):
            meiosis_haplotypes = tuple(int(hap) for hap in meiosis_haplotypes)
            rhmm = RigidHMM(
                states=rhmm_states,
                n_haplotypes=2,
                rfactor=rfactor,
                term_rfactor=term_rfactor,
                trans_prob=trans_prob,
                fg_params=_extract_haplotype_params(fg_params, meiosis_haplotypes),
                bg_params=_extract_haplotype_params(bg_params, meiosis_haplotypes),
                dist_type=dist_type,
                trans_prob_decay_rate=trans_prob_decay_rate,
                allow_uneven_ploidy=uneven_ploidy,
                device=device
            )
            if corrected_dosage is not None:
                rhmm.set_state_haplotype_dosage(corrected_dosage)
            self.rhmms.append(rhmm)

        # wrapper states are the product of meioses
        self.states = tuple(
            (left, right)
            for left in self.meioses[0]
            for right in self.meioses[1]
        )

        # for persistence when converting to params dict
        self.rfactor = int(rfactor)
        self.term_rfactor = int(term_rfactor)
        self.trans_prob = float(trans_prob)
        self.trans_prob_decay_rate = float(trans_prob_decay_rate)
        self.dist_type = dist_type.lower()
        self.fg_params = fg_params
        self.bg_params = bg_params

        self.nstates = len(self.states)
        self.n_haplotypes = int(max(hap for state in self.states for hap in state) + 1)
        self.ploidy = len(self.meioses)

    def _meiosis_array(self, X, meiosis_idx):
        arr = X[:, :, self.meioses[meiosis_idx]]
        if isinstance(arr, np.ma.MaskedArray):
            return np.ma.array(
                np.ascontiguousarray(arr.data),
                mask=np.ascontiguousarray(np.ma.getmaskarray(arr)),
            )
        return np.ascontiguousarray(arr)

    def _validate_input(self, X):
        if X.ndim != 3:
            raise ValueError('Input must have shape (n_seq, n_bins, n_haplotypes)')
        if X.shape[2] < self.n_haplotypes:
            raise ValueError(
                f'Input has {X.shape[2]} haplotype channels, but model expects at least '
                f'{self.n_haplotypes}'
            )

    def predict_state_proba(self, X, batch_size=128):
        return NotImplemented

    def predict_haplo_proba(self, X, batch_size=128):
        self._validate_input(X)
        p0 = self.rhmms[0].predict_haplo_proba(
            self._meiosis_array(X, 0),
            batch_size=batch_size,
        )
        p1 = self.rhmms[1].predict_haplo_proba(
            self._meiosis_array(X, 1),
            batch_size=batch_size,
        )
        haplo_proba = np.zeros((*p0.shape[:2], self.n_haplotypes), dtype=p0.dtype)
        haplo_proba[:, :, self.meioses[0]] += p0
        haplo_proba[:, :, self.meioses[1]] += p1
        return np.clip(haplo_proba, 0, self.ploidy)

    def predict(self, X, batch_size=128):
        return self.predict_haplo_proba(X, batch_size=batch_size)

    def log_probability(self, X, batch_size=128):
        self._validate_input(X)
        lp0 = self.rhmms[0].log_probability(
            self._meiosis_array(X, 0),
            batch_size=batch_size,
        )
        lp1 = self.rhmms[1].log_probability(
            self._meiosis_array(X, 1),
            batch_size=batch_size,
        )
        return lp0 + lp1

    def sample(self, X, n=1, batch_size=128, rng=DEFAULT_RNG):
        self._validate_input(X)
        s0 = self.rhmms[0].sample(
            self._meiosis_array(X, 0),
            n=n,
            batch_size=batch_size,
            rng=rng,
        )
        s1 = self.rhmms[1].sample(
            self._meiosis_array(X, 1),
            n=n,
            batch_size=batch_size,
            rng=rng,
        )
        sample = np.zeros((*s0.shape[:-1], self.n_haplotypes), dtype=s0.dtype)
        sample[..., self.meioses[0]] += s1
        sample[..., self.meioses[1]] += s1
        return sample

    @property
    def params(self):
        n = self.n_haplotypes


        params = {
            'is_independent_meioses': 1.0,
            'crossing_strategy': 3.0 if self.crossing_strategy == 'three_way' else 4.0,
            'states': [list(s) for s in self.states],
            'n_haplotypes': float(self.n_haplotypes),
            'rfactor': float(self.rfactor),
            'term_rfactor': float(self.term_rfactor),
            'trans_prob': float(self.trans_prob),
            'trans_prob_decay_rate': float(self.trans_prob_decay_rate),
            'is_poisson': 1.0 if self.dist_type == 'poisson' else 0.0,
            'fg_lambda': list(self.fg_params['lambda']) if self.dist_type == 'poisson' else [np.nan,] * n,
            'bg_lambda': list(self.bg_params['lambda']) if self.dist_type == 'poisson' else [np.nan,] * n,
            'fg_mean': list(self.fg_params['mean']) if self.dist_type == 'nb' else [np.nan,] * n,
            'bg_mean': list(self.bg_params['mean']) if self.dist_type == 'nb' else [np.nan,] * n,
            'fg_alpha': list(self.fg_params['alpha']) if self.dist_type == 'nb' else [np.nan,] * n,
            'bg_alpha': list(self.bg_params['alpha']) if self.dist_type == 'nb' else [np.nan,] * n,
            'fg_empty_fraction': list(self.fg_params['empty_fraction']),
            'bg_empty_fraction': list(self.bg_params['empty_fraction'])
        }
        return params

    @classmethod
    def from_params(cls, params, device=None):
        if not params.get('is_independent_meioses'):
            raise ValueError('params do not describe an IndependentMeiosesHMM')
        if params['is_poisson']:
            fg_params = {
                'lambda': np.array(params['fg_lambda']),
                'empty_fraction': np.array(params['fg_empty_fraction'])
            }
            bg_params = {
                'lambda': np.array(params['bg_lambda']),
                'empty_fraction': np.array(params['bg_empty_fraction'])
            }
        else:
            fg_params = {
                'mean': np.array(params['fg_mean']),
                'alpha': np.array(params['fg_alpha']),
                'empty_fraction': np.array(params['fg_empty_fraction'])
            }
            bg_params = {
                'mean': np.array(params['bg_mean']),
                'alpha': np.array(params['bg_alpha']),
                'empty_fraction': np.array(params['bg_empty_fraction'])
            }
        n_haplotypes = params.get('n_haplotypes')
        if n_haplotypes is None:
            n_haplotypes = max(hap for state in params['states'] for hap in state) + 1

        if int(params['crossing_strategy']) == 3:
            crossing_strategy = 'three_way'
        elif int(params['crossing_strategy']) == 4:
            crossing_strategy = 'four_way'
        else:
            return NotImplemented
        return cls(
            crossing_strategy, params['rfactor'], params['term_rfactor'],
            params['trans_prob'], fg_params, bg_params,
            dist_type='poisson' if params['is_poisson'] else 'nb',
            trans_prob_decay_rate=params['trans_prob_decay_rate'],
            n_haplotypes=int(n_haplotypes),
            device=device
        )
