"""
Independent-meiosis wrapper models for rigid HMM prediction.
"""
import numpy as np

from coelsch.defaults import DEFAULT_RANDOM_SEED


DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


class IndependentMeiosesHMM:
    """
    Run one haploid RigidHMM independently over multiple meioses.

    This wrapper currently targets four-way crosses, where marker channels
    ``(0, 1)`` and ``(2, 3)`` correspond to independent haploid meioses. It
    exposes the public methods used by crossover prediction while returning
    full four-way state and marginal haplotype outputs.
    """

    def __init__(self, haploid_hmm, meioses=((0, 1), (2, 3)), states=None):
        if len(meioses) != 2:
            raise ValueError('IndependentMeiosesHMM currently requires exactly two meioses')
        if any(len(meiosis) != 2 for meiosis in meioses):
            raise ValueError('Each meiosis must contain exactly two haplotype channels')

        self.haploid_hmm = haploid_hmm
        self.meioses = tuple(tuple(int(hap) for hap in meiosis) for meiosis in meioses)
        if states is None:
            states = tuple(
                (left, right)
                for left in self.meioses[0]
                for right in self.meioses[1]
            )
        self.states = tuple(tuple(int(hap) for hap in state) for state in states)
        self.nstates = len(self.states)
        self.n_haplotypes = int(max(hap for state in self.states for hap in state) + 1)
        self.ploidy = len(self.meioses)

    def _meiosis_array(self, X, meiosis_idx):
        return X[:, :, self.meioses[meiosis_idx]]

    def _validate_input(self, X):
        if X.ndim != 3:
            raise ValueError('Input must have shape (n_seq, n_bins, n_haplotypes)')
        if X.shape[2] < self.n_haplotypes:
            raise ValueError(
                f'Input has {X.shape[2]} haplotype channels, but model expects at least '
                f'{self.n_haplotypes}'
            )

    def predict_state_proba(self, X, batch_size=128):
        self._validate_input(X)
        p0 = self.haploid_hmm.predict_haplo_proba(
            self._meiosis_array(X, 0),
            batch_size=batch_size,
        )
        p1 = self.haploid_hmm.predict_haplo_proba(
            self._meiosis_array(X, 1),
            batch_size=batch_size,
        )
        return np.stack(
            [
                p0[:, :, 0] * p1[:, :, 0],
                p0[:, :, 0] * p1[:, :, 1],
                p0[:, :, 1] * p1[:, :, 0],
                p0[:, :, 1] * p1[:, :, 1],
            ],
            axis=-1,
        )

    def predict_haplo_proba(self, X, batch_size=128):
        self._validate_input(X)
        p0 = self.haploid_hmm.predict_haplo_proba(
            self._meiosis_array(X, 0),
            batch_size=batch_size,
        )
        p1 = self.haploid_hmm.predict_haplo_proba(
            self._meiosis_array(X, 1),
            batch_size=batch_size,
        )
        haplo_proba = np.zeros((*p0.shape[:2], self.n_haplotypes), dtype=p0.dtype)
        haplo_proba[:, :, self.meioses[0]] = p0
        haplo_proba[:, :, self.meioses[1]] = p1
        return np.clip(haplo_proba, 0, self.ploidy)

    def predict(self, X, batch_size=128):
        return self.predict_haplo_proba(X, batch_size=batch_size)

    def log_probability(self, X, batch_size=128):
        self._validate_input(X)
        lp0 = self.haploid_hmm.log_probability(
            self._meiosis_array(X, 0),
            batch_size=batch_size,
        )
        lp1 = self.haploid_hmm.log_probability(
            self._meiosis_array(X, 1),
            batch_size=batch_size,
        )
        return lp0 + lp1

    def sample(self, X, n=1, batch_size=128, rng=DEFAULT_RNG):
        self._validate_input(X)
        s0 = self.haploid_hmm.sample(
            self._meiosis_array(X, 0),
            n=n,
            batch_size=batch_size,
            rng=rng,
        )
        s1 = self.haploid_hmm.sample(
            self._meiosis_array(X, 1),
            n=n,
            batch_size=batch_size,
            rng=rng,
        )
        return (s0 * 2 + s1).astype(np.int16)

    @property
    def params(self):
        params = {
            'is_independent_meioses': 1.0,
            'states': [list(state) for state in self.states],
            'n_haplotypes': float(self.n_haplotypes),
            'ploidy': float(self.ploidy),
            'meioses': [list(meiosis) for meiosis in self.meioses],
        }
        params.update({f'haploid_{key}': value for key, value in self.haploid_hmm.params.items()})
        return params

    @classmethod
    def from_params(cls, params, device=None):
        from .model import RigidHMM, DEFAULT_DEVICE

        if not params.get('is_independent_meioses'):
            raise ValueError('params do not describe an IndependentMeiosesHMM')
        if device is None:
            device = DEFAULT_DEVICE

        haploid_params = {
            key[len('haploid_'):]: value
            for key, value in params.items()
            if key.startswith('haploid_')
        }
        haploid_hmm = RigidHMM.from_params(haploid_params, device=device)
        meioses = params.get('meioses', ((0, 1), (2, 3)))
        states = params.get('states')
        return cls(haploid_hmm, meioses=meioses, states=states)
