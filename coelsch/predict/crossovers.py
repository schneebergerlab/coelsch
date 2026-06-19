import logging
from collections import namedtuple, Counter
import numpy as np
import pandas as pd

import torch

from .rhmm.utils import mask_array_zeros
from ..records import PredictionRecords, NestedData, NestedDataArray
from coelsch.main.logger import progress_bar
from coelsch.defaults import DEFAULT_RANDOM_SEED

log = logging.getLogger('coelsch')
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


def _transition_event(old_state, new_state, experiment_params):
    changed_pos = next(
        i for i, (old_hap, new_hap) in enumerate(zip(old_state, new_state))
        if old_hap != new_hap
    )

    if experiment_params.crossing_strategy == 'f2':
        return -1, int(np.sign(sum(new_state) - sum(old_state)))

    sign = int(np.sign(new_state[changed_pos] - old_state[changed_pos]))

    if experiment_params.ploidy == 1 or experiment_params.genotyping_strategy == 'recombinant':
        meiosis = 0
    elif experiment_params.crossing_strategy in {'backcross', 'testcross'}:
        meiosis = 1
    elif experiment_params.crossing_strategy in {'three_way', 'four_way'}:
        meiosis = changed_pos
    else:
        meiosis = 0

    return meiosis, sign


def samples_to_crossover_events(state_samples, states, experiment_params):
    """
    Convert sampled HMM state paths to crossover event arrays.

    Each event row is ``[bin_idx, meiosis, sign]``. ``meiosis`` is -1 for
    unphased F2 transitions, 0 for haploid/recombinant transitions, 1 for the
    segregating backcross/testcross meiosis, and the changed tuple position for
    three-way/four-way transitions.
    """
    n_seq, n_samples, _ = state_samples.shape
    states = tuple(tuple(state) for state in states)
    events = []

    for i in range(n_seq):
        seq_events = []
        for sample_idx in range(n_samples):
            path = state_samples[i, sample_idx]
            positions = np.nonzero(np.diff(path))[0]
            sample_events = []
            for bin_idx in positions:
                old_idx = path[bin_idx]
                new_idx = path[bin_idx + 1]
                meiosis, sign = _transition_event(
                    states[old_idx],
                    states[new_idx],
                    experiment_params,
                )
                sample_events.append((bin_idx, meiosis, sign))
            if sample_events:
                sample_events = np.asarray(sample_events, dtype=np.int32)
            else:
                sample_events = np.empty((0, 3), dtype=np.int32)
            seq_events.append(sample_events)
        events.append(seq_events)

    return events


def detect_crossovers(co_markers, rhmm, mask_empty_bins=True,
                      sample_paths=True, n_samples=10,
                      batch_size=128, processes=1, rng=DEFAULT_RNG,
                      show_progress=True):
    """
    Applies an rHMM to predict crossovers from marker data.

    Parameters
    ----------
    co_markers : MarkerRecords
        MarkerRecords dataset with haplotype specific read/variant information.
    rhmm : RigidHMM
        Fitted RigidHMM model.
    batch_size : int, optional
        Batch size for prediction (default: 128).
    processes : int, optional
        Number of threads for prediction (default: 1).

    Returns
    -------
    PredictionRecords
        PredictionRecords dataset with haplotype probabilities.
    """
    seen_barcodes = co_markers.barcodes
    co_preds = PredictionRecords.new_like(co_markers)
    if sample_paths:
        log.debug(f'Probable crossover locations will be sampled with {n_samples} bootstraps')
        crossover_samples=NestedDataArray(
            levels=('cb', 'chrom', 'sample')
        )
    torch.set_num_threads(processes)
    chrom_progress = progress_bar(
        co_markers.chrom_sizes,
        label='Predicting COs',
        item_show_func=str,
        hidden=not show_progress
    )
    logprobs = Counter()
    with chrom_progress:
        for chrom in chrom_progress:
            X = np.array([co_markers[cb, chrom] for cb in seen_barcodes])
            if mask_empty_bins:
                X = mask_array_zeros(X, axis=1)
            X_pred = rhmm.predict(X, batch_size=batch_size)
            X_logprob = rhmm.log_probability(X, batch_size=batch_size)
            for cb, p, lp in zip(seen_barcodes, X_pred, X_logprob):
                co_preds[cb, chrom] = p
                logprobs[cb] += lp
            if sample_paths:
                X_samp = rhmm.sample(X, n=n_samples, batch_size=batch_size, rng=rng)
                events = samples_to_crossover_events(
                    X_samp,
                    rhmm.states,
                    co_markers.experiment_params,
                )
                for cb, cb_events in zip(seen_barcodes, events):
                    for samp, sample_events in enumerate(cb_events):
                        crossover_samples[cb, chrom, str(samp)] = sample_events
    co_preds.add_metadata(
        rhmm_params=NestedData(
            levels=('misc', ),
            dtype=(float, list),
            data=rhmm.params
        ),
        logprobs=NestedData(
            levels=('cb',),
            dtype=(float),
            data=dict(logprobs),
        )
    )
    if sample_paths:
        co_preds.add_metadata(crossover_samples=crossover_samples)
    return co_preds
