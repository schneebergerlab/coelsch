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


def samples_to_crossover_positions(haplotype_samples):
    """
    Convert haplotype state samples to crossover positions and directions.

    Parameters
    ----------
    haplotype_samples : np.ndarray, shape (n_seq, n_samples, n_bins)
        Haplotype identity paths (0/1 per bin).

    Returns
    -------
    co_pos : list[list[np.ndarray]]
        For each sequence and sample, array of crossover bin indices.
    co_signs : list[list[np.ndarray]]
        Same structure, with +1 for 0→1 and –1 for 1→0 transitions.
    """
    n_seq, n_samples, n_bins = haplotype_samples.shape
    co_pos, co_signs = [], []

    diffs = np.diff(haplotype_samples, axis=2)

    for i in range(n_seq):
        seq_diffs = diffs[i]
        seq_pos, seq_sign = [], []
        for d in seq_diffs:
            p = np.nonzero(d)[0]
            seq_pos.append(p)
            seq_sign.append(np.sign(d[p]))
        co_pos.append(seq_pos)
        co_signs.append(seq_sign)

    return co_pos, co_signs


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
                co_pos, co_signs = samples_to_crossover_positions(X_samp)
                for cb, pos, sgn in zip(seen_barcodes, co_pos, co_signs):
                    for samp, (p, s) in enumerate(zip(pos, sgn)):
                        crossover_samples[cb, chrom, str(samp)] = np.stack([p, s], axis=-1)
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
