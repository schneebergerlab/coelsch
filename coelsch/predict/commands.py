import os
import logging

import numpy as np
import torch

from .rhmm import train_rhmm, RigidHMM
from .crossovers import detect_crossovers
from .doublet import detect_doublets

from coelsch.utils import load_json, validate_ploidy
from coelsch import stats
from coelsch.defaults import DEFAULT_RANDOM_SEED


log = logging.getLogger('coelsch')
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)
DEFAULT_DEVICE = torch.device('cpu')


def run_predict(marker_json_fn, output_json_fn, *,
                co_markers=None,
                cb_whitelist_fn=None, bin_size=25_000, ploidy_type=None,
                segment_size=1_000_000, terminal_segment_size=50_000,
                cm_per_mb=4.5, interference_half_life=100_000, distribution_type='poisson',
                sample_paths=True, n_samples=10,
                predict_doublets=True, n_doublets=0.25, k_neighbours=0.25,
                generate_stats=True, write_bed=True, nco_min_prob_change=2.5e-3,
                output_precision=3, processes=1,
                batch_size=1_000, device=DEFAULT_DEVICE,
                rng=DEFAULT_RNG):
    """
    Runs the full haplotype prediction pipeline from marker JSON to output.

    Parameters
    ----------
    marker_json_fn : str
        Path to input JSON with haplotype specific marker data.
    output_json_fn : str
        Path to output predictions JSON file.
    co_markers : MarkerDataset, optional
        Loaded haplotype specific marker dataset.
    cb_whitelist_fn : str, optional
        Path to barcode whitelist file.
    bin_size : int, optional
        Genomic bin size (default: 25,000).
    ploidy_type : str, optional
        Ploidy type of data used to infer type of model to use. Options are
        "haploid" with model states [0, 1], "diploid_bc1" with states [00, 01],
        or "diploid_f2" with states [00, 01, 11]. Default is to infer from data
        if possible, else "haploid"
    segment_size : int, optional
        Size of internal segments for modeling (default: 1,000,000).
    terminal_segment_size : int, optional
        Size of terminal segments (default: 50,000).
    cm_per_mb : float, optional
        Centimorgan per megabase rate (default: 4.5).
    interference_half_life : int, optional
        Distance below segment_size at which recombination rate halves (exponentially)
        to enforce crossover-interference. If <1, this is considered a fraction of
        segment_size, else it is an absolute value in basepairs (default 0.1).
    distribution_type : str, optional
        Type of distribution to use in rHMM - can be either poisson or nb.
    sample_paths : bool, optional
        Whether to sample stochastic crossover locations for each barcode (default: True)
    n_samples : int, optional
        The number of samples to take for each barcode (default: 10)
    predict_doublets : bool, optional
        Whether to detect doublets (default: True).
    n_doublets : float or int, optional
        Number or fraction of doublets to simulate (default: 0.25).
    k_neighbours : float or int, optional
        Number or fraction of neighbors for KNN (default: 0.25).
    generate_stats : bool, optional
        Whether to generate prediction statistics (default: True).
    write_bed : bool, optional
        Optional bed file of predicted haplotypes (default: True)
    nco_min_prob_change : float, optional
        Threshold for detecting non-crossover changes (default: 2.5e-3).
    output_precision : int, optional
        Decimal precision for JSON output.
    processes : int, optional
        Number of threads (default: 1).
    batch_size : int, optional
        Batch size for prediction.
    device : torch.device, optional
        Device to run models on.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    PredictionRecords
        Final predictions including crossover and optional doublet annotations.
    """
    if co_markers is None:
        co_markers = load_json(marker_json_fn, cb_whitelist_fn, bin_size)
    ploidy_type = validate_ploidy(co_markers, ploidy_type)

    rhmm = train_rhmm(
        co_markers,
        model_type=ploidy_type,
        cm_per_mb=cm_per_mb,
        segment_size=segment_size,
        terminal_segment_size=terminal_segment_size,
        interference_half_life=interference_half_life,
        dist_type=distribution_type,
        device=device,
    )
    co_preds = detect_crossovers(
        co_markers, rhmm,
        sample_paths=sample_paths, n_samples=n_samples,
        batch_size=batch_size, processes=processes, rng=rng,
    )
    if predict_doublets:
        co_preds = detect_doublets(
            co_markers, co_preds, rhmm, n_doublets, batch_size=batch_size,
            processes=processes, rng=rng,
        )

    if output_json_fn is not None:
        if generate_stats:
            output_tsv_fn = f'{os.path.splitext(output_json_fn)[0]}.stats.tsv'
            stats.run_stats(
                None, None, output_tsv_fn,
                co_markers=co_markers,
                co_preds=co_preds,
                nco_min_prob_change=nco_min_prob_change,
                output_precision=output_precision
            )
        log.info(f'Writing predictions to {output_json_fn}')
        co_preds.write_json(output_json_fn, output_precision)
        if write_bed:
            output_bed_fn = f'{os.path.splitext(output_json_fn)[0]}.bed'
            co_preds.write_bed(output_bed_fn, precision=2)
    return co_preds


def run_doublet(marker_json_fn, pred_json_fn, output_json_fn, *,
                cb_whitelist_fn=None, bin_size=25_000, n_doublets=500,
                generate_stats=True, output_precision=3, batch_size=1_000,
                processes=1, device=DEFAULT_DEVICE, rng=DEFAULT_RNG):
    """
    Loads pre-existing crossover predictions and performs doublet detection.

    Parameters
    ----------
    marker_json_fn : str
        Path to crossover marker JSON.
    pred_json_fn : str
        Path to precomputed predictions JSON.
    output_json_fn : str
        Output path to write updated predictions.
    cb_whitelist_fn : str, optional
        Whitelist file for barcodes.
    bin_size : int, optional
        Genomic bin size (default: 25,000).
    n_doublets : float or int, optional
        Number or fraction of doublets to simulate.
    generate_stats : bool, optional
        Whether to generate output statistics (default: True).
    output_precision : int, optional
        Decimal precision for output (default: 3).
    batch_size : int, optional
        Batch size for HMM prediction.
    processes : int, optional
        Number of threads to use.
    device : torch.device, optional
        Model device (default: CPU).
    rng : np.random.Generator, optional
        Random number generator instance.

    Returns
    -------
    PredictionRecords
        Crossover/haplotype predictions with doublet probabilities added.
    """
    co_markers = load_json(marker_json_fn, cb_whitelist_fn, bin_size)
    co_preds = load_json(
        pred_json_fn, cb_whitelist_fn, bin_size, data_type='predictions'
    )

    if set(co_preds.barcodes) != set(co_markers.barcodes):
        raise ValueError('Cell barcodes from marker-json-fn and predict-json-fn do not match')

    rhmm = RigidHMM.from_params(co_preds.metadata['rhmm_params'], device=device)

    co_preds = detect_doublets(
        co_markers, co_preds, rhmm, n_doublets=n_doublets,
        batch_size=batch_size, processes=processes, rng=rng,
    )

    if output_json_fn is not None:
        if generate_stats:
            output_tsv_fn = f'{os.path.splitext(output_json_fn)[0]}.stats.tsv'
            stats.run_stats(
                None, None, output_tsv_fn,
                co_markers=co_markers,
                co_preds=co_preds,
                output_precision=output_precision
            )
        log.info(f'Writing predictions to {output_json_fn}')
        co_preds.write_json(output_json_fn, output_precision)
    return co_preds
