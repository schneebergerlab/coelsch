import os
import logging
import numpy as np
import pandas as pd

from coelsch.utils import load_json
from coelsch.records import MarkerRecords, PredictionRecords, NestedDataArray
from coelsch.experiment.params import ExperimentParams
from coelsch.clean.filter import filter_low_coverage_barcodes
from coelsch.defaults import DEFAULT_RANDOM_SEED
from coelsch.stats import marker_agreement_fraction


log = logging.getLogger('coelsch')
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


def co_invs_to_gt(co_invs, bin_size, chrom_nbins):
    """
    Convert intervals from a BED file representing haplotypes into a binary NumPy array.

    Parameters
    ----------
    co_invs : DataFrame
        A DataFrame containing columns ['start', 'end', 'haplo'] representing haplotype intervals.
    bin_size : int
        Size of genomic bins.
    chrom_nbins : int
        Number of bins for the chromosome.

    Returns
    -------
    gt : ndarray
        Binary NumPy array representing haplotype values per bin.

    Raises
    ------
    ValueError
        If intervals do not completely cover all chromosomes.
    """
    gt = np.full(chrom_nbins, np.nan)

    start_bin = np.ceil(co_invs.start.values / bin_size).astype(int)
    end_bin = np.ceil(co_invs.end.values / bin_size).astype(int)
    haplo = co_invs.haplo.values

    for s, e, h in zip(start_bin, end_bin, haplo):
        gt[s: e] = h
    if np.isnan(gt).any():
        raise ValueError('Supplied intervals in haplo-bed-fn do not completely cover chromosomes')
    return gt


def read_ground_truth_haplotypes_bed(co_invs_fn, chrom_sizes, bin_size=25_000):
    """
    Read legacy F1 gamete BED ground truth as scalar 0/1 haplotype intervals.
    """
    co_invs = pd.read_csv(
        co_invs_fn,
        sep='\t',
        names=['chrom', 'start', 'end', 'sample_id', 'haplo', 'strand']
    )
    if not set(co_invs.haplo.unique()).issubset({0, 1}):
        raise ValueError('BED ground truth only supports haplotype labels 0 and 1')

    experiment_params = ExperimentParams(
        lifecycle_stage='gametes',
        crossing_strategy='f1',
        sequencing_type='other',
        genotyping_strategy='founder',
    )
    gt = PredictionRecords(chrom_sizes, bin_size, experiment_params)

    for sample_id, sample_invs in co_invs.groupby('sample_id'):
        for chrom, n in gt.nbins.items():
            chrom_invs = sample_invs.query('chrom == @chrom')
            gt[sample_id, chrom] = co_invs_to_gt(chrom_invs, bin_size, n)
    return gt


def random_bg_sample(m, n_bg, bg_signal=None, rng=DEFAULT_RNG):
    """
    Randomly sample background signal proportionally to observed counts and background model.

    Parameters
    ----------
    m : np.ndarray
        Marker count matrix with shape (bins, haplotypes).
    bg_signal : np.ndarray
        Background probabilities with shape (bins, haplotypes).
    n_bg : int
        Number of background markers to sample.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        Matrix of sampled background markers.
    """
    if bg_signal is None:
        bg_signal = np.ones_like(m)
    bg_idx = np.nonzero(m)
    m_valid = m[bg_idx]
    p = m_valid * bg_signal[bg_idx]
    bg = np.zeros_like(m)
    p_denom = p.sum(axis=None)
    if p_denom == 0:
        return bg
    p = p / p_denom
    n_p = p.shape[0]
    bg_c = np.bincount(rng.choice(np.arange(n_p), size=n_bg, replace=True, p=p), minlength=n_p)
    bg[bg_idx] = np.minimum(bg_c, m_valid)
    return bg


def _ground_truth_dosage(ground_truth, sample_id, chrom, thresholded=True):
    dosage = ground_truth.get_haplotype_dosage(sample_id, chrom)
    if thresholded:
        dosage = np.round(dosage)
    return dosage.astype(float, copy=False)


def _barcode_noise_fraction(co_markers, co_preds, cb, thresholded=True):
    source_dosage = {
        chrom: co_preds.get_haplotype_dosage(cb, chrom)
        for chrom in co_preds.chrom_sizes
    }
    agreement = marker_agreement_fraction(
        co_markers[cb],
        source_dosage,
        thresholded=thresholded,
    )
    if np.isnan(agreement):
        return 0.0
    return np.clip(1.0 - agreement, 0.0, 1.0)


def _validate_source_predictions(co_markers, co_preds):
    if set(co_markers.barcodes) != set(co_preds.barcodes):
        raise ValueError('Source marker and prediction barcodes do not match')
    if co_markers.chrom_sizes != co_preds.chrom_sizes:
        raise ValueError('Source marker and prediction chromosome sizes do not match')
    if co_markers.bin_size != co_preds.bin_size:
        raise ValueError('Source marker and prediction bin sizes do not match')


def _validate_ground_truth_compatibility(co_markers, ground_truth):
    if co_markers.chrom_sizes != ground_truth.chrom_sizes:
        raise ValueError('Source markers and ground truth chromosome sizes do not match')
    if co_markers.bin_size != ground_truth.bin_size:
        raise ValueError('Source markers and ground truth bin sizes do not match')


def apply_gt_dosage_to_markers(gt_dosage, m, noise_fraction, rng=DEFAULT_RNG):
    """
    Redistribute source marker row sums across target haplotypes.

    ``gt_dosage`` defines the target inherited haplotype dosage per bin. The
    barcode-specific ``noise_fraction`` mixes this foreground profile with a
    uniform background profile before multinomially sampling the source row sums.
    """
    if gt_dosage.ndim != 2:
        raise ValueError('ground-truth dosage must have shape (bins, haplotypes)')
    if len(gt_dosage) != len(m):
        raise ValueError('ground-truth dosage and marker arrays have different bin counts')

    row_totals = m.sum(axis=1).astype(int)
    dosage_total = gt_dosage.sum(axis=1, keepdims=True)
    fg_profile = gt_dosage / np.maximum(dosage_total, 1e-12)
    bg_profile = np.full_like(fg_profile, 1.0 / fg_profile.shape[1], dtype=float)

    p = (1.0 - noise_fraction) * fg_profile + noise_fraction * bg_profile
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)

    sim = np.zeros_like(gt_dosage, dtype=m.dtype)
    for i, n in enumerate(row_totals):
        if n > 0:
            sim[i] = rng.multinomial(int(n), p[i])
    return sim


def simulate_singlets(co_markers, co_preds, ground_truth, nsim_per_sample,
                      thresholded=True, noise_fraction=None, rng=DEFAULT_RNG):
    """
    Simulate single-cell barcodes from source depth/noise and target ground truth.

    Source marker row sums provide depth, source predictions provide barcode-level
    noise estimates, and ground truth provides the target haplotype dosage profile.
    """
    _validate_source_predictions(co_markers, co_preds)
    _validate_ground_truth_compatibility(co_markers, ground_truth)

    sim_co_markers = MarkerRecords(
        ground_truth.chrom_sizes,
        co_markers.bin_size,
        ground_truth.experiment_params,
    )
    sim_co_markers.add_metadata(ground_truth=NestedDataArray(levels=('cb', 'chrom')))

    for sample_id in ground_truth.barcodes:
        cbs_to_sim = rng.choice(co_markers.barcodes, replace=False, size=nsim_per_sample)
        for cb in cbs_to_sim:
            sim_id = f'{sample_id}:{cb}'
            cb_noise = (
                np.clip(noise_fraction, 0.0, 1.0)
                if noise_fraction is not None
                else _barcode_noise_fraction(co_markers, co_preds, cb, thresholded=thresholded)
            )
            for chrom in ground_truth.chrom_sizes:
                gt_dosage = _ground_truth_dosage(
                    ground_truth, sample_id, chrom, thresholded=thresholded
                )
                sim_co_markers[sim_id, chrom] = apply_gt_dosage_to_markers(
                    gt_dosage,
                    co_markers[cb, chrom],
                    cb_noise,
                    rng=rng,
                )
                sim_co_markers.metadata['ground_truth'][sim_id, chrom] = gt_dosage

    return sim_co_markers


def simulate_doublets(co_markers, n_doublets, doublet_weight=None, doublet_ratio_scale=0.1, ratio_clip=0.1, rng=DEFAULT_RNG):
    """
    Simulate doublet barcodes by summing markers from random barcode pairs.

    Parameters
    ----------
    co_markers : MarkerRecords
        Haplotype-specific markers from a real dataset, to use as basis for simulation.
    n_doublets : int
        Number of doublet barcodes to simulate.
    doublet_ratio_scale : float
        The scale of the normal distribution (around 0.5) used to create mixing ratios of doublets
    ratio_clip : float
        The minimum fraction that a barcode can contribute to a doublet
    rng : Generator, optional
        NumPy random generator.

    Returns
    -------
    sim_co_markers_doublets : MarkerRecords
        Simulated haplotype-specific marker records for doublets.
    """
    sim_co_markers_doublets = MarkerRecords.new_like(co_markers, copy_metadata=False)
    barcodes = np.array(co_markers.barcodes)
    if doublet_weight is not None:
        dw = np.array([doublet_weight[cb] for cb in barcodes])
    else:
        dw = np.ones(len(barcodes))
    dw /= dw.sum()
    n_barcodes = len(barcodes)
    marker_counts = np.array([co_markers.total_marker_count(b) for b in barcodes])
    i_positions = rng.choice(np.arange(len(barcodes)), size=n_doublets, p=dw)
    probs = 1.0 / (np.abs(marker_counts[None, :] - marker_counts[i_positions, None]) + 1e-6)
    for k, i in enumerate(i_positions):
        cb_i = barcodes[i]
        p = probs[k] * dw
        p[i] = 0.0
        p /= p.sum()
        cb_j = rng.choice(barcodes, p=p)
        assert cb_i != cb_j
        sim_id = f'doublet{k}:{cb_i}_{cb_j}'
        i_frac = np.clip(
            rng.normal(loc=0.5, scale=doublet_ratio_scale),
            a_min=ratio_clip,
            a_max=1.0 - ratio_clip
        )
        for chrom in sim_co_markers_doublets.chrom_sizes:
            m_i = co_markers[cb_i, chrom]
            m_j = co_markers[cb_j, chrom]
            doublet_n_markers = (m_i.sum() + m_j.sum()) // 2
            m_i_samp = random_bg_sample(m_i, int(doublet_n_markers * i_frac))
            m_j_samp = random_bg_sample(m_j, int(doublet_n_markers * (1 - i_frac)))
            sim_co_markers_doublets[sim_id, chrom] = m_i_samp + m_j_samp
    return sim_co_markers_doublets


def generate_simulated_data(co_markers, co_preds, ground_truth,
                            noise_fraction=None, nsim_per_sample=100,
                            doublet_rate=0.0, thresholded=True,
                            rng=DEFAULT_RNG):
    """
    Generate simulated crossover marker data using source depth/noise and target ground truth.

    Parameters
    ----------
    co_markers : MarkerRecords
        Source haplotype-specific markers whose row sums provide simulated depth.
    co_preds : PredictionRecords
        Source predictions for ``co_markers`` used to estimate barcode-specific noise.
    ground_truth : PredictionRecords
        Target ground truth haplotype dosage/calls to be simulated.
    noise_fraction : float, optional
        Fixed noise fraction. If omitted, estimate per-source-barcode noise from ``co_preds``.
    nsim_per_sample : int, optional
        Number of simulated cells per ground truth sample.
    doublet_rate : float, optional
        Fraction or number of doublets to simulate.
    thresholded : bool, optional
        If True, round source predictions and target ground truth dosage before simulation.
    rng : Generator, optional
        NumPy random generator.

    Returns
    -------
    sim_co_markers : MarkerRecords
        Simulated haplotype-specific marker records, including optional doublet barcodes.
    """
    sim_co_markers = simulate_singlets(
        co_markers,
        co_preds,
        ground_truth,
        nsim_per_sample,
        thresholded=thresholded,
        noise_fraction=noise_fraction,
        rng=rng,
    )

    if doublet_rate:
        doublet_rate = int(len(sim_co_markers) * doublet_rate) if doublet_rate < 1 else int(doublet_rate)
        log.info(f'Simulating {doublet_rate} doublet barcodes')
        sim_co_markers.merge(
            simulate_doublets(sim_co_markers, doublet_rate, rng=rng),
            inplace=True
        )
    return sim_co_markers


def ground_truth_from_marker_records(co_markers):
    """
    Extract ground truth haplotype calls from marker record metadata.

    Parameters
    ----------
    co_markers : MarkerRecords
        Simulated marker records containing ground truth metadata.

    Returns
    -------
    ground_truth : PredictionRecords
        Extracted ground truth haplotypes.
    """
    ground_truth = PredictionRecords.new_like(co_markers)
    for cb, sd in co_markers.metadata['ground_truth'].items():
        for chrom, arr in sd.items():
            ground_truth[cb, chrom] = np.array(arr)
    return ground_truth


def run_sim(marker_json_fn, pred_json_fn, output_json_fn, ground_truth_fn, *,
            cb_whitelist_fn=None, bin_size=25_000,
            min_markers_per_cb=100, min_markers_per_chrom=20,
            noise_fraction=None, nsim_per_sample=100, n_doublets=0.0,
            rng=DEFAULT_RNG):
    """
    Run the full simulation pipeline to create synthetic marker data from ground truth.

    Parameters
    ----------
    marker_json_fn : str
        Path to marker JSON file.
    pred_json_fn : str
        Path to prediction JSON for source marker barcodes. Used to estimate noise.
    output_json_fn : str
        Output path for simulated JSON file.
    ground_truth_fn : str
        Path to ground truth file (BED or JSON).
    cb_whitelist_fn : str, optional
        Optional path to cell barcode whitelist.
    bin_size : int, optional
        Genomic bin size.
    min_markers_per_cb : int, optional
        Minimum number of markers required per barcode (default is 100).
    min_markers_per_chrom : int, optional
        Minimum number of markers required per chromosome (default is 20).
    noise_fraction : float, optional
        Fixed fraction of markers to sample as uniform background noise.
    nsim_per_sample : int, optional
        Number of simulations per ground truth haplotype.
    n_doublets : float, optional
        Number or fraction of doublets to simulate.
    rng : Generator, optional
        NumPy random generator.

    Returns
    -------
    sim_co_markers : MarkerRecords
        Simulated marker data.
    """
    co_markers = load_json(marker_json_fn, cb_whitelist_fn, bin_size)
    co_preds = load_json(
        pred_json_fn, cb_whitelist_fn, bin_size, data_type='predictions'
    )
    if min_markers_per_cb or min_markers_per_chrom:
        co_markers = filter_low_coverage_barcodes(
            co_markers, min_markers_per_cb, min_markers_per_chrom
        )
        co_preds = co_preds.filter(co_markers.barcodes, inplace=False)

    if os.path.splitext(ground_truth_fn)[1] == '.bed':
        ground_truth_haplotypes = read_ground_truth_haplotypes_bed(
            ground_truth_fn, co_markers.chrom_sizes, bin_size
        )
    else:
        ground_truth_haplotypes = PredictionRecords.read_json(ground_truth_fn)
    log.info(f'Read {len(ground_truth_haplotypes)} ground truth samples from {ground_truth_fn}')
    sim_co_markers = generate_simulated_data(
        co_markers,
        co_preds,
        ground_truth_haplotypes,
        noise_fraction=noise_fraction,
        nsim_per_sample=nsim_per_sample,
        doublet_rate=n_doublets,
        rng=rng
    )
    log.info(f'Simulated {len(sim_co_markers)} barcodes total')
    if output_json_fn is not None:
        log.info(f'Writing markers to {output_json_fn}')
        sim_co_markers.write_json(output_json_fn)
    return sim_co_markers
