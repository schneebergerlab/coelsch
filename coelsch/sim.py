import os
import logging
from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd

from coelsch.utils import load_json
from coelsch.records import MarkerRecords, PredictionRecords, NestedDataArray
from coelsch.clean.filter import filter_low_coverage_barcodes
from coelsch.clean.background import estimate_overall_background_signal
from coelsch.signal import smooth_counts_sum
from coelsch.defaults import DEFAULT_RANDOM_SEED


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
    Read a BED file containing haplotype intervals and convert to binned binary arrays.

    Parameters
    ----------
    co_invs_fn : str
        Path to BED file.
    chrom_sizes : dict
        Dictionary mapping chromosome names to their lengths.
    bin_size : int, optional
        Size of genomic bins. Default is 25,000.

    Returns
    -------
    gt : PredictionRecords
        PredictionRecords object with ground truth haplotype data.
    """
    co_invs = pd.read_csv(
        co_invs_fn,
        sep='\t',
        names=['chrom', 'start', 'end', 'sample_id', 'haplo', 'strand']
    )

    gt = PredictionRecords(chrom_sizes, bin_size, set(co_invs.sample_id))

    for sample_id, sample_invs in co_invs.groupby('sample_id'):
        for chrom, n in gt.nbins.items():
            chrom_invs = sample_invs.query('chrom == @chrom')
            gt[sample_id, chrom] = co_invs_to_gt(chrom_invs, bin_size, n)
    return gt


def read_ground_truth_haplotypes_json(pred_json_fn, bc_haplotype=0):
    """
    Read ground truth haplotypes from a JSON file.

    Parameters
    ----------
    pred_json_fn : str
        Path to the JSON file.

    Returns
    -------
    gt : PredictionRecords
        Ground truth haplotypes with haplotype probabilities rounded to integers.
    """
    gt = PredictionRecords.read_json(pred_json_fn)
    if gt.ploidy_type == 'haploid':
        for *_, m in gt.deep_items():
            np.round(m, decimals=0, out=m)
    elif gt.ploidy_type == 'diploid_bc1':
        for *_, m in gt.deep_items():
            np.round(2 * (m - 0.5 * bc_haplotype), decimals=0, out=m)
    elif gt.ploidy_type == 'diploid_f2':
        raise NotImplementedError('Simulation with a "diploid_f2" ground truth sample is currently not supported')
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


def apply_gt_to_markers(gt, m, bg_rate, bg_signal, conv_bins, rng=DEFAULT_RNG):
    """
    Simulate marker counts using a known ground truth haplotype.

    Parameters
    ----------
    gt : ndarray
        Ground truth haplotype (binary array).
    m : ndarray
        Input marker array for a single chromosome.
    bg_rate : float
        Estimated background rate for the cell.
    bg_signal : ndarray
        Per-bin background signal for the chromosome.
    conv_bins : convolution size for rough fg/bg estimation
    rng : Generator, optional
        NumPy random generator.

    Returns
    -------
    sim : ndarray
        Simulated marker array with shape (n_bins, 2).
    """
    gt = gt.astype(int)
    s = len(m)

    m_smooth = smooth_counts_sum(m, conv_bins)
    nf = m_smooth.mean()
    bg_probs = 1 - (m_smooth + nf / 2) / (m_smooth.sum(axis=1, keepdims=True) + nf)
    # simulate a realistic background signal using the average background across the dataset
    tot = m.sum(axis=None)
    if tot:
        n_bg = round(tot * bg_rate)
        bg = random_bg_sample(m, n_bg, bg_signal * bg_probs, rng=rng)
    else:
        bg = m.copy()
    fg = m - bg

    # flatten haplotypes
    fg = fg.sum(axis=1)
    bg = bg.sum(axis=1)

    sim = np.zeros(shape=(s, 2))
    idx = np.arange(s)

    # apply fg at ground truth haplotype, and bg on other haplotype
    sim[idx, gt] = fg
    sim[idx, 1 - gt] = bg

    return sim


def simulate_singlets(co_markers, ground_truth, bg_signal, frac_bg, nsim_per_sample,
                     conv_window_size, rng=DEFAULT_RNG):
    """
    Simulate single-cell barcodes based on ground truth haplotypes.

    Parameters
    ----------
    co_markers : MarkerRecords
        Haplotype-specific markers from a real dataset, to use as basis for simulation.
    ground_truth : PredictionRecords
        Ground truth haplotype calls from a different dataset, to be simulated.
    bg_signal : dict
        Per-bin background signal per chromosome.
    frac_bg : dict
        Background fraction per cell barcode.
    nsim_per_sample : int
        Number of simulated cells per ground truth haplotype.
    conv_window_size : int, optional
        Window size for convolution-based background estimation.
    rng : Generator, optional
        NumPy random generator.

    Returns
    -------
    sim_co_markers : MarkerRecords
        Simulated haplotype-specific marker records.
    """
    sim_co_markers = MarkerRecords.new_like(co_markers, copy_metadata=False)
    # copy non- barcode-specific metadata, regenerate barcode-specific metadata with new sim barcodes:
    metadata_to_recreate = {}
    for md_name, metadata in co_markers.metadata.items():
        if 'cb' in metadata.levels:
            sim_co_markers.add_metadata(**{md_name: metadata.new_like(metadata)})
            metadata_to_recreate[md_name] = metadata.levels.index('cb')
        else:
            sim_co_markers.add_metadata(**{md_name: metadata.copy()})
    sim_co_markers.add_metadata(ground_truth=NestedDataArray(levels=('cb', 'chrom')))
    sim_id_mapping = defaultdict(list)
    conv_bins = conv_window_size // co_markers.bin_size
    for sample_id in ground_truth.barcodes:
        cbs_to_sim = rng.choice(co_markers.barcodes, replace=False, size=nsim_per_sample)
        for cb in cbs_to_sim:
            sim_id = f'{sample_id}:{cb}'
            sim_id_mapping[cb].append(sim_id)
            for chrom in ground_truth.chrom_sizes:
                gt = ground_truth[sample_id, chrom]
                sim_co_markers[sim_id, chrom] = apply_gt_to_markers(
                    gt, co_markers[cb, chrom],
                    frac_bg[cb], bg_signal[chrom], conv_bins, rng=rng
                )
                sim_co_markers.metadata['ground_truth'][sim_id, chrom] = gt
    for md_name, lvl_idx in metadata_to_recreate.items():
        for metadata_idx, val in co_markers.metadata[md_name].deep_items():
            for sim_id in sim_id_mapping[metadata_idx[lvl_idx]]:
                sim_metadata_idx = metadata_idx[:lvl_idx] + (sim_id, ) + metadata_idx[lvl_idx + 1:]
                sim_co_markers.metadata[md_name][sim_metadata_idx] = val

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
            if rng.random() > 0.5:
                m_i = np.flip(m_i, axis=1)
            if rng.random() > 0.5:
                m_j = np.flip(m_j, axis=1)
            doublet_n_markers = (m_i.sum() + m_j.sum()) // 2
            m_i_samp = random_bg_sample(m_i, int(doublet_n_markers * i_frac))
            m_j_samp = random_bg_sample(m_j, int(doublet_n_markers * (1 - i_frac)))
            sim_co_markers_doublets[sim_id, chrom] = m_i_samp + m_j_samp
    return sim_co_markers_doublets


def generate_simulated_data(co_markers, ground_truth,
                            conv_window_size=2_500_000,
                            bg_rate=None, nsim_per_sample=100,
                            doublet_rate=0.0, rng=DEFAULT_RNG):
    """
    Generate simulated crossover marker data using real background and ground truth haplotypes.

    Parameters
    ----------
    co_markers : MarkerRecords
        Haplotype-specific markers from a real dataset, to use as basis for simulation.
    ground_truth : PredictionRecords
        Ground truth haplotype calls from a different dataset, to be simulated.
    conv_window_size : int, optional
        Window size for convolution-based background estimation.
    bg_rate : float, optional
        Fixed background rate (override background rate estimation from data).
    nsim_per_sample : int, optional
        Number of simulated cells per ground truth haplotype.
    doublet_rate : float, optional
        Fraction or number of doublets to simulate.
    rng : Generator, optional
        NumPy random generator.

    Returns
    -------
    sim_co_markers : MarkerRecords
        Simulated haplotype-specific marker records, including simulated singlet and doublet barcodes
    """
    co_markers = estimate_overall_background_signal(
        co_markers,
        conv_window_size,
        max_frac_bg=1.0,
        apply_per_geno=False # todo, should this be exposed?
    )
    bg_signal = co_markers.metadata['background_signal']['ungrouped'] # bit of a hack
    frac_bg = co_markers.metadata['estimated_background_fraction']
    if bg_rate is not None:
        frac_bg = defaultdict(lambda: bg_rate)

    sim_co_markers = simulate_singlets(
        co_markers, ground_truth, bg_signal, frac_bg, nsim_per_sample,
        conv_window_size, rng=rng
    )

    if doublet_rate:
        doublet_rate = int(len(sim_co_markers) * doublet_rate) if doublet_rate < 1 else int(doublet_rate)
        log.info(f'Simulating {doublet_rate} doublet barcodes')
        sim_co_markers.merge(
            simulate_doublets(co_markers, doublet_rate, rng=rng),
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


def run_sim(marker_json_fn, output_json_fn, ground_truth_fn, *,
            cb_whitelist_fn=None, bin_size=25_000,
            min_markers_per_cb=100, min_markers_per_chrom=20,
            bg_marker_rate=None, bg_window_size=2_500_000,
            nsim_per_sample=100, n_doublets=0.0,
            rng=DEFAULT_RNG):
    """
    Run the full simulation pipeline to create synthetic marker data from ground truth.

    Parameters
    ----------
    marker_json_fn : str
        Path to marker JSON file.
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
    bg_marker_rate : float, optional
        Fixed background rate.
    bg_window_size : int, optional
        Window size for convolution-based background estimation.
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
    if min_markers_per_cb or min_markers_per_chrom:
        co_markers = filter_low_coverage_barcodes(
            co_markers, min_markers_per_cb, min_markers_per_chrom
        )

    if os.path.splitext(ground_truth_fn)[1] == '.bed':
        ground_truth_haplotypes = read_ground_truth_haplotypes_bed(
            ground_truth_fn, co_markers.chrom_sizes, bin_size
        )
    else:
        ground_truth_haplotypes = read_ground_truth_haplotypes_json(ground_truth_fn)
    log.info(f'Read {len(ground_truth_haplotypes)} ground truth samples from {ground_truth_fn}')
    sim_co_markers = generate_simulated_data(
        co_markers,
        ground_truth_haplotypes,
        bg_rate=bg_marker_rate,
        conv_window_size=bg_window_size,
        nsim_per_sample=nsim_per_sample,
        doublet_rate=n_doublets,
        rng=rng
    )
    log.info(f'Simulated {len(sim_co_markers)} barcodes total')
    if output_json_fn is not None:
        log.info(f'Writing markers to {output_json_fn}')
        sim_co_markers.write_json(output_json_fn)
    return sim_co_markers
