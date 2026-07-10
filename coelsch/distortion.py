import logging
import itertools as it
import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed

from .utils import load_json


log = logging.getLogger('coelsch')
LOD = 2 * np.log(10)


def segregation_distortion_chroms(chrom_haps, order, bin_size):
    """
    Evaluate segregation distortion across a set of chromosomes at a given resolution.

    Parameters
    ----------
    chrom_haps : dict of np.ndarray
        Dictionary mapping chromosome names to (haplotypes x bins) arrays of predicted haplotype
        probabilities or binary calls.
    order : int
        Number of loci to test jointly (e.g., 1 for single-locus, 2 for pairwise).
    bin_size : int
        Size of genomic bins in base pairs (used to calculate positions).

    Returns
    -------
    pandas.DataFrame
        Table of distortion results containing:
        - chromosome names and bin positions
        - LOD score (log-likelihood ratio)
        - p-value of chi-squared test
    """
    res = []
    for positions in it.product(*(np.arange(c.shape[1]) for c in chrom_haps.values())):
        _, ct = stats.contingency.crosstab(*[
            (ch[:, p] > 0.5)
            for ch, p in zip(chrom_haps.values(), positions)
        ])
        if order == 1:
            exp = ct.sum() // 2
            chi2, pval, *_ = stats.chi2_contingency([ct, [exp, exp]], lambda_='log-likelihood')
        else:
            chi2, pval, *_ = stats.chi2_contingency(ct, lambda_='log-likelihood')
        res.append([*chrom_haps.keys(), *(p * bin_size for p in positions), chi2 / LOD, pval])
    res = pd.DataFrame(res, columns=[
        *(f'chrom_{i}' for i in range(1, order + 1)),
        *(f'pos_{i}' for i in range(1, order + 1)),
        'lod_score', 'pval'
    ])
    return res


def downsample_chrom(chrom_co_preds, bin_size, resolution):
    """
    Downsample high-resolution marker predictions to a coarser resolution.

    Parameters
    ----------
    chrom_co_preds : np.ndarray
        Haplotype matrix of shape (barcodes, bins) for a single chromosome.
    bin_size : int
        Original resolution of the binning.
    resolution : int
        Desired resolution to downsample to. Must be a multiple of `bin_size`.

    Returns
    -------
    np.ndarray
        Downsampled (barcodes, new_bins) matrix.
    """
    assert resolution >= bin_size and not resolution % bin_size
    cs = int(resolution / bin_size)
    splits = np.arange(cs, chrom_co_preds.shape[1], cs)
    return np.stack([m.mean(axis=1) for m in np.split(chrom_co_preds, splits, axis=1)], axis=1)


def segregation_distortion(co_preds,
                           order=1,
                           resolution=250_000,
                           processes=1):
    """
    Perform genome-wide scan for segregation distortion.

    Parameters
    ----------
    co_preds : MarkerRecords
        Predictions object with crossover/haplotype probabilities per chromosome.
    order : int, default=1
        Number of loci to test jointly. 1 for single-locus, 2+ for multi-locus distortion.
    resolution : int, default=250000
        Bin resolution to downsample to before distortion testing.
    processes : int, default=1
        Number of parallel processes to use.

    Returns
    -------
    pandas.DataFrame
        Results of segregation distortion tests, including:
        - chromosomal coordinates
        - LOD scores
        - adjusted p-values after FDR correction
    """
    co_preds_low_res = {
        c: downsample_chrom(co_preds[:, c].stack_values(),
                            co_preds.bin_size,
                            resolution)
        for c in co_preds.chrom_sizes
    }
    with Parallel(n_jobs=processes) as pool:
        res = pool(
            delayed(segregation_distortion_chroms)(
                {c: co_preds_low_res[c] for c in chrom_perm}, order=order, bin_size=resolution
            ) for chrom_perm in it.combinations(co_preds.chrom_sizes, r=order)
        )
    res = pd.concat(res).reset_index(drop=True)
    res['pval'] = stats.false_discovery_control(res.pval)
    return res


def run_segdist(pred_json_fn, output_tsv_fn, *,
                cb_whitelist_fn=None, bin_size=25_000,
                segdist_order=2, downsample_resolution=250_000,
                output_precision=3, processes=1):
    """
    Compute segregation distortion and write results to TSV.

    Parameters
    ----------
    pred_json_fn : str
        Path to input JSON file with predicted haplotype/crossover data.
    output_tsv_fn : str
        Path to write the distortion results as a tab-separated file.
    cb_whitelist_fn : str, optional
        Path to a whitelist of barcodes to include.
    bin_size : int, default=25000
        Resolution of input bins in base pairs.
    segdist_order : int, default=2
        Number of loci to jointly test for distortion.
    downsample_resolution : int, default=250000
        Resolution to downsample the prediction matrix to before testing.
    output_precision : int, default=3
        Number of significant digits to write to output file.
    processes : int, default=1
        Number of processes to use for parallel computation.

    Returns
    -------
    pandas.DataFrame
        Table of segregation distortion results with LOD scores and p-values.
    """
    co_preds = load_json(
        pred_json_fn, cb_whitelist_fn, bin_size, data_type='predictions'
    )
    log.info('Running segregation distortion analysis')
    segdist_results = segregation_distortion(
        co_preds,
        order=segdist_order,
        resolution=downsample_resolution,
        processes=processes
    )
    if output_tsv_fn is not None:
        log.info(f'Writing segregation distortion results to {output_tsv_fn}')
        segdist_results.to_csv(
            output_tsv_fn, sep='\t', index=False, float_format=f'%.{output_precision}g'
        )
    return segdist_results