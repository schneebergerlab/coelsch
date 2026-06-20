import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

from ..records import MarkerRecords, NestedDataArray


def _normalise_expected_ratio(expected_ratio, n_haplotypes):
    if expected_ratio is None:
        expected_ratio = np.ones(n_haplotypes, dtype=float)
    expected_ratio = np.asarray(expected_ratio, dtype=float)
    if expected_ratio.ndim == 0:
        if n_haplotypes != 2:
            raise ValueError('scalar expected_ratio is only valid for two haplotypes')
        expected_ratio = np.array([expected_ratio, 1.0 - expected_ratio], dtype=float)
    if expected_ratio.shape[0] != n_haplotypes:
        raise ValueError(
            f'expected_ratio has length {expected_ratio.shape[0]}, '
            f'but MarkerRecords has {n_haplotypes} channels'
        )
    total = expected_ratio.sum()
    if total <= 0:
        raise ValueError('expected_ratio must have positive sum')
    return expected_ratio / total


def create_single_cell_haplotype_imbalance_mask(co_markers, max_imbalance_mask=0.75, min_cb=20,
                                                expected_ratio=None, apply_per_geno=True):
    """
    Create a mask for bins with high haplotype imbalance.

    Parameters
    ----------
    co_markers : MarkerRecords
        Marker data with haplotype-specific read counts.
    max_imbalance_mask : float, default=0.75
        Maximum allowed absolute deviation around the expected haplotype ratios.
    min_cb : int, default=20
        Minimum number of cell barcodes required per bin.
    expected_ratio : array-like, optional
        Expected haplotype ratio aligned to marker columns. Defaults to 1:1.
    apply_per_geno : bool, default=True
        Mask separately per genotype.

    Returns
    -------
    NestedDataArray
        metadata array of masks by genotype and chromosome.
    int
        Total number of bins masked.
    """
    n_haplotypes = co_markers.n_haplotypes
    expected_ratio = _normalise_expected_ratio(expected_ratio, n_haplotypes)
    tolerance = max_imbalance_mask - 0.5
    if tolerance < 0:
        raise ValueError('max_imbalance_mask must be at least 0.5')

    imbalance_mask = NestedDataArray(levels=('genotype', 'chrom'))
    n_masked_all_genos = []
    for geno, geno_co_markers in co_markers.groupby(by='genotype' if apply_per_geno else 'none'):
        tot_signal = {}
        tot_obs = {}
        for _, chrom, m in geno_co_markers.deep_items():
            if chrom not in tot_signal:
                tot_signal[chrom] = m.copy()
                tot_obs[chrom] = np.minimum(m.sum(axis=1), 1)
            else:
                tot_signal[chrom] += m
                tot_obs[chrom] += np.minimum(m.sum(axis=1), 1)
        imbalance_mask[geno] = {}
        n_masked = 0
        for chrom, m in tot_signal.items():
            bin_sum = m.sum(axis=1, keepdims=True)
            with np.errstate(invalid='ignore', divide='ignore'):
                ratio = m / bin_sum
            ratio = np.where(bin_sum > 0, ratio, expected_ratio[None, :])
            ratio_mask = np.any(np.abs(ratio - expected_ratio[None, :]) > tolerance, axis=1)
            count_mask = tot_obs[chrom] >= min_cb
            mask = np.logical_and(ratio_mask, count_mask)
            n_masked += mask.sum(axis=None)
            imbalance_mask[geno, chrom] = np.repeat(mask[:, None], n_haplotypes, axis=1)
        n_masked_all_genos.append(n_masked)
    co_markers.add_metadata(haplotype_imbalance_mask=imbalance_mask)
    return imbalance_mask, int(np.median(n_masked_all_genos))

def median_absolute_deviation(arr):
    """
    Compute the median absolute deviation (MAD) of an array.

    Parameters
    ----------
    arr : array_like
        Input array.

    Returns
    -------
    float
        Median of the absolute deviations from the median.
    """
    return np.median(np.abs(arr - np.median(arr)))


def create_resequencing_haplotype_imbalance_mask(co_markers, expected_ratio='auto',
                                                 nmad_mask=5, correction=1e-2,
                                                 apply_per_geno=True):
    """
    Special haplotype imbalance method for resequencing data (not scRNA)
    that identifies bins with extreme haplotype composition outliers.

    Parameters
    ----------
    co_markers : MarkerRecords
        Object containing per-cell, per-chromosome haplotype marker counts.
    expected_ratio : array-like or 'auto', optional
        Expected haplotype ratio. If 'auto', it is estimated from the data as the median ratio.
    nmad_mask : int, optional
        Number of median absolute deviations (MADs) to use for outlier detection.
    correction : float, optional
        Small value added to numerator and denominator to avoid division by zero.
    apply_per_geno : bool, default=True
        Mask separately per genotype.

    Returns
    -------
    dict of str to np.ndarray
        Dictionary mapping chromosome names to boolean masks of shape (bins, haplotypes),
        where True indicates a bin to exclude due to outlier haplotype composition.
    """
    n_haplotypes = co_markers.n_haplotypes
    imbalance_mask = NestedDataArray(levels=('genotype', 'chrom'))
    n_masked_all_genos = []
    for geno, geno_co_markers in co_markers.groupby(by='genotype' if apply_per_geno else 'none'):
        n_masked = 0
        chrom_ratios = {}
        chrom_marker_masks = {}
        for chrom in geno_co_markers.chrom_sizes:
            m = geno_co_markers[:, chrom].stack_values()
            cb_totals = m.sum(axis=(1, 2))[:, np.newaxis, np.newaxis]
            with np.errstate(invalid='ignore', divide='ignore'):
                m_norm = np.where(cb_totals > 0, m / cb_totals, 0.0).sum(axis=0)
            tot = m_norm.sum(axis=1, keepdims=True)
            chrom_marker_masks[chrom] = tot[:, 0] > 0
            chrom_ratios[chrom] = (m_norm + correction) / (tot + correction * n_haplotypes)

        valid_ratios = np.concatenate([
            ratios[chrom_marker_masks[chrom]]
            for chrom, ratios in chrom_ratios.items()
            if np.any(chrom_marker_masks[chrom])
        ])
        if len(valid_ratios) == 0:
            expected = np.ones(n_haplotypes, dtype=float) / n_haplotypes
            mad = np.ones(n_haplotypes, dtype=float)
        else:
            if isinstance(expected_ratio, str) and expected_ratio == 'auto':
                expected = np.median(valid_ratios, axis=0)
                expected /= expected.sum()
            else:
                expected = _normalise_expected_ratio(expected_ratio, n_haplotypes)
            mad = np.array([
                median_absolute_deviation(valid_ratios[:, i] - expected[i])
                for i in range(n_haplotypes)
            ])
            mad = np.maximum(mad, correction)

        for chrom, ratios in chrom_ratios.items():
            ratios[~chrom_marker_masks[chrom]] = expected
            mask = binary_dilation(np.any(
                np.abs(ratios - expected[None, :]) > (mad[None, :] * nmad_mask),
                axis=1,
            ))
            n_masked += mask.sum(axis=None)
            imbalance_mask[geno, chrom] = np.repeat(mask[:, None], n_haplotypes, axis=1)
        n_masked_all_genos.append(n_masked)
    co_markers.add_metadata(haplotype_imbalance_mask=imbalance_mask)
    return imbalance_mask, int(np.median(n_masked_all_genos))


def apply_haplotype_imbalance_mask(co_markers, mask, apply_per_geno=True):
    """
    Apply mask to bins with excessive haplotype imbalance.

    Parameters
    ----------
    co_markers : MarkerRecords
        Marker data to mask.
    mask : NestedDataArray
        Mask to apply to data.
    apply_per_geno : bool, default=True
        Apply masking per genotype.

    Returns
    -------
    MarkerRecords
        Masked marker records.
    int
        Number of bins masked.
    """
    co_markers_m = MarkerRecords.new_like(co_markers)
    if apply_per_geno:
        genotypes = co_markers.metadata['genotypes']
    for cb, chrom, m in co_markers.deep_items():
        geno = genotypes[cb] if apply_per_geno else 'ungrouped'
        co_markers_m[cb, chrom] = np.where(mask[geno, chrom], 0, m)
    return co_markers_m


def apply_marker_threshold(co_markers, max_marker_threshold):
    """
    Cap bin counts to a maximum marker threshold.

    Parameters
    ----------
    co_markers : MarkerRecords
        Marker data.
    max_marker_threshold : int
        Maximum allowed marker count per bin.

    Returns
    -------
    MarkerRecords
        Thresholded marker records.
    """
    co_markers_t = MarkerRecords.new_like(co_markers)
    for cb, chrom, m in co_markers.deep_items():
        co_markers_t[cb, chrom] = np.minimum(m, max_marker_threshold)
    return co_markers_t


def mask_regions_bed(co_markers, mask_bed_fn, inplace=False):
    """
    Mask regions listed in a BED file.

    Parameters
    ----------
    co_markers : MarkerRecords
        Marker data to be masked.
    mask_bed_fn : str
        Path to BED file with regions to mask.
    inplace: bool
        Whether to perform masking inplace

    Returns
    -------
    MarkerRecords
        Masked marker records.
    """
    if not inplace:
        co_markers = co_markers.copy()
    mask_invs = pd.read_csv(
        mask_bed_fn,
        sep='\t',
        usecols=[0, 1, 2],
        names=['chrom', 'start', 'end'],
        dtype={'chrom': str, 'start': int, 'end': int}
    )
    bs = co_markers.bin_size
    for chrom, invs in mask_invs.groupby('chrom'):
        start_bins = np.floor(invs.start / bs).astype(int)
        end_bins = np.ceil(invs.end / bs).astype(int)
        
        for m in co_markers[:, chrom].values():
            for s, e in zip(start_bins, end_bins):
                m[s: e] = 0
    return co_markers
