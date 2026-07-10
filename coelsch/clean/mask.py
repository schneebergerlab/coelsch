import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

from ..records import MarkerRecords, NestedDataArray


def create_single_cell_haplotype_imbalance_mask(co_markers, max_imbalance_mask=0.75, min_cb=20,
                                                ploidy_type='haploid', apply_per_geno=True):
    """
    Create a mask for bins with high haplotype imbalance.

    Parameters
    ----------
    co_markers : MarkerRecords
        Marker data with haplotype-specific read counts.
    max_imbalance_mask : float, default=0.75
        Maximum allowed haplotype ratio before masking.
    min_cb : int, default=20
        Minimum number of cell barcodes required per bin.
    ploidy_type: str, default="haploid"
        The ploidy type of the data - one of "haploid", "diploid_bc1" or "diploid"
    apply_per_geno : bool, default=True
        Mask separately per genotype.
    

    Returns
    -------
    NestedDataArray
        metadata array of masks by genotype and chromosome.
    int
        Total number of bins masked.
    """
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
            with np.errstate(invalid='ignore'):
                bin_sum = m.sum(axis=1)
                ratio = m[:, 0] / bin_sum
            # map ratios to 0.5 for backcross data
            if ploidy_type == "diploid_bc1":
                ratio /= (2 / 3)
            np.nan_to_num(ratio, nan=0.5, copy=False)
            ratio_mask = (ratio > max_imbalance_mask) | (ratio < (1 - max_imbalance_mask))
            count_mask = tot_obs[chrom] >= min_cb
            mask = np.logical_and(ratio_mask, count_mask)
            n_masked += mask.sum(axis=None)
            imbalance_mask[geno, chrom] = np.stack([mask, mask], axis=1)
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
                                                 nmad_mask=3, correction=1e-2,
                                                 apply_per_geno=True):
    """
    Special haplotype imbalance method for resequencing data (not scRNA)
    that identifies bins with extreme allele imbalance.

    Parameters
    ----------
    co_markers : MarkerRecords
        Object containing per-cell, per-chromosome haplotype marker counts.
    expected_ratio : float or 'auto', optional
        Expected allele ratio. If 'auto', it is estimated from the data as the median ratio.
    nmad_mask : int, optional
        Number of median absolute deviations (MADs) to use for outlier detection.
    correction : float, optional
        Small value added to numerator and denominator to avoid division by zero.
    apply_per_geno : bool, default=True
        Mask separately per genotype.

    Returns
    -------
    dict of str to np.ndarray
        Dictionary mapping chromosome names to boolean masks of shape (bins,),
        where True indicates a bin to exclude due to outlier allele ratio.
    """
    imbalance_mask = NestedDataArray(levels=('genotype', 'chrom'))
    n_masked_all_genos = []
    for geno, geno_co_markers in co_markers.groupby(by='genotype' if apply_per_geno else 'none'):
        n_masked = 0
        chrom_allele_ratios = {}
        chrom_marker_masks = {}
        for chrom in geno_co_markers.chrom_sizes:
            m = geno_co_markers[:, chrom].stack_values()
            m_norm = (m / m.sum(axis=(1, 2))[:, np.newaxis, np.newaxis]).sum(axis=0)
            tot = m_norm.sum(axis=1)
            chrom_marker_masks[chrom] = tot > 0
            chrom_allele_ratios[chrom] = (m_norm[:, 0] + correction) / (tot + correction)

        ar = np.concatenate([ar[chrom_marker_masks[chrom]] for chrom, ar in chrom_allele_ratios.items()])
        if expected_ratio == 'auto':
            expected_ratio = np.median(ar)
        mad = median_absolute_deviation(ar)

        for chrom, allele_ratio in chrom_allele_ratios.items():
            allele_ratio[~chrom_marker_masks[chrom]] = expected_ratio
            mask = binary_dilation(np.logical_or(
                allele_ratio < (expected_ratio - mad * nmad_mask),
                allele_ratio > (expected_ratio + mad * nmad_mask),
            ))
            n_masked += mask.sum(axis=None)
            imbalance_mask[geno, chrom] = np.stack([mask, mask], axis=1)
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
