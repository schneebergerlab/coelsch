import logging

import numpy as np

from .background import estimate_overall_background_signal, clean_marker_background
from .filter import filter_low_coverage_barcodes, filter_genotyping_score
from .normalise import normalise_bin_coverage, normalise_barcode_depth
from .mask import (
    create_single_cell_haplotype_imbalance_mask,
    create_resequencing_haplotype_imbalance_mask,
    apply_haplotype_imbalance_mask,
    apply_marker_threshold, mask_regions_bed
)
from ..utils import load_json, validate_ploidy
from coelsch.defaults import DEFAULT_RANDOM_SEED


log = logging.getLogger('coelsch')
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


def run_clean(marker_json_fn, output_json_fn, *,
              co_markers=None, cb_whitelist_fn=None, mask_bed_fn=None, bin_size=25_000,
              ploidy_type=None, min_markers_per_cb=0, min_markers_per_chrom=0,
              normalise_bins=True, bin_shrinkage_quantile=0.99,
              normalise_depth=True, max_bin_count=20,
              clean_bg=True, bg_window_size=2_500_000, max_frac_bg=0.2,
              min_geno_prob=0.9, max_geno_error_rate=0.25,
              mask_imbalanced=True, max_marker_imbalance=0.75, apply_per_geno=True,
              rng=DEFAULT_RNG):
    """
    Complete pipeline for cleaning single-cell marker data by removing ambient background,
    extremely imbalanced bins and poorly supported barcodes.

    Parameters
    ----------
    marker_json_fn : str
        Input JSON file with marker data.
    output_json_fn : str
        Path to write cleaned marker data.
    co_markers : MarkerRecords, optional
        Pre-loaded marker records.
    cb_whitelist_fn : str, optional
        File path to barcode whitelist.
    mask_bed_fn : str, optional
        BED file of regions to mask.
    bin_size : int, default=25000
        Bin size for aggregation in base pairs.
    ploidy_type : str, default=None
        The ploidy type of the data, can be "haploid", "diploid_bc1" or "diploid_f2"
    min_markers_per_cb : int, default=0
        Minimum total markers required for a barcode.
    min_markers_per_chrom : int, default=0
        Minimum markers per chromosome per barcode.
    normalise_bins : bool, default=True
        Whether to normalise the coverage of bins to account for marker density/expression variation.
    normalise_depth : bool, default=True
        Whether to normalise the coverage of barcodes to make them approximately equivalent.
    bin_shrinkage_quantile : float, default=0.99
        The quantile used when computing the shrinkage parameter for bin normalisation.
    max_bin_count : int, default=20
        Maximum marker count allowed per bin.
    clean_bg : bool, default=True
        Whether to subtract estimated background signal.
    bg_window_size : int, default=2500000
        Size of convolution window for background estimation.
    max_frac_bg : float, default=0.2
        Maximum tolerated background contamination.
    min_geno_prob : float, default=0.9
        Minimum genotype confidence required.
    max_geno_error_rate : float, optional
        Maximum genotyping background noise rate allowed for each barcode to be included (default is 0.25).
    mask_imbalanced : bool, default=True
        Whether to mask bins with haplotype imbalance.
    max_marker_imbalance : float, default=0.75
        Maximum tolerated haplotype ratio before masking.
    apply_per_geno : bool, default=True
        Apply all corrections separately per genotype.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    MarkerRecords
        Cleaned marker records.
    """
    if co_markers is None:
        co_markers = load_json(marker_json_fn, cb_whitelist_fn, bin_size)
    ploidy_type = validate_ploidy(co_markers, ploidy_type)

    n = len(co_markers)
    if min_markers_per_cb:
        co_markers = filter_low_coverage_barcodes(
            co_markers, min_markers_per_cb, min_markers_per_chrom, with_copy=False
        )
        log.info(
            f'Removed {n - len(co_markers)} barcodes with fewer than {min_markers_per_cb} markers '
            f'or fewer than {min_markers_per_chrom} markers per chromosome'
        )
    n = len(co_markers)
    if 'genotypes' in co_markers.metadata:
        co_markers = filter_genotyping_score(co_markers, min_geno_prob, max_geno_error_rate, with_copy=False)
        log.info(
            f'Removed {n - len(co_markers)} barcodes with genotyping probability < {min_geno_prob}'
        )

    if mask_imbalanced:
        # mask any bins that still have extreme imbalance
        # (e.g. due to extreme allele-specific expression differences)
        if co_markers.seq_type != "wgs":
            log.info(
                f'Masking marker imbalances with single-cell method'
            )
            mask, n_masked = create_single_cell_haplotype_imbalance_mask(
                co_markers, max_marker_imbalance,
                apply_per_geno=apply_per_geno,
                ploidy_type=ploidy_type,
            )
        else:
            log.info(
                f'Masking marker imbalances with whole-genome resequencing method'
            )
            # special masking method for wgs data which has much greater coverage
            mask, n_masked = create_resequencing_haplotype_imbalance_mask(
                co_markers, apply_per_geno=apply_per_geno # maybe should expose params?
            )
        co_markers = apply_haplotype_imbalance_mask(
            co_markers, mask, apply_per_geno=apply_per_geno
        )
        tot_bins = sum(co_markers.nbins.values())
        log.info(
            f'Masked {n_masked:d}/{tot_bins} bins with extreme marker imbalance'
        )

    if normalise_bins:
        log.info(
            f'Normalising bin coverage to reduce marker or expression biases'
        )
        co_markers = normalise_bin_coverage(
            co_markers, shrinkage_q=bin_shrinkage_quantile, correct_hap_bias=False,
            allow_upweight=True if co_markers.seq_type == "wgs" else False,
            binwise_hap_mode='independent' if co_markers.seq_type in ('10x_rna', 'bd_rna') else 'shared'
        )

    n = len(co_markers)
    # currently this method does not make sense for non-haploid samples
    if ploidy_type == 'haploid':
        # estimate ambient marker rate for each CB and try to scrub common background markers
        log.info('Estimating background marker rates.')
        co_markers = estimate_overall_background_signal(
            co_markers, bg_window_size, max_frac_bg, apply_per_geno=apply_per_geno
        )
        log.info(
            f'Removed {n - len(co_markers)} barcodes with greater than {max_frac_bg * 100}%'
            ' background contamination'
        )
        av_bg = np.mean(
            list(co_markers.metadata['estimated_background_fraction'].values())
        )
        log.info(f'Average estimated background fraction is {av_bg:.3f}')
        if clean_bg:
            log.info('Attempting to filter likely background markers')
            co_markers = clean_marker_background(
                co_markers, apply_per_geno=apply_per_geno
            )

    if normalise_depth:
        log.info(
            f'Normalising total depth to remove sequencing depth differences'
        )
        co_markers = normalise_barcode_depth(co_markers)

    n = len(co_markers)
    co_markers = filter_low_coverage_barcodes(
        co_markers, min_cov=0, min_cov_per_chrom=1
    )
    log.info(
        f'Removed {n - len(co_markers)} barcodes with at least '
        'one chromosome without markers after cleaning'
    )

    # threshold bins that have a large number of reads
    if max_bin_count is not None:
        co_markers = apply_marker_threshold(co_markers, max_bin_count)
        log.info(f'Thresholded bins with >{max_bin_count} markers')

    if mask_bed_fn is not None:
        co_markers = mask_regions_bed(co_markers, mask_bed_fn, inplace=True)
        log.info(f'Masked regions blacklisted in {mask_bed_fn}')

    if output_json_fn is not None:
        log.info(f'Writing markers to {output_json_fn}')
        co_markers.write_json(output_json_fn)
    return co_markers
