import logging
import numpy as np

from .loadbam import bam_to_co_markers
from .loadcsl import cellsnp_lite_to_co_markers
from .barcodes import read_cb_whitelist
from coelsch.clean.filter import filter_low_coverage_barcodes, filter_genotyping_score
from coelsch.defaults import DEFAULT_RANDOM_SEED


log = logging.getLogger('coelsch')
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


def _post_load_filtering(co_markers, min_markers_per_cb, min_markers_per_chrom,
                         min_geno_prob, max_geno_error_rate):
    '''
    Post load filtering funcs, common to both loadbam and loadcsl
    '''
    n = len(co_markers)
    log.info(f'Identified {n} cell barcodes')
    if min_markers_per_cb or min_markers_per_chrom:
        co_markers = filter_low_coverage_barcodes(
            co_markers, min_markers_per_cb, min_markers_per_chrom, with_copy=False
        )
        log.info(
            f'Removed {n - len(co_markers)} barcodes with fewer than {min_markers_per_cb} markers '
            f'or fewer than {min_markers_per_chrom} markers per chromosome'
        )
    n = len(co_markers)
    if min_geno_prob:
        co_markers = filter_genotyping_score(co_markers, min_geno_prob, max_geno_error_rate, with_copy=False)
        log.info(
            f'Removed {n - len(co_markers)} barcodes with genotyping probability < {min_geno_prob}'
        )
    return co_markers


def run_loadbam(bam_fn, output_json_fn, *,
                cb_whitelist_fn=None, bin_size=25_000,
                seq_type=None, ploidy_type='haploid',
                cb_tag='CB', cb_correction_method='exact',
                umi_tag='UB', umi_collapse_method='directional',
                hap_tag='ha', hap_tag_type='star_diploid',
                min_alignment_score=0.95, min_mapq=None,
                run_genotype=False, genotype_crossing_combinations=None,
                genotype_recombinant_parental_haplotypes=None,
                genotype_em_max_iter=1000, genotype_em_min_delta=1e-3,
                genotype_em_bootstraps=25, validate_barcodes=True,
                min_markers_per_cb=100, min_markers_per_chrom=20,
                min_geno_prob=0.9, max_geno_error_rate=0.25,
                exclude_contigs=None, processes=1, rng=DEFAULT_RNG):
    """
    Read a BAM file with cell barcode, UMI, and haplotype tags, and generate a JSON file 
    containing binned haplotype marker distributions for each cell barcode.

    Parameters
    ----------
    bam_fn : str
        The BAM file path.
    output_json_fn : str
        The output JSON file path to store the processed marker records.
    cb_whitelist_fn : str, optional
        Path to a cell barcode whitelist file (default is None).
    bin_size : int, optional
        The size of each bin in base pairs (default is 25,000).
    seq_type : str, optional
        The type of sequencing data (default is None).
    ploidy_type : str or None
        A string describing the ploidy type and crossing strategy of the data
        (e.g. "haploid", "diploid_bc1", "diploid_f2", default is "haploid").
    cb_tag : str, optional
        The tag used to identify cell barcodes in the BAM file (default is 'CB').
    cb_correction_method : str, optional
        Method for correcting cell barcode tags (default is 'exact').
    umi_tag : str, optional
        The tag used to identify UMIs in the BAM file (default is 'UB').
    umi_collapse_method : str, optional
        Method for collapsing UMIs (default is 'directional').
    hap_tag : str, optional
        The tag used to identify haplotypes in the BAM file (default is 'ha').
    hap_tag_type : str, optional
        The haplotype tag type (default is 'star_diploid').
    run_genotype : bool, optional
        If True, perform genotyping (default is False).
    genotype_crossing_combinations : list of frozenset, optional
        List of allowed crossing combinations for genotyping (default is None).
    genotype_recombinant_parental_haplotypes : tuple or None, optional
        Switched on recombinant mode. A tuple of length 2 containing the paths to the two 
        PredictionRecords json objects, which encode the recombination patterns of the two 
        haplotypes of the parental genotypes
    genotype_em_max_iter : int, optional
        Maximum number of EM iterations for genotyping (default is 1000).
    genotype_em_min_delta : float, optional
        Minimum change in probabilities for convergence (default is 1e-3).
    genotype_em_bootstraps : int, optional
        Number of bootstrap re-samples for genotyping (default is 25).
    min_markers_per_cb : int, optional
        Minimum number of markers required per barcode (default is 100).
    min_markers_per_chrom : int, optional
        Minimum number of markers required per chromosome (default is 20).
    min_geno_prob : float, optional
        Minimum genotyping probability, barcodes with lower probabilities are filtered (default is 0.9).
    max_geno_error_rate : float, optional
        Maximum genotyping background noise rate allowed for each barcode to be included (default is 0.25).
    exclude_contigs : list of str, optional
        List of contig names to exclude (default is None).
    processes : int, optional
        Number of parallel processes to use (default is 1).
    rng : np.random.Generator, optional
        Random number generator (default is `DEFAULT_RNG`).

    Returns
    -------
    MarkerRecords
        A `MarkerRecords` object containing processed haplotype marker distributions.
    """
    cb_whitelist = read_cb_whitelist(
        cb_whitelist_fn,
        validate_barcodes=validate_barcodes,
        cb_correction_method=cb_correction_method
    )
    co_markers = bam_to_co_markers(
        bam_fn, processes=processes,
        bin_size=bin_size,
        seq_type=seq_type,
        ploidy_type=ploidy_type,
        cb_tag=cb_tag,
        umi_tag=umi_tag,
        umi_collapse_method=umi_collapse_method,
        hap_tag=hap_tag,
        hap_tag_type=hap_tag_type,
        run_genotype=run_genotype,
        recombinant_mode=genotype_recombinant_parental_haplotypes is not None,
        genotype_kwargs={
            'crossing_combinations': genotype_crossing_combinations,
            'recombinant_parental_haplotypes': genotype_recombinant_parental_haplotypes,
            'max_iter': genotype_em_max_iter,
            'min_delta': genotype_em_min_delta,
            'n_bootstraps': genotype_em_bootstraps,
            'min_markers_per_cb': min_markers_per_cb,
            'processes': processes,
            'rng': rng
        },
        cb_whitelist=cb_whitelist,
        min_alignment_score=min_alignment_score,
        min_mapq=min_mapq,
        exclude_contigs=exclude_contigs,
    )
    co_markers = _post_load_filtering(
        co_markers,
        min_markers_per_cb,
        min_markers_per_chrom,
        min_geno_prob,
        max_geno_error_rate,
    )
    if output_json_fn is not None:
        log.info(f'Writing markers to {output_json_fn}')
        co_markers.write_json(output_json_fn)
    return co_markers


def run_loadcsl(cellsnp_lite_dir, chrom_sizes_fn, output_json_fn, *,
                cb_whitelist_fn=None, bin_size=25_000, snp_counts_only=False,
                seq_type=None, ploidy_type='haploid',
                run_genotype=False, genotype_vcf_fn=None,
                genotype_crossing_combinations=None,
                genotype_recombinant_parental_haplotypes=None,
                reference_genotype_name='col0',
                genotype_em_max_iter=1000, genotype_em_min_delta=1e-3, genotype_em_bootstraps=25,
                min_markers_per_cb=100, min_markers_per_chrom=20,
                min_geno_prob=0.9, max_geno_error_rate=0.25,
                validate_barcodes=True, processes=1, rng=DEFAULT_RNG):
    """
    Reads matrix files generated by cellSNP-lite to generate a JSON file of binned
    haplotype marker distributions for each cell barcode. These can be used to
    call recombinations using the downstream `predict` command.

    Parameters
    ----------
    cellsnp_lite_dir : str
        Directory containing the cellSNP-lite output files.
    chrom_sizes_fn : str
        Path to a file containing chromosome sizes.
    output_json_fn : str
        Path to the output JSON file where the marker data will be saved.
    cb_whitelist_fn : str, optional
        Path to a file containing a list of cell barcodes to be used (default is None).
    bin_size : int, optional
        The bin size for partitioning the genome into intervals (default is 25,000).
    seq_type : str, optional
        The type of sequencing data (default is None).
    ploidy_type : str or None
        A string describing the ploidy type and crossing strategy of the data
        (e.g. "haploid", "diploid_bc1", "diploid_f2", default is None).
    snp_counts_only : bool, optional
        Whether to only count SNPs, instead of the number of reads per SNP (default is False).
    run_genotype : bool, optional
        If True, performs genotyping using the provided VCF files (default is False).
    genotype_vcf_fn : str, optional
        Path to the VCF file for genotyping. Required if `run_genotype` is True.
    genotype_crossing_combinations : list, optional
        List of genotype crossing combinations for the genotyping process (default is None).
    genotype_recombinant_parental_haplotypes : tuple or None, optional
        Switched on recombinant mode. A tuple of length 2 containing the paths to the two 
        PredictionRecords json objects, which encode the recombination patterns of the two 
        haplotypes of the parental genotypes
    reference_genotype_name : str, optional
        The name of the reference sample in the VCF file (default is 'col0').
    genotype_em_max_iter : int, optional
        Maximum number of iterations for the EM algorithm during genotyping (default is 1000).
    genotype_em_min_delta : float, optional
        Minimum change in probability for EM convergence (default is 1e-3).
    genotype_em_bootstraps : int, optional
        Number of bootstrap samples for genotyping (default is 25).
    min_markers_per_cb : int, optional
        Minimum number of markers required for each cell barcode to be included (default is 100).
    min_markers_per_chrom : int, optional
        Minimum number of markers required for each chromosome of a barcode to be included (default is 20).
    min_geno_prob : float, optional
        Minimum genotyping probability required for each barcode to be included (default is 0.9).
    max_geno_error_rate : float, optional
        Maximum genotyping background noise rate allowed for each barcode to be included (default is 0.25).
    validate_barcodes : bool, optional
        If True, validates barcodes against the whitelist (default is True).
    processes : int, optional
        Number of processes to use for parallel processing - only used for genotyping (default is 1).
    rng : numpy.random.Generator, optional
        Random number generator instance for reproducibility (default is the global default RNG).

    Returns
    -------
    MarkerRecords
        A MarkerRecords object containing haplotype marker distributions for each cell barcode.
    """

    cb_whitelist = read_cb_whitelist(cb_whitelist_fn, validate_barcodes=validate_barcodes)
    co_markers = cellsnp_lite_to_co_markers(
        cellsnp_lite_dir,
        chrom_sizes_fn,
        bin_size=bin_size,
        cb_whitelist=cb_whitelist,
        seq_type=seq_type,
        ploidy_type=ploidy_type,
        validate_barcodes=validate_barcodes,
        snp_counts_only=snp_counts_only,
        run_genotype=run_genotype,
        recombinant_mode=genotype_recombinant_parental_haplotypes is not None,
        genotype_vcf_fn=genotype_vcf_fn,
        reference_name=reference_genotype_name,
        genotype_kwargs={
            'crossing_combinations': genotype_crossing_combinations,
            'recombinant_parental_haplotypes': genotype_recombinant_parental_haplotypes,
            'max_iter': genotype_em_max_iter,
            'min_delta': genotype_em_min_delta,
            'n_bootstraps': genotype_em_bootstraps,
            'min_markers_per_cb': min_markers_per_cb,
            'processes': processes,
            'rng': rng
        },
    )
    co_markers = _post_load_filtering(
        co_markers,
        min_markers_per_cb,
        min_markers_per_chrom,
        min_geno_prob,
        max_geno_error_rate,
    )
    if output_json_fn is not None:
        log.info(f'Writing markers to {output_json_fn}')
        co_markers.write_json(output_json_fn)
    return co_markers