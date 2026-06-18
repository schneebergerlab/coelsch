'''
functions for aggregating marker information from a bam file containing cell barcode, haplotype 
and optionally UMI tags, into the coelsch.MarkerRecords format.
'''
import logging
from functools import reduce
from collections import defaultdict
import itertools as it
from joblib import Parallel, delayed

import numpy as np
import pysam

from .bam import BAMHaplotypeIntervalReader
from .utils import get_chrom_sizes_bam, chrom_chunks
from ..genotyping import genotype_from_inv_counts, resolve_inv_counts_to_co_markers
from ..utils import genotyping_results_formatter

from coelsch.records import MarkerRecords, NestedData
from coelsch.experiment import GenotypeKey
from coelsch.clean.filter import filter_low_coverage_barcodes, filter_genotyping_score
from coelsch.defaults import DEFAULT_RANDOM_SEED, DEFAULT_EXCLUDE_CONTIGS


log = logging.getLogger('coelsch')
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


def _get_interval_co_markers(bam_fn, chrom, bin_start, bin_end, **kwargs):
    chrom_inv_counts = []
    with BAMHaplotypeIntervalReader(bam_fn, **kwargs) as bam:
        for bin_idx in range(bin_start, bin_end):
            chrom_inv_counts.append(bam.fetch_interval_counts(chrom, bin_idx))
    return chrom_inv_counts


def bam_to_co_markers(bam_fn, experimental_design, processes=1,
                      run_genotype=False, genotype_kwargs=None, **kwargs):
    """
    Read from a BAM file, identify reads aligning to each haplotype for each cell barcode,
    and summarize the data into a `MarkerRecords` object.

    Parameters
    ----------
    bam_fn : str
        The BAM file path.
    experimental_design : coelsch.experiment.ExperimentalDesign
        The experimental design/parameters
    processes : int, optional
        The number of parallel processes to use (default is 1).
    run_genotype : bool, optional
        If True, perform genotyping of parental accessions based on interval counts (default is False).
    genotype_kwargs : dict, optional
        Additional arguments passed to the genotyping function (default is None).
    kwargs : dict
        Additional arguments passed to the BAM file processing functions.

    Returns
    -------
    MarkerRecords
        A `MarkerRecords` object containing aggregated haplotype marker information for the cell barcodes.
    """
    chrom_sizes = get_chrom_sizes_bam(bam_fn, exclude_contigs=kwargs.get('exclude_contigs', None))
    bin_size = kwargs.get('bin_size')

    log.debug(f'Starting job pool to process bam with {processes} processes')
    with Parallel(n_jobs=processes, backend='loky') as pool:
        inv_counts = pool(
            delayed(_get_interval_co_markers)(bam_fn, chrom, start, end, **kwargs)
            for chrom, start, end in chrom_chunks(chrom_sizes, bin_size, processes)
        )
        inv_counts = list(it.chain(*inv_counts))

    # create empty MarkerRecords object to be filled with inv_counts (post genotyping)
    co_markers = MarkerRecords(
        chrom_sizes,
        bin_size,
        experiment_params=experimental_design.experiment_params
    )

    if genotype_kwargs is None:
        genotype_kwargs = {}

    if run_genotype:       
        if kwargs.get('hap_tag_type', 'star_diploid') != "multi_haplotype":
            raise ValueError('must use "multi_haplotype" type hap tag to perform genotyping')

        (genotypes, genotype_probs,
         genotype_nmarkers, genotype_error_rates,
         inv_counts) = genotype_from_inv_counts(
            inv_counts, experimental_design, **genotype_kwargs
        )
        if log.isEnabledFor(logging.DEBUG):
            log.debug(genotyping_results_formatter(genotypes))
        co_markers.add_metadata(
            genotypes=genotypes,
            genotype_probability=genotype_probs,
            genotyping_nmarkers=genotype_nmarkers,
            genotype_error_rates=genotype_error_rates
        )

    elif kwargs.get('hap_tag_type', 'star_diploid') == "multi_haplotype":
        # genotyping is switched off but we can infer the genotype of barcodes from the haplotype tags
        if len(experimental_design.genotypes) != 1:
            raise ValueError(
                'If genotyping is switched off, only one crossing_combination or '
                'haplotype combination is allowed'
            )
        geno = experimental_design.genotypes[0]

        # create a dummy genotypes object where all barcodes have the same genotype
        genotypes = NestedData(
            levels=('cb',),
            dtype=GenotypeKey,
            data={cb: geno for ic in inv_counts for cb in ic.counts},
        )
        # inv_counts are still in haplotype form, need to be resolved using dummy genotype
        inv_counts = resolve_inv_counts_to_co_markers(inv_counts, genotypes, experimental_design)
        co_markers.add_metadata(
            genotypes=NestedData(
                levels=('cb',),
                dtype=str,
                data={cb: str(geno) for ic in inv_counts for cb in ic.counts}
            )
        )
    for ic in inv_counts:
        co_markers.update(ic)
    return co_markers
