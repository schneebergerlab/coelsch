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
from .utils import get_ha_samples, get_chrom_sizes_bam, chrom_chunks
from ..genotype import GenotypeKey, GenotypesSet, genotype_from_inv_counts, resolve_inv_counts_to_co_markers
from ..utils import genotyping_results_formatter

from coelsch.records import MarkerRecords, NestedData
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


def bam_to_co_markers(bam_fn, processes=1, seq_type=None, ploidy_type='haploid',
                      run_genotype=False, recombinant_mode=False, genotype_kwargs=None, **kwargs):
    """
    Read from a BAM file, identify reads aligning to each haplotype for each cell barcode,
    and summarize the data into a `MarkerRecords` object.

    Parameters
    ----------
    bam_fn : str
        The BAM file path.
    processes : int, optional
        The number of parallel processes to use (default is 1).
    seq_type : str, optional
        The type of sequencing data (e.g., "10x_atac", "10x_rna", "bd_rna", "takara", "wgs").
    ploidy_type : str or None
        A string describing the ploidy type and crossing strategy of the data
        (e.g. "haploid", "diploid_bc1", "diploid_f2").
    run_genotype : bool, optional
        If True, perform genotyping of parental accessions based on interval counts (default is False).
    recombinant_mode : bool, optional
        If True, parental genotypes are themselves recombinants provided in genotype_kwargs (default is False).
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
        seq_type=seq_type,
        ploidy_type=ploidy_type,
    )

    if genotype_kwargs is None:
        genotype_kwargs = {}

    if run_genotype:       
        if kwargs.get('hap_tag_type', 'star_diploid') != "multi_haplotype":
            raise ValueError('must use "multi_haplotype" type hap tag to perform genotyping')
        if recombinant_mode and genotype_kwargs.get('crossing_combinations', None) is not None:
            raise ValueError('Cannot provide crossing combinations when recombinant genotyping mode is switched on')

        all_haplotypes = get_ha_samples(bam_fn)
        (genotypes, genotype_probs,
         genotype_nmarkers, genotype_error_rates,
         inv_counts) = genotype_from_inv_counts(
            inv_counts, recombinant_mode=recombinant_mode,
            all_haplotypes=all_haplotypes, **genotype_kwargs
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
        if genotype_kwargs.get('crossing_combinations', None):
            if len(genotype_kwargs['crossing_combinations']) > 1:
                raise ValueError('When genotyping is switched off, only one crossing_combination can be provided')
            geno = GenotypeKey(list(genotype_kwargs['crossing_combinations'])[0])
        else:
            all_haplotypes = get_ha_samples(bam_fn)
            if len(all_haplotypes) == 2:
                geno = GenotypeKey(all_haplotypes)
            else:
                raise ValueError(
                    'If haplotyping is switched off and crossing_combinations are not provided, the bam file '
                    'can only contain two haplotypes'
                )

        # create a dummy genotypes object where all barcodes have the same genotype
        genotypes = NestedData(
            levels=('cb',),
            dtype=GenotypeKey,
            data={cb: geno for ic in inv_counts for cb in ic.counts},
        )
        # inv_counts are still in haplotype form, need to be resolved using dummy genotype
        inv_counts = resolve_inv_counts_to_co_markers(inv_counts, genotypes, GenotypesSet([geno,]))
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
