import numpy as np
import pysam
from coelsch.defaults import DEFAULT_EXCLUDE_CONTIGS


def get_chrom_sizes_bam(bam_fn, exclude_contigs=None):
    """
    Load a dictionary of chromosome lengths from the header of a BAM file.

    Parameters
    ----------
    bam_fn : str
        The BAM file path.
    exclude_contigs : list of str, optional
        List of contig names to exclude (default is `DEFAULT_EXCLUDE_CONTIGS`).

    Returns
    -------
    dict
        A dictionary where keys are chromosome names and values are their respective lengths.
    """
    if exclude_contigs is None:
        exclude_contigs = DEFAULT_EXCLUDE_CONTIGS
    with pysam.AlignmentFile(bam_fn) as bam:
        chrom_sizes = {k: bam.get_reference_length(k)
                       for k in bam.references
                       if k not in exclude_contigs}
    return chrom_sizes


def chrom_chunks(chrom_sizes, bin_size, processes):
    """
    Generate chunks of chromosome bins for parallel processing.

    Parameters
    ----------
    chrom_sizes : dict
        A dictionary where keys are chromosome names and values are their respective lengths.
    bin_size : int
        The bin size in base pairs.
    processes : int
        The number of processes used. processes * 10 chunks are generated

    Yields
    ------
    tuple
        Tuples of `(chrom, start, end)` where `chrom` is the chromosome name and
        `start` and `end` are the bin indices defining a chunk.
    """
    tot = 0
    chrom_nbins = {}
    for chrom, cs in chrom_sizes.items():
        nbins = int(np.ceil(cs / bin_size))
        tot += nbins
        chrom_nbins[chrom] = nbins

    chunk_size = max(tot // (processes * 10), 1)
    for chrom, nbins in chrom_nbins.items():
        start = 0
        while start < nbins:
            end = min(start + chunk_size, nbins)
            yield chrom, start, end
            start = end