'''classes for reading bam files and aggregating count information'''
import logging
import numpy as np
import pysam

from .haplotypes import MultiHaplotypeValidator
from .utils import get_ha_samples
from ..counts import IntervalUMICounts
from coelsch.defaults import DEFAULT_EXCLUDE_CONTIGS


log = logging.getLogger('coelsch')


def length_normalised_alignment_score(aln):
    read_length = aln.query_length
    if not read_length:
        read_length = aln.infer_query_length()
    if aln.is_paired:
        read_length *= 2
    try:
        alignment_score = aln.get_tag('AS')
    except KeyError:
        log.warn('Some reads do not have an AS tag and cannot be filtered by alignment score')
        return 1.0
    return alignment_score / read_length
        

class BAMHaplotypeIntervalReader:
    """
    File wrapper class for reading aggregated bin counts for each cell barcode/haplotype.

    This class facilitates the reading of a BAM file, specifically extracting and aggregating
    count information for each interval (bin) within the genome for a given cell barcode (CB)
    and haplotype.

    Attributes
    ----------
    bin_size : int
        The size of each bin used for aggregation (default: 25,000).
    cb_tag : str
        The tag used to store the cell barcode (default: 'CB').
    umi_tag : str
        The tag used to store the UMI (default: 'UB').
    hap_tag : str
        The tag used to store the haplotype (default: 'ha').
    hap_tag_type : str
        The haplotype tag type ('star_diploid' or 'multi_haplotype').
    haplotypes : frozenset
        A set of allowed haplotypes.
    cb_whitelist : object
        An optional whitelist for cell barcodes.
    umi_collapse_method : str
        The method for collapsing UMIs ('directional', None, etc.).
    exclude_contigs : set
        Set of contig names to exclude (default: mitochondrial, plastid).

    Methods
    -------
    fetch_interval_counts(chrom, bin_idx)
        Aggregates alignments into an IntervalMarkerCounts object for the specified bin and chromosome.
    close()
        Closes the BAM file.
    """

    def __init__(self, bam_fn, *,
                 bin_size=25_000,
                 cb_tag='CB',
                 umi_tag='UB',
                 hap_tag='ha',
                 hap_tag_type='star_diploid',
                 allowed_haplotypes=None,
                 cb_whitelist=None,
                 min_alignment_score=None,
                 min_mapq=None,
                 umi_collapse_method='directional',
                 exclude_contigs=None):
        """
        File wrapper class for reading aggregated bin counts for each cell barcode/haplotype.
    
        This class facilitates the reading of a BAM file, specifically extracting and aggregating
        count information for each interval (bin) within the genome for a given cell barcode (CB)
        and haplotype.

        Parameters
        ----------
        bam_fn : str
            Path to the BAM file.
        bin_size : int, optional
            The size of each bin used for aggregation (default is 25,000).
        cb_tag : str, optional
            The tag used for the cell barcode (default is 'CB').
        umi_tag : str, optional
            The tag used for the UMI (default is 'UB').
        hap_tag : str, optional
            The tag used for the haplotype (default is 'ha').
        hap_tag_type : str, optional
            The haplotype tag type ('star_diploid' or 'multi_haplotype', default is 'star_diploid').
        allowed_haplotypes : list of str, optional
            List of allowed haplotypes. If None, the haplotypes are extracted from the BAM header.
        cb_whitelist : coelsch.barcodes.CellBarcodeWhitelist, optional
            A whitelist for cell barcodes, supports 1MM fuzzy matching.
        umi_collapse_method : str, optional
            The method for collapsing UMIs ('directional', 'exact', None).
        exclude_contigs : set, optional
            A set of contig names to exclude from processing (default is mitochondrial and plastid contigs).
        """
        self._bam_fn = bam_fn
        self.bin_size = bin_size
        self.cb_tag = cb_tag
        self.umi_tag = umi_tag if umi_collapse_method is not None else None
        self.hap_tag = hap_tag
        self.hap_tag_type = hap_tag_type
        self.haplotypes = MultiHaplotypeValidator(allowed_haplotypes)
        self.cb_whitelist = cb_whitelist

        if min_alignment_score is not None:
            self.alnscore_filter = lambda aln: length_normalised_alignment_score(aln) < min_alignment_score
        else:
            self.alnscore_filter = lambda aln: False

        if min_mapq is not None:
            self.mapq_filter = lambda aln: aln.mapq < min_mapq
        else:
            self.mapq_filter = lambda aln: False

        self.umi_collapse_method = umi_collapse_method
        if exclude_contigs is None:
            exclude_contigs = DEFAULT_EXCLUDE_CONTIGS
        self.exclude_contigs = exclude_contigs
        self._open()

    def _open(self):
        self.bam = pysam.AlignmentFile(self._bam_fn)
        self.is_closed = False

        self.chrom_sizes = {k: self.bam.get_reference_length(k)
                            for k in self.bam.references
                            if k not in self.exclude_contigs}
        self.nbins = {}
        for chrom, cs in self.chrom_sizes.items():
            self.nbins[chrom] = int(np.ceil(cs / self.bin_size))
        if self.hap_tag_type == 'multi_haplotype' and self.haplotypes == None:
            self.haplotypes = MultiHaplotypeValidator(get_ha_samples(self._bam_fn))

    def fetch_interval_counts(self, chrom, bin_idx):
        """
        Aggregates alignments into an IntervalMarkerCounts object for the specified bin and chromosome.

        This function fetches alignments from the BAM file for a given chromosome and bin index,
        and aggregates the alignments into an IntervalMarkerCounts object based on the cell barcode,
        UMI, and haplotype.

        Parameters
        ----------
        chrom : str
            The chromosome name.
        bin_idx : int
            The index of the bin within the chromosome.

        Returns
        -------
        IntervalMarkerCounts
            An object containing the aggregated counts for the specified bin.

        Raises
        ------
        IOError
            If the BAM records do not contain the expected tags (e.g., CB, UB, ha).
        ValueError
            If the haplotype information in the BAM file does not match the expected format.
        """
        assert bin_idx < self.nbins[chrom]
        bin_start = self.bin_size * bin_idx
        bin_end = bin_start + self.bin_size - 1

        interval_counts = IntervalUMICounts(
            chrom, bin_idx, self.umi_collapse_method,
            self.hap_tag_type, self.haplotypes,
        )

        for aln in self.bam.fetch(chrom, bin_start, bin_end):

            # only keep primary, unduplicated alignments, as multimappers may cause artefacts
            if aln.is_secondary or aln.is_supplementary or aln.is_duplicate:
                continue

            # only consider fwd mapping of properly paired 2xreads, prevents double counting
            if aln.is_paired:
                if aln.is_reverse or not aln.is_proper_pair:
                    continue

            # remove alignments with low AS or MAPQ
            if self.alnscore_filter(aln) or self.mapq_filter(aln):
                continue

            try:
                hap = aln.get_tag(self.hap_tag)
            except KeyError as exc:
                raise IOError(
                    f'bam records do not all have the haplotype tag "{self.hap_tag}"'
                ) from exc

            if self.hap_tag_type == 'star_diploid':
                hap -= 1
                if hap < 0:
                    continue
            elif self.hap_tag_type == 'multi_haplotype':
                hap = self.haplotypes[str(hap).split(',')]
                if self.haplotypes == hap:
                    # read does not align better to one haplotype than others,
                    # so is not useful for haplotyping analysis
                    continue
                if not hap:
                    # read aligns to haplotypes that are not allowed by validator
                    log.warn('saw alignment with no allowed haplotypes, this may or may not be expected')
                    continue
            else:
                raise ValueError(f'hap_tag_type "{self.hap_tag_type}" not recognised')

            # only consider reads where the left mapping position is within the bin,
            # to prevent duplicates in adjacent bins
            if aln.reference_start < bin_start:
                continue

            try:
                cb = aln.get_tag(self.cb_tag)
                if cb == '-':
                    cb = None
            except KeyError as exc:
                raise IOError(
                    f'bam records do not all have the cell barcode tag "{self.cb_tag}"'
                ) from exc

            # finally filter for barcodes in the whitelist
            # use 1mm correction to match cbs with no more than 1 mismatch to the whitelist
            if self.cb_whitelist is not None:
                cb = self.cb_whitelist.correct(cb)

            if cb is not None:
                if self.umi_tag is not None:
                    try:
                        umi = aln.get_tag(self.umi_tag)
                    except KeyError as exc:
                        raise IOError(
                            f'bam records do not all have the UMI tag "{self.umi_tag}". '
                            'Maybe you meant to run without UMI deduplication?'
                        ) from exc
                else:
                    umi = None
                interval_counts[cb, umi, hap] += 1

        return interval_counts.collapse()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def close(self):
        self.bam.close()
