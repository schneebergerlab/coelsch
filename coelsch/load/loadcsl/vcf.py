import pysam


def get_vcf_samples(vcf_fn, ref_name):
    """
    Extract the list of sample names from a VCF file and add the reference name.

    Parameters
    ----------
    vcf_fn : str
        Path to the VCF file.
    ref_name : str
        The name of sample used as the reference genome to add to the list of samples.

    Returns
    -------
    frozenset
        A frozenset containing the sample names (including the reference sample).
    """
    samples = [ref_name,]
    with pysam.VariantFile(vcf_fn) as vcf:
        samples += sorted(vcf.header.samples)
    return tuple(samples)


def parse_sample_alleles(variant, ref_name):
    """
    Parse the alleles for each sample in a variant record and assign them to reference and alternate groups.

    Parameters
    ----------
    variant : pysam.VariantRecord
        The VCF variant record to process.
    ref_name : str
        The reference name for the variant.

    Returns
    -------
    tuple
        A tuple of two frozensets:
        - The first frozenset contains the samples that have the reference allele.
        - The second frozenset contains the samples that have the alternate allele.

    Raises
    ------
    ValueError
        If the variant is not biallelic or if any sample has more than one allele.
    """
    if len(variant.alleles) != 2:
        raise ValueError('Only biallelic variants are allowed')
    ref_samples = set()
    ref_samples.add(ref_name)
    alt_samples = set()
    for sample_id, sample in variant.samples.items():
        if len(sample.alleles) > 1:
            raise ValueError('Only haploid haplotype calls are allowed for each sample')
        if not sample.allele_indices[0]:
            ref_samples.add(sample_id)
        else:
            alt_samples.add(sample_id)
    return frozenset(ref_samples), frozenset(alt_samples)


class VariantRecords:

    """
    A class to store variant records and associated sample allele information.

    Methods
    ----------
    add(contig, pos, sample_alleles=None)
        Add a variant record with its sample alleles to the VariantRecords object.
    get_samples(contig, pos)
        Get the sample alleles for a specific variant.
    """

    def __init__(self):
        """
        Initialize an empty VariantRecords object.
        """
        self._records = []
        self._samples = {}

    def add(self, contig, pos, sample_alleles=None):
        """
        Add a variant with its sample alleles to the VariantRecords object.

        Parameters
        ----------
        contig : str
            The name of the contig (chromosome).
        pos : int
            The position of the variant on the contig.
        sample_alleles : tuple, optional
            A tuple containing two frozensets: one for the reference allele samples
            and one for the alternate allele samples. Default is None.
        """
        self._records.append((contig, pos))
        self._samples[(contig, pos)] = sample_alleles

    def __getitem__(self, index):
        return self._records[index]

    def get_samples(self, contig, pos):
        """
        Get the samples with ref and alt alleles for a specific variant.

        Parameters
        ----------
        contig : str
            The name of the contig (chromosome).
        pos : int
            The position of the variant.

        Returns
        -------
        tuple
            A tuple containing two frozensets: one for the reference allele samples
            and one for the alternate allele samples.
        """
        return self._samples[(contig, pos)]


def read_vcf(vcf_fn, drop_samples=True, reference_name='col0'):
    """
    Read a VCF file and parse the variants into VariantRecords.

    Parameters
    ----------
    vcf_fn : str
        Path to the VCF file to read.
    drop_samples : bool, optional
        Whether to exclude sample data (default is True).
    reference_name : str, optional
        The name of the reference sample to include in the records (default is 'col0').

    Returns
    -------
    VariantRecords
        A VariantRecords object containing the parsed variant data.
    """

    # cellsnp-lite vcf files have some missing definitions in the header
    # setting pysam verbosity to 0 prevents warnings
    prev_verbosity = pysam.set_verbosity(0)

    with pysam.VariantFile(vcf_fn, drop_samples=drop_samples) as vcf:
        variants = VariantRecords()
        for record in vcf.fetch():
            if drop_samples:
                sample_alleles = None
            else:
                try:
                    sample_alleles = parse_sample_alleles(record, reference_name)
                except ValueError:
                    continue
            variants.add(record.contig, record.pos, sample_alleles)

    pysam.set_verbosity(prev_verbosity)

    return variants
