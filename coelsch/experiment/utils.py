import pysam


def get_all_haplotypes_bam(bam_fn):
    """
    Retrieves haplotype samples from a BAM file.

    This function attempts to extract haplotype sample information from the header of the BAM file
    in multiple ways (using the 'ha' field in the HD header, or extracting from the 'ha_flag_accessions' 
    comment or RG fields).

    Parameters
    ----------
    bam_fn : str
        Path to the BAM file.

    Returns
    -------
    list of str
        A sorted list of haplotype sample identifiers.

    Raises
    ------
    ValueError
        If no haplotype information is found in the BAM file header.
    """
    with pysam.AlignmentFile(bam_fn) as bam:
        try:
            return set(bam.header['HD']['ha'].split(','))
        except KeyError:
            for comment in bam.header['CO']:
                if comment.startswith('ha_flag_accessions'):
                    samples = set(comment.split(' ')[1].split(','))
                    break
            else:
                # final attempt, to use RGs
                try:
                    samples = sorted(rg['ID'] for rg in bam.header['RG'])
                except KeyError:
                    raise ValueError('Could not find ha_flag accession information in header')
    return sorted(samples)


def get_all_haplotypes_vcf(vcf_fn, ref_name):
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
