import logging
from coelsch.load.loadbam.utils import get_ha_samples


def log_parameters(log_name):
    log = logging.getLogger(log_name)
    def _log_parameters(kwargs):
        '''decorator which logs the final values of all parameters used to execute coelsch'''
        for param, value in kwargs.items():
            log.debug(f'set parameter {param} to {value}')
        return kwargs
    return _log_parameters


log = logging.getLogger('coelsch')


def validate_loadbam_input(kwargs):
    '''decorator to validate the input of the loadbam command'''
    seq_type = kwargs.get('seq_type')
    if kwargs.get('ploidy_type') is None:
        # set to haploid for loadbam and loadcsl, other commands infer from data
        kwargs['ploidy_type'] = 'haploid'
    if kwargs.get('cb_correction_method') == 'auto':
        cb_tag = kwargs.get('cb_tag')
        method = 'exact' if cb_tag in ('CB', 'RG') else '1mm'
        log.info(f"setting CB correction method to '{method}' for data with CB tag '{cb_tag}'")
        kwargs['cb_correction_method'] = method 
    if kwargs.get('umi_collapse_method') == 'auto':
        if seq_type in ('10x_rna', 'bd_rna', 'bd_atac'):
            umi_tag = kwargs.get('umi_tag')
            method = 'exact' if umi_tag == 'UB' else 'directional'
            log.info(f"setting UMI dedup method to '{method}' for 10x RNA or BD data with UMI tag '{umi_tag}'")
            kwargs['umi_collapse_method'] = method
        elif seq_type in ('10x_atac', 'takara_dna', 'wgs'):
            log.info('turning off UMI processing for 10x/Takara ATAC data, or WGS data')
            kwargs['umi_tag'] = None
            kwargs['umi_collapse_method'] = None
        elif seq_type is None:
            log.error("'-x' / '--seq-type' must be specified when '--umi-collapse-method' is set to 'auto'")
    elif kwargs.get('umi_collapse_method') == 'none':
        log.info('turning off UMI processing')
        kwargs['umi_tag'] = None
        kwargs['umi_collapse_method'] = None
    if kwargs.get('cb_correction_method') == 'none':
        log.info('turning off CB processing')
        # use exact just without validation of barcodes (they no longer have to be sequences)
        kwargs['cb_correction_method'] = 'exact'
        kwargs['validate_barcodes'] = False
    if kwargs.get('cb_tag') == 'CB' and kwargs.get('cb_correction_method') == '1mm':
        log.warning("'--cb-tag' is set to 'CB', which usually indicates pre-corrected barcodes, but "
                    "'--cb-correction-method' is set to '1mm'. This may lead to overcorrection")
    if kwargs.get('umi_tag') == 'UB' and kwargs.get('umi_collapse_method') == 'directional':
        log.warning("'--umi-tag' is set to 'UB', which usually indicates pre-corrected UMIs, but "
                    "'--cb-correction-method' is set to 'directional'. This may lead to overcorrection")
    if kwargs.get('run_genotype') and kwargs.get('hap_tag_type') == "star_diploid":
        log.error('--hap-tag-type must be "multi_haplotype" when --genotype is switched on')
    if kwargs.get('genotype_recombinant_parental_haplotypes') and kwargs.get('genotype_crossing_combinations'):
        log.error("Provide either --recombinant-parent-jsons or --crossing-combinations, not both.")
    if not kwargs.get('run_genotype') and kwargs.get('hap_tag_type') == "multi_haplotype":
        crossing_combinations = kwargs.get('genotype_crossing_combinations')
        if crossing_combinations is not None:
            if len(crossing_combinations) != 1:
                log.error('when --genotype is switched off only one --crossing-combinations can be provided')
        else:
            genotypes = get_ha_samples(kwargs['bam_fn'])
            if len(genotypes) != 2:
                log.error(
                    'when --genotype is switched off and no --crossing-combinations are provided, '
                    'the bam file can only contain two haplotypes'
                )
    return kwargs


def validate_loadcsl_input(kwargs):
    '''decorator to validate the input of the loadcsl command'''
    if kwargs.get('ploidy_type') is None:
        # set to haploid for loadbam and loadcsl, other commands infer from data
        kwargs['ploidy_type'] = 'haploid'
    if kwargs.get('run_genotype') and kwargs.get('genotype_vcf_fn') is None:
        log.error('--genotype-vcf-fn must be provided when --genotype is switched on')
    if kwargs.get('genotype_recombinant_parental_haplotypes') and kwargs.get('genotype_crossing_combinations'):
        log.error("Provide either --recombinant-parent-jsons or --crossing-combinations, not both.")
    return kwargs


def validate_clean_input(kwargs):
    '''decorator to validate the input of the clean command'''
    if kwargs.get('normalise_depth') is None:
        kwargs['normalise_depth'] = 'auto'
    return kwargs


def validate_pred_input(kwargs):
    '''decorator to validate the input of the predict command'''
    bin_size = kwargs.get('bin_size')
    seg_size = kwargs.get('segment_size')
    tseg_size = kwargs.get('terminal_segment_size')
    if seg_size < bin_size:
        log.error("'-R' / '--segment-size' cannot be less than '-N' / '--bin-size'")
    if tseg_size < bin_size:
        log.error("'-t' / '--terminal-segment-size' cannot be less than '-N' / '--bin-size'")
    return kwargs
