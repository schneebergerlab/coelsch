import click
from .coelsch_opts import coelsch_opts
from .file_opts import _input_file_type
from coelsch.defaults import DEFAULT_EXCLUDE_CONTIGS


coelsch_opts.option(
    '--cb-correction-method',
    subcommands=['loadbam', 'bam2pred'],
    required=False,
    type=click.Choice(['exact', '1mm', 'none', 'auto'], case_sensitive=False),
    default='auto',
    help='method for correcting/matching cell barcodes to whitelist'
)


def _validate_bam_tag(ctx, param, value):
    if not len(value) == 2:
        raise ValueError(f'{param} is not of length 2')
    return value


coelsch_opts.option(
    '--cb-tag',
    subcommands=['loadbam', 'bam2pred'],
    required=False,
    type=str,
    default='CB',
    callback=_validate_bam_tag,
    help='bam file tag representing cell barcode'
)


coelsch_opts.option(
    '--umi-collapse-method',
    subcommands=['loadbam', 'bam2pred'],
    is_eager=True,
    required=False,
    type=click.Choice(['exact', 'directional', 'none', 'auto'], case_sensitive=False),
    default='auto',
    help='Method for deduplicating/collapsing UMIs'
)


coelsch_opts.option(
    '--umi-tag',
    subcommands=['loadbam', 'bam2pred'],
    required=False,
    type=str,
    default='UB',
    help='bam file tag representing UMI'
)


coelsch_opts.option(
    '--hap-tag',
    subcommands=['loadbam', 'bam2pred'],
    required=False,
    type=str,
    default='ha',
    callback=_validate_bam_tag,
    help='bam file tag representing haplotype.'
)


coelsch_opts.option(
    '--hap-tag-type',
    subcommands=['loadbam', 'bam2pred'],
    required=False,
    type=click.Choice(['star_diploid', 'multi_haplotype']),
    default='star_diploid',
    help='how the haplotype tag is encoded, see manual for details' # todo!
)


coelsch_opts.option(
    '--min-alignment-score',
    subcommands=['loadbam', 'bam2pred'],
    required=False,
    type=click.FloatRange(0, 1),
    default=0.95,
    help='only reads with a length normalised alignment score greater than this value are used'
)


coelsch_opts.option(
    '--genotype/--no-genotype', 'run_genotype',
    subcommands=['loadbam', 'loadcsl', 'bam2pred', 'csl2pred'],
    required=False,
    default=False,
    help='whether to use EM algorithm to infer genotypes (requires --hap-tag-type="multi_haplotype")',
)


def _parse_crossing_combinations(ctx, param, value):
    """
    Parse --crossing-combinations like "A:B,A:C,B:C" into a list of (hap1, hap2) tuples.
    Raises on duplicates and reciprocal-equivalent pairs (e.g., "A:B" and "B:A").
    """
    if not value:
        return None

    tokens = [t.strip() for t in value.split(',') if t.strip()]

    combos = []
    seen_unordered = set()

    for t in tokens:
        haps = [h.strip() for h in t.split(':')]
        if len(haps) != 2 or not haps[0] or not haps[1]:
            raise click.BadParameter(
                f"Invalid crossing combination '{t}'. Use 'hap1:hap2'.",
                ctx=ctx, param=param
            )
        haps = tuple(haps)
        haps_unordered = frozenset(haps)
        if haps_unordered in seen_unordered:
            raise click.BadParameter(
                f"Duplicate or reciprocal-equivalent crossing combination '{t}'. "
                "Supply each pair only once.",
                ctx=ctx, param=param
            )
        combos.append(haps)
        seen_unordered.add(haps_unordered)

    return combos


coelsch_opts.option(
    '-X', '--crossing-combinations', 'genotype_crossing_combinations',
    subcommands=['loadbam', 'loadcsl', 'bam2pred', 'csl2pred'],
    required=False,
    default=None,
    callback=_parse_crossing_combinations,
    help=('comma separated list of allowed combinations of parental haplotypes used in crosses, '
          'encoded in format "hap1:hap2,hap1:hap3" etc. Incompatible with recombinant mode')
)


coelsch_opts.option(
    '--recombinant-parent-jsons', 'genotype_recombinant_parental_haplotypes',
    subcommands=['loadbam', 'loadcsl', 'bam2pred', 'csl2pred'],
    nargs=2, type=click.Tuple([_input_file_type, _input_file_type]),
    required=False,
    default=None,
    help=('This option switches on recombinant genotyping mode. Two pred jsons must be provided, that '
          'encode the two recombinant haplotypes of each parental genotype. Barcodes from the input '
          'are then matched to these recombinant genotypes.')
)


coelsch_opts.option(
    '--genotype-em-max-iter',
    subcommands=['loadbam', 'loadcsl', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.IntRange(1, 10_000),
    default=1000,
    help='the maximum number of iterations to run the genotyping EM algorithm for'
)

coelsch_opts.option(
    '--genotype-em-min-delta',
    subcommands=['loadbam', 'loadcsl', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.FloatRange(0, 0.1),
    default=1e-3,
    help='the minimum difference in genotype probability between EM iterations, before stopping'
)


coelsch_opts.option(
    '--genotype-em-bootstraps',
    subcommands=['loadbam', 'loadcsl', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.IntRange(1, 1000),
    default=25,
    help='the number of bootstrap resamples to use for estimating genotype probabilities'
)

coelsch_opts.option(
    '--reference-geno-name', 'reference_genotype_name',
    subcommands=['loadcsl', 'csl2pred'],
    required=False,
    default='col0',
    help='name of the reference genotype (alt genotype names are taken from vcf samples)',
)


coelsch_opts.option(
    '--validate-barcodes/--no-validate',
    subcommands=['loadbam', 'loadcsl', 'bam2pred', 'csl2pred'],
    required=False,
    default=True,
    help='whether to check that cell barcodes are valid sequences',
)


coelsch_opts.option(
    '--count-snps-only/--count-cov-per-snp', 'snp_counts_only',
    subcommands=['loadcsl', 'csl2pred'],
    required=False,
    default=False,
    help='whether to record snp coverage or just consensus genotype of snp',
)


def _parse_excl_contigs(ctx, param, value):
    if value is None:
        value = DEFAULT_EXCLUDE_CONTIGS
    else:
        value = set(value.split(',')) # todo: check allowed fasta header names
    return value


coelsch_opts.option(
    '-e', '--exclude-contigs',
    subcommands=['loadbam', 'bam2pred'],
    required=False,
    type=str,
    default=None,
    callback=_parse_excl_contigs,
    help=('comma separated list of contigs to exclude. '
          'Default is a set of common organellar chrom names')
)
