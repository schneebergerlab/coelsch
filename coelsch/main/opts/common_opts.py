import click
from .coelsch_opts import coelsch_opts


coelsch_opts.option(
    '-N', '--bin-size',
    subcommands=['loadbam', 'loadcsl', 'bam2pred', 'csl2pred',
                 'sim', 'clean', 'predict',
                 'doublet', 'stats', 'segdist'],
    required=False,
    type=click.IntRange(1000, 5_000_000),
    default=25_000,
    help='Bin size for marker distribution'
)


def _replace_other_with_nonetype(ctx, param, value):
    if value == 'other':
        return None
    return value


coelsch_opts.option(
    '-x', '--seq-type',
    required=False,
    subcommands=['loadbam', 'loadcsl', 'bam2pred', 'csl2pred'],
    type=click.Choice(
        ['10x_rna', '10x_atac', 'bd_rna', 'bd_atac', 'takara_dna', 'wgs', 'other'],
        case_sensitive=False
    ),
    default='other',
    callback=_replace_other_with_nonetype,
    help='presets for different sequencing data, see manual' # todo !!
)


coelsch_opts.option(
    '-y', '--ploidy-type',
    required=False,
    subcommands=['loadbam', 'loadcsl', 'clean', 'predict', 'bam2pred', 'csl2pred'],
    type=click.Choice(
        ['haploid', 'diploid_bc1', 'diploid_f2'],
        case_sensitive=False
    ),
    default=None,
    help='presets for different data ploidy data, instructs what type of model to use'
)
