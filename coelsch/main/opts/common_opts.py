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


coelsch_opts.option(
    '-x', '--seq-type',
    required=False,
    subcommands=['loadbam', 'loadcsl', 'bam2pred', 'csl2pred'],
    type=click.Choice(
        ['10x_rna', '10x_atac', 'bd_rna', 'bd_atac', 'takara_dna', 'wgs', 'other'],
        case_sensitive=False
    ),
    default='other',
    help='presets for different sequencing data, see manual' # todo !!
)


coelsch_opts.option(
    '--lifecycle-stage',
    required=False,
    subcommands=['loadbam', 'loadcsl', 'bam2pred', 'csl2pred'],
    type=click.Choice(['gametes', 'progeny'], case_sensitive=False),
    default='gametes',
    help='sample lifecycle stage used to interpret the experimental design'
)


coelsch_opts.option(
    '--crossing-strategy',
    required=False,
    subcommands=['loadbam', 'loadcsl', 'bam2pred', 'csl2pred'],
    type=click.Choice(
        ['f1', 'f2', 'backcross', 'testcross', 'three_way', 'four_way'],
        case_sensitive=False
    ),
    default='f1',
    help='crossing strategy used to interpret the experimental design'
)


coelsch_opts.option(
    '-y', '--ploidy-type',
    required=False,
    subcommands=['clean', 'predict', 'bam2pred', 'csl2pred'],
    type=click.Choice(
        ['haploid', 'diploid_bc1', 'diploid_f2'],
        case_sensitive=False
    ),
    default=None,
    help='presets for different data ploidy data, instructs what type of model to use'
)
