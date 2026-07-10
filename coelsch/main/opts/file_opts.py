import click
from .coelsch_opts import coelsch_opts


_input_file_type = click.Path(exists=True, file_okay=True, dir_okay=False)
_input_dir_type = click.Path(exists=True, file_okay=False, dir_okay=True)
_output_file_type = click.Path(exists=False)
_output_dir_type = click.Path(exists=False, file_okay=False, dir_okay=True)


coelsch_opts.argument(
    'bam-fn',
    subcommands=['loadbam', 'bam2pred'],
    required=True,
    nargs=1,
    type=_input_file_type,
)


coelsch_opts.argument(
    'cellsnp-lite-dir',
    subcommands=['loadcsl', 'csl2pred'],
    required=True,
    nargs=1,
    type=_input_dir_type,
)


coelsch_opts.argument(
    'marker-json-fn',
    subcommands=['sim', 'clean', 'predict', 'doublet', 'stats', 'plot'],
    required=True,
    nargs=1,
    type=_input_file_type,
)


coelsch_opts.argument(
    'json-fn',
    subcommands=['concat'],
    required=True,
    nargs=-1,
    type=_input_file_type,
)


coelsch_opts.argument(
    'pred-json-fn',
    subcommands=['doublet', 'stats', 'segdist'],
    required=True,
    nargs=1,
    type=_input_file_type,
)


coelsch_opts.argument(
    'pred-json-fn',
    subcommands=['plot'],
    required=False,
    nargs=1,
    type=_input_file_type,
)


coelsch_opts.argument(
    'cell-barcode',
    subcommands=['plot'],
    required=False,
    type=str,
    nargs=1,
)


coelsch_opts.option(
    '-o', '--output-prefix',
    subcommands=['bam2pred', 'csl2pred'],
    required=True,
    type=_output_file_type,
    help='Output prefix'
)


coelsch_opts.option(
    '-o', '--output-json-fn',
    subcommands=['loadbam', 'loadcsl', 'sim', 'concat', 'clean', 'predict', 'doublet'],
    required=True,
    type=_output_file_type,
    help='Output JSON file name.'
)


coelsch_opts.option(
    '-o', '--output-tsv-fn',
    subcommands=['stats', 'segdist'],
    required=True,
    type=_output_file_type,
    help='Output TSV file name.'
)


coelsch_opts.option(
    '-o', '--output-fig-fn',
    subcommands=['plot'],
    required=False,
    type=_output_file_type,
    default=None,
    help='Output figure file name (filetype automatically determined)'
)


coelsch_opts.option(
    '-c', '--cb-whitelist-fn',
    subcommands=['loadbam', 'loadcsl', 'bam2pred', 'csl2pred',
                 'sim', 'clean', 'predict', 'doublet', 'stats',
                 'plot', 'segdist'],
    required=False,
    type=_input_file_type,
    help='Text file containing whitelisted cell barcodes, one per line'
)


coelsch_opts.option(
    '-s', '--chrom-sizes-fn',
    subcommands=['loadcsl', 'csl2pred'],
    required=True,
    type=_input_file_type,
    help='chrom sizes or faidx file'
)


coelsch_opts.option(
    '-g', '--genotype-vcf-fn',
    subcommands=['loadcsl', 'csl2pred'],
    required=False,
    type=_input_file_type,
    help='VCF file containing all the parental genotype SNPs'
)


coelsch_opts.option(
    '-m', '--mask-bed-fn',
    subcommands=['clean', 'bam2pred', 'csl2pred',],
    required=False,
    type=_input_file_type,
    default=None,
    help='A bed file of regions to mask when cleaning data'
)


coelsch_opts.option(
    '-g', '--ground-truth-fn',
    subcommands=['sim'],
    required=True,
    type=_input_file_type,
    help='pred json or bed file (6 column) containing ground truth haplotype intervals to simulate'
)
