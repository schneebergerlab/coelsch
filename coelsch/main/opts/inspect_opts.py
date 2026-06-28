from .coelsch_opts import coelsch_opts


coelsch_opts.option(
    '--genotypes/--no-genotypes',
    subcommands=['inspect'],
    default=True,
    help='Print genotyping summary.'
)

coelsch_opts.option(
    '--metadata/--no-metadata',
    subcommands=['inspect'],
    default=False,
    help='Print metadata slot summary.'
)


coelsch_opts.option(
    '--barcodes/--no-barcodes',
    subcommands=['inspect'],
    default=False,
    help='Prints all cell barcodes/sample names as plain text'
)


coelsch_opts.option(
    '-q', '--query',
    subcommands=['inspect'],
    default=None,
    type=str,
    help='Filter records with a BaseRecords query string before inspection.'
)
