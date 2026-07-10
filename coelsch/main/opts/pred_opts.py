import logging
import click
from .coelsch_opts import coelsch_opts

log = logging.getLogger('coelsch')


coelsch_opts.option(
    '-R', '--segment-size',
    subcommands=['predict', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.IntRange(100_000, 25_000_000),
    default=1_000_000,
    help='rfactor of the rigid HMM. Approximately controls minimum distance between COs'
)


coelsch_opts.option(
    '-t', '--terminal-segment-size',
    subcommands=['predict', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.IntRange(10_000, 5_000_000),
    default=250_000,
    help=('terminal rfactor of the rigid HMM. approx. controls min distance of COs '
          'from chromosome ends')
)


coelsch_opts.option(
    '-C', '--cm-per-mb',
    subcommands=['predict', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.FloatRange(1e-3, 20.0),
    default=4.5,
    help=('Approximate average centiMorgans per megabase. '
          'Used to parameterise the rigid HMM transitions')
)


coelsch_opts.option(
    '-h', '--interference-half-life',
    subcommands=['predict', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.FloatRange(0, 10_000_000),
    metavar='INTEGER OR FLOAT',
    default=5_000,
    help=('Distance below segment_size at which recombination rate halves (exponentially) '
          'to enforce crossover-interference. If <1, this is considered a fraction of '
          'segment_size, else it is an absolute value in bp')
)


coelsch_opts.option(
    '--distribution-type',
    subcommands=['predict', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.Choice(['poisson', 'nb']),
    default='poisson',
    help=('type of underlying distribution used to model the count data. Can be either poisson or nb')
)


coelsch_opts.option(
    '--sample-co-locs/--no-sample-cos', 'sample_paths',
    subcommands=['predict', 'bam2pred', 'csl2pred'],
    required=False,
    default=False,
    help='Whether to sample stochastic crossover locations for each barcode'
)


coelsch_opts.option(
    '--n-samples',
    subcommands=['predict', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.IntRange(1, 999),
    default=25,
    metavar='INTEGER OR FLOAT',
    help=('The number of stochastic samples to take for each barcode')
)


coelsch_opts.option(
    '--predict-doublets/--no-predict-doublets',
    subcommands=['predict', 'bam2pred', 'csl2pred'],
    required=False,
    default=True,
    help='whether to use synthetic doublet scoring to predict likely doublets in the data'
)


coelsch_opts.option(
    '--n-doublets',
    subcommands=['sim', 'predict', 'doublet', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.FloatRange(0.0, 10_000),
    default=1000,
    metavar='INTEGER OR FLOAT',
    help=('number of doublets to simulate. If >1, treated as integer number of doublets. '
          'If <1, treated as a fraction of the total dataset')
)


coelsch_opts.option(
    '--generate-stats/--no-stats',
    subcommands=['predict', 'doublet', 'bam2pred', 'csl2pred'],
    required=False,
    default=True,
    help='whether to generate a stats tsv file for the output predictions'
)


coelsch_opts.option(
    '-M', '--nco-min-prob-change',
    subcommands=['stats', 'predict', 'plot', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.FloatRange(0.0, 1.0),
    default=2.5e-3,
    help=('Minimum probability change to take into account when estimating stats e.g. number of '
          'crossovers from the haplotype predictions.')
)



def _check_device(ctx, param, value):
    import torch
    if value == 'cpu':
        return torch.device(value)
    try:
        device_type, device_number = value.split(':')
        device_number = int(device_number)
    except ValueError as exc:
        log.error(
            'device format should be "cpu" or device:number e.g. "cuda:0"'
        )
    if device_type not in {'cuda', 'mps'}:
        log.error(f'unrecognised device type {device_type}')

    n_devices = getattr(torch, device_type).device_count()
    if not n_devices:
        log.error(f'no devices available of type {device_type}')
    elif (device_number + 1) > n_devices:
        log.error(
            f'device number is too high, only {n_devices} device of type {device_type} avaiable'
        )
    return torch.device(value)


coelsch_opts.option(
    '-d', '--device',
    subcommands=['predict', 'doublet', 'bam2pred', 'csl2pred'],
    required=False,
    type=str,
    default='cpu',
    callback=_check_device,
    help='device to compute predictions on (default cpu)'
)

coelsch_opts.option(
    '--batch-size',
    subcommands=['predict', 'doublet', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.IntRange(1, 10_000),
    default=128,
    help='batch size for prediction. larger may be faster but use more memory'
)
