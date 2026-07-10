from logging import getLogger
from collections import Counter
import click

from coelsch.utils import load_json

log = getLogger('coelsch')


def _format_value(value):
    if isinstance(value, (list, tuple)):
        return ", ".join(map(str, value))
    return str(value)


def _print_experiment_params(experiment_params):
    log.info("experiment_params:")
    if experiment_params is None:
        log.info("  missing")
        return

    fields = (
        "lifecycle_stage",
        "crossing_strategy",
        "sequencing_type",
        "genotyping_strategy",
        "sample_unit",
        "ploidy",
        "n_haplotypes",
        "n_haplotype_states",
        "haplotype_states",
        "haplotype_dosage",
    )
    for field in fields:
        try:
            value = getattr(experiment_params, field)
        except AttributeError:
            continue
        log.info(f"  {field}: {_format_value(value)}")


def _print_info_summary(records):
    log.info(f"record_type: {records.__class__.__qualname__}")
    log.info(f"barcodes: {len(records)}")
    log.info(f"bin_size: {records.bin_size}")
    log.info("chrom_sizes:")
    for chrom, size in records.chrom_sizes.items():
        log.info(f"  {chrom}: {size} bp, {records.nbins[chrom]} bins")
    _print_experiment_params(records.experiment_params)


def _print_genotypes(records):
    if 'genotypes' not in records.metadata:
        log.warn('No genotyping metadata found')
        return
    genotypes = records.metadata['genotypes']
    counts = Counter(genotypes._data.values())
    log.info('genotypes:')
    for geno, count in counts.most_common():
        log.info(f"  {geno}: {count} barcodes")


def _print_metadata(records):
    log.info("metadata:")
    if not records.metadata:
        log.info("  none")
        return
    for name, data in records.metadata.items():
        levels = getattr(data, "levels", None)
        length = len(data) if hasattr(data, "__len__") else "unknown"
        if levels is None:
            log.info(f"  {name}: {data.__class__.__qualname__}, length={length}")
        else:
            log.info(
                f"  {name}: {data.__class__.__qualname__}, "
                f"levels={_format_value(levels)}, length={length}"
            )


def _print_barcodes(records):
    for cb in records.barcodes:
        click.echo(f"{cb}")


def run_inspect(json_fn, cb_whitelist_fn, genotypes=True, metadata=False, barcodes=False, query=None):
    records = load_json(
        json_fn,
        cb_whitelist_fn=cb_whitelist_fn,
        bin_size=None,
        data_type="auto",
        frozen=True,
    )
    if query is not None:
        records = records.query(query)

    if barcodes:
        _print_barcodes(records)
    else:
        _print_info_summary(records)
        if genotypes:
            _print_genotypes(records)
        if metadata:
            _print_metadata(records)
