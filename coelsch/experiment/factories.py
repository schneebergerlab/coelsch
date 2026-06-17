from collections import defaultdict

from coelsch.records import PredictionRecords
from .design import ExperimentalDesign
from .params import ExperimentParams
from .genotypes import GenotypeKey


def from_recombinant_parental_haplotypes(
    recombinant_parental_haplotypes,
    experiment_params,
):
    if not isinstance(experiment_params, ExperimentParams):
        raise TypeError("experiment_params must be an ExperimentParams instance")

    if experiment_params.genotyping_strategy != "recombinant":
        raise ValueError(
            "from_recombinant_parental_haplotypes requires "
            "experiment_params.genotyping_strategy == 'recombinant'"
        )

    records = _normalise_prediction_records(recombinant_parental_haplotypes)
    _validate_prediction_records(records)

    if len(records) == 1:
        return _setup_from_diploid_parental_predictions(
            records[0],
            experiment_params,
        )

    if len(records) == 2:
        return _setup_from_two_gamete_parental_predictions(
            records[0],
            records[1],
            experiment_params,
        )

    raise ValueError(
        "recombinant_parental_haplotypes must be a PredictionRecords object "
        "or a list/tuple containing one or two PredictionRecords objects"
    )


def _normalise_prediction_records(recombinant_parental_haplotypes):
    if isinstance(recombinant_parental_haplotypes, PredictionRecords):
        return (recombinant_parental_haplotypes,)

    if isinstance(recombinant_parental_haplotypes, (list, tuple)):
        records = tuple(recombinant_parental_haplotypes)

        if len(records) not in {1, 2}:
            raise ValueError(
                "recombinant_parental_haplotypes must contain one or two "
                "PredictionRecords objects"
            )

        return records

    raise TypeError(
        "recombinant_parental_haplotypes must be a PredictionRecords object "
        "or a list/tuple of PredictionRecords objects"
    )


def _validate_prediction_records(records):
    for record in records:
        if not isinstance(record, PredictionRecords):
            raise TypeError(
                "All recombinant_parental_haplotypes entries must be "
                "PredictionRecords objects"
            )

        if "genotypes" not in record.metadata:
            raise ValueError(
                "PredictionRecords objects require metadata['genotypes']"
            )

    if len(records) == 2:
        left, right = records

        if sorted(left.barcodes) != sorted(right.barcodes):
            raise ValueError(
                "Sample/barcode names in recombinant_parental_haplotypes "
                "must match exactly"
            )

        if left.chrom_sizes != right.chrom_sizes:
            raise ValueError(
                "PredictionRecords chrom_sizes must match exactly"
            )

        if left.bin_size != right.bin_size:
            raise ValueError(
                "PredictionRecords bin_size values must match exactly"
            )


def _validate_single_parental_predictions(record):
    design = record.experiment_params

    if design.ploidy != 2:
        raise ValueError(
            "A single PredictionRecords input must come from a diploid experiment"
        )


def _validate_double_parental_predictions(left, right):
    for record in (left, right):
        design = record.experiment_params

        if design.ploidy != 1:
            raise ValueError(
                "Gamete parental PredictionRecords inputs must represent "
                "haploid predictions"
            )


def _metadata_genotype(record, sample):
    try:
        genotype = record.metadata["genotypes"][sample]
    except KeyError as exc:
        raise ValueError(
            f"PredictionRecords metadata['genotypes'] is missing sample {sample!r}"
        ) from exc

    return GenotypeKey.from_any(genotype)


def _setup_from_double_parental_predictions(left, right, experiment_params):
    _validate_double_parental_predictions(left, right)

    genotypes = []
    positional_genotypes = defaultdict(dict)

    for sample in left.barcodes:
        left_genotype = _metadata_genotype(left, sample)
        right_genotype = _metadata_genotype(right, sample)

        genotype = GenotypeKey(
            (left_genotype.to_nested_tuple(),
             right_genotype.to_nested_tuple()),
            name=sample,
        )
        genotypes.append(genotype)

        for chrom in left.chrom_sizes:
            left_path = left.get_state_labels(sample, chrom)
            right_path = right.get_state_labels(sample, chrom)

            for bin_idx, pos_geno in enumerate(zip(left_path, right_path)):
                positional_genotypes[(chrom, bin_idx)][genotype] = GenotypeKey(pos_geno)

    return ExperimentalDesign(
        genotypes,
        experiment_params=experiment_params,
        positional_genotypes=dict(positional_genotypes),
    )


def _setup_from_single_parental_predictions(record, experiment_params):
    _validate_single_parental_predictions(record)

    genotypes = []
    positional_genotypes = defaultdict(dict)

    for sample in record.barcodes:
        genotype = _metadata_genotype(record, sample)
        genotypes.append(genotype)

        for chrom in record.chrom_sizes:
            state_path = record.get_state_labels(sample, chrom)
            for bin_idx, pos_geno in enumerate(state_path):
                positional_genotypes[(chrom, bin_idx)][genotype] = GenotypeKey(pos_geno)

    return ExperimentalDesign(
        genotypes,
        experiment_params=experiment_params,
        positional_genotypes=dict(positional_genotypes),
    )