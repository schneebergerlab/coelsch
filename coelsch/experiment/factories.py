from collections import defaultdict
import itertools as it

from coelsch.records import PredictionRecords
from .design import ExperimentalDesign
from .params import ExperimentParams
from .genotypes import GenotypeKey, PositionalGenotypes
from .utils import get_all_haplotypes_bam, get_all_haplotypes_vcf


def create_experimental_design(
    lifecycle_stage, crossing_strategy, sequencing_type,
    genotyping_strategy, sample_unit="auto", crossing_combinations=None,
    recombinant_parental_haplotypes=None,
    all_haplotypes=None, bam_fn=None, vcf_fn=None, ref_name=None,
    has_named_haplotypes=True,
):
    experiment_params = ExperimentParams(
        lifecycle_stage=lifecycle_stage,
        crossing_strategy=crossing_strategy,
        sequencing_type=sequencing_type,
        genotyping_strategy=genotyping_strategy,
        sample_unit=sample_unit,
    )

    if not has_named_haplotypes:
        if genotyping_strategy != "founder":
            raise ValueError(
                "unnamed two-column haplotypes only support "
                "genotyping_strategy='founder'"
            )
        if crossing_strategy not in {"f1", "f2"}:
            raise ValueError(
                "unnamed two-column haplotypes only support "
                "crossing_strategy='f1' or crossing_strategy='f2'; use "
                "multi_haplotype BAM tags or a genotype VCF for complex crosses"
            )
        if crossing_combinations is not None:
            raise ValueError(
                "crossing_combinations require named haplotypes; omit them for "
                "star_diploid BAM or cellSNP-lite without a genotype VCF"
            )
        if recombinant_parental_haplotypes is not None:
            raise ValueError(
                "recombinant_parental_haplotypes require named haplotypes"
            )

        return ExperimentalDesign(
            genotypes=[("ref", "alt")],
            experiment_params=experiment_params,
        )

    if genotyping_strategy == "founder":

        if all_haplotypes is None:
            if bam_fn is not None and vcf_fn is None:
                all_haplotypes = get_all_haplotypes_bam(bam_fn)
            elif vcf_fn is not None and bam_fn is None:
                all_haplotypes = get_all_haplotypes_vcf(vcf_fn, ref_name)

        if crossing_combinations is None:
            if crossing_strategy not in ("f1", "f2"):
                raise ValueError("for crosses involving three or more founders, crossing_combinations "
                                 "must be specified")
            if all_haplotypes is None:
                raise ValueError("when genotyping_strategy == 'founder', either crossing_combinations or "
                                 "a method of determining all_haplotypes is required")
            crossing_combinations = list(it.combinations(sorted(all_haplotypes), r=2))
        elif all_haplotypes is not None:
            all_haplotypes = set(all_haplotypes)
            crossing_haplotypes = {
                hap
                for crossing_combination in crossing_combinations
                for hap in GenotypeKey.from_any(crossing_combination).leaves
            }
            missing_haplotypes = crossing_haplotypes - all_haplotypes
            if missing_haplotypes:
                raise ValueError(
                    "crossing_combinations include haplotypes not present in the input: "
                    f"{sorted(missing_haplotypes)!r}"
                )

        experimental_design = ExperimentalDesign(
            genotypes=crossing_combinations,
            experiment_params=experiment_params
        )
    else:
        if recombinant_parental_haplotypes is None:
            raise ValueError("recombinant_parental_haplotypes must be provided "
                             "when genotyping_strategy == 'recombinant'")
        experimental_design = from_recombinant_parental_haplotypes(
            recombinant_parental_haplotypes,
            experiment_params=experiment_params,
        )
    return experimental_design


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
        return _setup_from_single_parental_predictions(
            records[0],
            experiment_params,
        )

    if len(records) == 2:
        return _setup_from_double_parental_predictions(
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

        if record.experiment_params is None:
            raise ValueError(
                "PredictionRecords objects require experiment_params"
            )

        if record._ndim not in {1, 2}:
            raise ValueError(
                "PredictionRecords must have a valid scalar or multistate array schema"
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

    return GenotypeKey.from_any(genotype, name=sample)


def _setup_from_double_parental_predictions(left, right, experiment_params):
    _validate_double_parental_predictions(left, right)

    positional_genotypes = PositionalGenotypes(
        left.nbins,
        genotyping_strategy=experiment_params.genotyping_strategy
    )

    for sample in left.barcodes:
        left_genotype = _metadata_genotype(left, sample)
        right_genotype = _metadata_genotype(right, sample)

        genotype = GenotypeKey(
            (left_genotype.to_nested_tuple(),
             right_genotype.to_nested_tuple()),
            name=sample,
        )
        experiment_params.check_compatibility(genotype)
        positional_genotypes.add_genotype(genotype)

        for chrom in left.chrom_sizes:
            left_path = left.get_haplotype_labels(sample, chrom, as_genotype_keys=True)
            right_path = right.get_haplotype_labels(sample, chrom, as_genotype_keys=True)

            for bin_idx, (left_pos_genotype, right_pos_genotype) in enumerate(zip(left_path, right_path)):
                pos_tree = (
                    left_pos_genotype.to_nested_tuple(),
                    right_pos_genotype.to_nested_tuple(),
                )
                positional_genotypes[(chrom, bin_idx)][genotype] = GenotypeKey(pos_tree, name=sample)

    return ExperimentalDesign(
        positional_genotypes.genotypes,
        experiment_params=experiment_params,
        positional_genotypes=positional_genotypes,
    )


def _setup_from_single_parental_predictions(record, experiment_params):
    _validate_single_parental_predictions(record)

    positional_genotypes = PositionalGenotypes(
        record.nbins,
        genotyping_strategy=experiment_params.genotyping_strategy
    )

    for sample in record.barcodes:
        genotype = _metadata_genotype(record, sample)
        experiment_params.check_compatibility(genotype)
        positional_genotypes.add_genotype(genotype)

        for chrom in record.chrom_sizes:
            pos_genotypes = record.get_haplotype_labels(sample, chrom, as_genotype_keys=True)
            for bin_idx, pos_genotype in enumerate(pos_genotypes):
                positional_genotypes[(chrom, bin_idx)][genotype] = pos_genotype

    return ExperimentalDesign(
        positional_genotypes.genotypes,
        experiment_params=experiment_params,
        positional_genotypes=positional_genotypes,
    )