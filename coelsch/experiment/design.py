from .params import ExperimentParams
from .genotypes import GenotypeKey, PositionalGenotypes


class ExperimentalDesign:

    def __init__(
        self,
        genotypes,
        experiment_params,
        positional_genotypes=None,
    ):
        if not isinstance(experiment_params, ExperimentParams):
            raise TypeError("experiment_params must be an ExperimentParams instance")

        genotypes = tuple(GenotypeKey.from_any(g) for g in genotypes)
        if not genotypes:
            raise ValueError("genotypes cannot be empty")

        if experiment_params.genotyping_strategy == "founder":
            positional_genotypes = PositionalGenotypes(
                nbins=None,
                genotypes=genotypes,
                genotyping_strategy=experiment_params.genotyping_strategy
            )
        elif positional_genotypes is None:
            raise ValueError("positional_genotypes must be supplied when experiment_params.genotyping_strategy == 'recombinant'")
        else:
            positional_genotypes.genotypes = list(genotypes)

        self.experiment_params = experiment_params
        self.positional_genotypes = positional_genotypes
        self.genotypes = genotypes

        self._check_compatible()
        self._check_duplicates()

        self.idx = {
            g: i
            for i, g in enumerate(self.genotypes)
        }

        self._geno_dict = {
            g.name: g
            for g in self.genotypes
            if g.name is not None
        }

    def _check_compatible(self):
        """
        Check that every genotype matches the expected ExperimentalDesign.
        """
        for genotype in self.genotypes:
            self.experiment_params.check_compatibility(genotype)

    def _check_duplicates(self):
        """
        Check for duplicate names and equivalent genotype entries.
        """
        seen = {}

        for genotype in self.genotypes:
            if genotype in seen:
                raise ValueError(
                    "Duplicate or equivalent genotypes detected: "
                    f"{seen[genotype]!r} and {genotype!r}"
                )
            seen[genotype] = genotype

        seen_names = {}

        for genotype in self.genotypes:
            if genotype.name is None:
                continue

            if genotype.name in seen_names:
                raise ValueError(
                    f"Duplicate genotype name detected: {genotype.name!r}"
                )

            seen_names[genotype.name] = genotype

    def __getattr__(self, attr):
        try:
            return getattr(self.experiment_params, attr)
        except AttributeError as exc:
            raise AttributeError(
                f"{type(self).__qualname__} object has no attribute {attr!r}"
            ) from exc

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, TypeError, IndexError):
            return default

    def __len__(self):
        return len(self.genotypes)

    def __iter__(self):
        return iter(self.genotypes)

    @property
    def founder_haplotypes(self):
        founders = []
        seen = set()
        for genotype in self.genotypes:
            for founder in genotype.founders:
                if founder not in seen:
                    founders.append(founder)
                    seen.add(founder)
        return tuple(founders)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.genotypes[key]
        if isinstance(key, str):
            return self._geno_dict[key]
        genotype = GenotypeKey.from_any(key)
        return self.genotypes[self.idx[genotype]]

    def __contains__(self, key):
        try:
            self[key]
        except (KeyError, TypeError, IndexError):
            return False
        return True

    def __repr__(self):
        return (
            f"{type(self).__qualname__}("
            f"n_genotypes={len(self)}, "
            f"lifecycle_stage={self.lifecycle_stage!r}, "
            f"crossing_strategy={self.crossing_strategy!r}, "
            f"genotyping_strategy={self.genotyping_strategy!r})"
        )
