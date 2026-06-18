from dataclasses import dataclass
from .genotypes import GenotypeKey


@dataclass(frozen=True)
class ExperimentParams:
    """
    Dataset-level experimental params.
    This defines the expected lifecycle stage, crossing structure, and
    genotyping interpretation for all genotype strings in a dataset.

    Concrete founder names are supplied by GenotypeKey instances.
    """

    lifecycle_stage: str
    crossing_strategy: str
    sequencing_type: str
    genotyping_strategy: str = "founder"

    VALID_LIFECYCLE_STAGES = frozenset({"gametes", "progeny"})
    VALID_CROSSING_STRATEGIES = frozenset({
        "f1",
        "f2",
        "backcross",
        "testcross",
        "three_way",
        "four_way",
    })
    VALID_SEQUENCING_TYPES = frozenset({
        '10x_rna', '10x_atac', 'bd_rna', 'bd_atac',
        'takara_dna', 'wgs', 'other'
    })
    VALID_GENOTYPING_STRATEGIES = frozenset({
        "founder",
        "recombinant",
    })

    def __post_init__(self):
        lifecycle_stage = self.lifecycle_stage.lower()
        crossing_strategy = self.crossing_strategy.lower()
        sequencing_type = self.sequencing_type.lower()
        genotyping_strategy = self.genotyping_strategy.lower()

        self._validate_choice(
            "lifecycle_stage",
            lifecycle_stage,
            self.VALID_LIFECYCLE_STAGES,
        )
        self._validate_choice(
            "crossing_strategy",
            crossing_strategy,
            self.VALID_CROSSING_STRATEGIES,
        )
        self._validate_choice(
            "sequencing_type",
            sequencing_type,
            self.VALID_SEQUENCING_TYPES,
        )
        self._validate_choice(
            "genotyping_strategy",
            genotyping_strategy,
            self.VALID_GENOTYPING_STRATEGIES,
        )

        object.__setattr__(self, "lifecycle_stage", lifecycle_stage)
        object.__setattr__(self, "crossing_strategy", crossing_strategy)
        object.__setattr__(self, "sequencing_type", sequencing_type)
        object.__setattr__(self, "genotyping_strategy", genotyping_strategy)

        self.check_sane()

    def check_sane(self):
        if self.lifecycle_stage == "gametes":
            if self.crossing_strategy == "f1":
                if self.genotyping_strategy != "founder":
                    raise ValueError("f1 gametes should use genotyping_strategy='founder'")
                return True

            if self.crossing_strategy in {"backcross", "testcross", "three_way", "four_way"}:
                if self.genotyping_strategy != "recombinant":
                    raise ValueError(
                        f"{self.crossing_strategy} gametes requires "
                        "genotyping_strategy='recombinant' so the pre-meiotic "
                        "parental haplotypes can be resolved before crossover inference"
                    )
                return True

            raise ValueError(
                f"crossing_strategy={self.crossing_strategy!r} is not supported for gametes"
            )

        if self.lifecycle_stage == "progeny":
            if self.crossing_strategy == "f1":
                raise ValueError(
                    "crossing_strategy='f1' is ambiguous for progeny; use 'f2', "
                    "'backcross', 'testcross', 'three_way', or 'four_way'"
                )

            if self.crossing_strategy in {"f2", "backcross", "testcross", "three_way", "four_way"}:
                if self.genotyping_strategy != "founder":
                    raise ValueError(
                        f"{self.crossing_strategy} progeny should use "
                        "genotyping_strategy='founder'"
                    )
                return True

        raise ValueError(
            f"Unsupported design: lifecycle_stage={self.lifecycle_stage!r}, "
            f"crossing_strategy={self.crossing_strategy!r}, "
            f"genotyping_strategy={self.genotyping_strategy!r}"
        )

    @staticmethod
    def _validate_choice(name, value, valid):
        """
        Validate an enumerated string field.
        """
        if value not in valid:
            raise ValueError(
                f"Unsupported {name} {value!r}; expected one of {sorted(valid)}"
            )

    @property
    def ploidy(self):
        """
        Ploidy implied by lifecycle stage.
        """
        return 1 if self.lifecycle_stage == "gametes" else 2

    @property
    def n_haplotypes(self):
        """
        number of founder haplotypes implied by the crossing_strategy
        """
        if self.crossing_strategy in ("f1", "f2", "backcross"):
            return 2
        elif self.crossing_strategy in ("testcross", "three_way"):
            return 3
        elif self.crossing_strategy == "four_way":
            return 4
        return NotImplemented

    @property
    def haplotype_states(self):
        if self.genotyping_strategy == "recombinant":
            # recombinant genotyping always collapses to two haplotypes
            return ((0,), (1,))

        states = {
            "f1": ((0,), (1,)),
            "f2": ((0, 0), (0, 1), (1, 0), (1, 1)),
            "backcross": ((0, 0), (0, 1)),
            "testcross": ((0, 1),(0, 2)),
            "three_way": ((0, 0), (0, 2), (1, 0), (1, 2)),
            "four_way": ((0, 2), (0, 3), (1, 2), (1, 3)),
        }

        return states[self.crossing_strategy]

    @property
    def n_haplotype_states(self):
        return len(self.haplotype_states)

    def check_compatibility(self, genotype_key):
        """
        Check whether a genotype is compatible with this design.

        Parameters
        ----------
        genotype_key
            GenotypeKey, string, or nested tuple.

        Returns
        -------
        bool
            True if compatible.

        Raises
        ------
        ValueError
            If the genotype tree does not match the crossing strategy.
        """
        genotype_key = GenotypeKey.from_any(genotype_key)
        tree = genotype_key.to_nested_tuple()

        checkers = {
            "f1": self._is_f1,
            "f2": self._is_f2,
            "backcross": self._is_backcross,
            "testcross": self._is_testcross,
            "three_way": self._is_three_way,
            "four_way": self._is_four_way,
        }

        if not checkers[self.crossing_strategy](tree):
            raise ValueError(
                f"Genotype {genotype_key!r} is not compatible with "
                f"crossing_strategy={self.crossing_strategy!r}"
            )

        return True

    def _is_f1(self, tree):
        """
        (A*B)
        """
        return GenotypeKey.is_simple_cross_node(tree)

    def _is_f2(self, tree):
        """
        (A*B), interpreted as F2 in progeny context, or ((A*B)*(A*B)).
        """
        if GenotypeKey.is_simple_cross_node(tree):
            return True

        left, right = tree
        if left == right and GenotypeKey.is_simple_cross_node(left):
            return True

        return False

    def _is_backcross(self, tree):
        """
        A*(A*B) or (A*B)*A.
        """
        canon_tree = GenotypeKey.fixed_by_simple_cross_node(tree)
        if canon_tree is None:
            return False

        left, right = canon_tree
        return GenotypeKey.is_simple_cross_node(right) and left in right

    def _is_testcross(self, tree):
        """
        A*(B*C) or (B*C)*A.
        """
        canon_tree = GenotypeKey.fixed_by_simple_cross_node(tree)
        if canon_tree is None:
            return False

        left, right = canon_tree
        return GenotypeKey.is_simple_cross_node(right) and left not in right

    def _is_three_way(self, tree):
        """
        (A*B)*(A*C).
        """
        left, right = tree
        if GenotypeKey.is_simple_cross_node(left) and GenotypeKey.is_simple_cross_node(right):
            left_set, right_set = set(left), set(right)
            if len(left_set | right_set) == 3 and len(left_set & right_set) == 1:
                return True

        return False

    def _is_four_way(self, tree):
        """
        (A*B)*(C*D).
        """
        left, right = tree
        if GenotypeKey.is_simple_cross_node(left) and GenotypeKey.is_simple_cross_node(right):
            left_set, right_set = set(left), set(right)
            if len(left_set | right_set) == 4:
                return True

        return False

    def to_json(self):
        return {
            'lifecycle_stage': self.lifecycle_stage,
            'crossing_strategy': self.crossing_strategy,
            'sequencing_type': self.sequencing_type,
            'genotyping_strategy': self.genotyping_strategy
        }

    @classmethod
    def from_json(cls, obj):
        if obj is None:
            raise ValueError('experiment_params JSON object is missing')

        lifecycle_stage = obj.get('lifecycle_stage', obj.get('lifecyle_stage'))
        if lifecycle_stage is None:
            raise ValueError("experiment_params JSON requires 'lifecycle_stage'")

        return cls(
            lifecycle_stage=lifecycle_stage,
            crossing_strategy=obj['crossing_strategy'],
            sequencing_type=obj['sequencing_type'],
            genotyping_strategy=obj.get('genotyping_strategy', 'founder')
        )

    @classmethod
    def from_legacy(cls, seq_type, ploidy_type):
        if ploidy_type == 'haploid':
            lifecycle_stage = 'gametes'
            crossing_strategy = 'f1'
        elif ploidy_type == 'diploid_f2':
            lifecycle_stage = 'progeny'
            crossing_strategy = 'f2'
        elif ploidy_type == 'diploid_bc1':
            lifecycle_stage = 'progeny'
            crossing_strategy = 'backcross'
        else:
            raise ValueError(f'Unexpected ploidy_type "{ploidy_type}" whilst '
                             f'parsing legacy {cls.__qualname__}')
        return cls(
            lifecycle_stage=lifecycle_stage,
            crossing_strategy=crossing_strategy,
            sequencing_type=seq_type,
            genotyping_strategy='founder',
        )