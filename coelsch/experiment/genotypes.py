from dataclasses import dataclass, field


@dataclass(frozen=True, eq=False)
class GenotypeKey:

    genotype: object
    name: str = None
    parental_roles: str = "undefined"
    sex_roles: tuple = ()
    leaves: tuple = field(init=False, repr=False)
    founders: tuple = field(init=False, repr=False)
    founder_set: frozenset = field(init=False)
    haplotype_index: dict = field(init=False, repr=False)
    depth: int = field(init=False, repr=False)

    VALID_PARENTAL_ROLES = frozenset({"undefined", "sexed"})
    VALID_SEX_LABELS = frozenset({"f", "m"})

    def __post_init__(self):
        tree = self._normalise_tree(self.genotype)
        parental_roles = self.parental_roles.lower()
        role_map = self._normalise_sex_roles(self.sex_roles)

        if parental_roles not in self.VALID_PARENTAL_ROLES:
            raise ValueError(
                f"Unsupported parental_roles {self.parental_roles!r}; "
                f"expected one of {sorted(self.VALID_PARENTAL_ROLES)}"
            )

        if role_map and parental_roles == "undefined":
            parental_roles = "sexed"

        if parental_roles == "sexed":
            role_map = self._fill_default_sex_roles(tree, role_map)
            self._validate_sex_roles(tree, role_map)
            tree, role_map = self._canonicalise_sexed_tree(tree, role_map)
        else:
            tree = self._canonicalise_undefined_tree(tree)
            role_map = {}

        leaves = self._leaves(tree)
        founders = self._ordered_founders(tree, role_map)
        haplotype_index = {hap: i for i, hap in enumerate(founders)}

        object.__setattr__(self, "genotype", tree)
        object.__setattr__(self, "parental_roles", parental_roles)
        object.__setattr__(self, "sex_roles", tuple(sorted(role_map.items())))
        object.__setattr__(self, "leaves", leaves)
        object.__setattr__(self, "founders", founders)
        object.__setattr__(self, "founder_set", frozenset(founders))
        object.__setattr__(self, "haplotype_index", haplotype_index)
        object.__setattr__(self, "depth", self._depth(tree))

    @classmethod
    def from_str(cls, genotype_str, name=None, parental_roles="infer"):
        genotype_str = genotype_str.strip()
        i = genotype_str.find("(")

        if i < 0:
            if any(x in genotype_str for x in "*[]"):
                raise ValueError(f"Malformed genotype string: {genotype_str!r}")

            if parental_roles == "infer":
                parental_roles = "undefined"

            return cls(
                genotype_str,
                name=name,
                parental_roles=parental_roles,
            )

        if i and name is None:
            name = genotype_str[:i].rstrip(": ").strip()

        expr = genotype_str[i:].strip()
        tree, role_map = cls._parse_expr_with_roles(expr)

        if parental_roles == "infer":
            parental_roles = "sexed" if role_map else "undefined"

        return cls(
            tree,
            name=name,
            parental_roles=parental_roles,
            sex_roles=tuple(role_map.items()),
        )

    @classmethod
    def from_any(cls, obj, parental_roles="infer"):
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, str):
            return cls.from_str(obj, parental_roles=parental_roles)
        if isinstance(obj, tuple):
            if parental_roles == "infer":
                parental_roles = "undefined"
            return cls(obj, parental_roles=parental_roles)
        raise TypeError("Expected GenotypeKey, str, or tuple")

    @classmethod
    def _parse_expr_with_roles(cls, expr):
        tree, role_map, suffix, pos = cls._parse_node(expr, 0)
        pos = cls._skip_ws(expr, pos)

        if suffix is not None:
            raise ValueError("Top-level genotype expression should not have a sex label")

        if pos != len(expr):
            raise ValueError(f"Trailing characters after genotype expression: {expr[pos:]!r}")

        return tree, role_map

    @classmethod
    def _parse_node(cls, expr, pos):
        pos = cls._skip_ws(expr, pos)

        if pos >= len(expr):
            raise ValueError("Unexpected end of genotype expression")

        if expr[pos] == "(":
            return cls._parse_cross_node(expr, pos)

        return cls._parse_leaf_node(expr, pos)

    @classmethod
    def _parse_cross_node(cls, expr, pos):
        pos += 1

        left, left_roles, left_suffix, pos = cls._parse_node(expr, pos)

        pos = cls._skip_ws(expr, pos)
        if pos >= len(expr) or expr[pos] != "*":
            raise ValueError(f"Expected '*' in genotype expression near {expr[pos:]!r}")
        pos += 1

        right, right_roles, right_suffix, pos = cls._parse_node(expr, pos)

        pos = cls._skip_ws(expr, pos)
        if pos >= len(expr) or expr[pos] != ")":
            raise ValueError(f"Expected ')' in genotype expression near {expr[pos:]!r}")
        pos += 1

        role_map = {}
        role_map.update({(0,) + path: roles for path, roles in left_roles.items()})
        role_map.update({(1,) + path: roles for path, roles in right_roles.items()})

        if (left_suffix is None) ^ (right_suffix is None):
            raise ValueError(
                "Sex labels must be supplied for both parents of a cross, "
                "or for neither parent"
            )

        if left_suffix is not None:
            if left_suffix == right_suffix:
                raise ValueError(
                    "Sex labels for a cross must contain one [f] parent "
                    "and one [m] parent"
                )
            role_map[()] = (left_suffix, right_suffix)

        suffix, pos = cls._parse_suffix(expr, pos)
        return (left, right), role_map, suffix, pos

    @classmethod
    def _parse_leaf_node(cls, expr, pos):
        start = pos

        while pos < len(expr) and expr[pos] not in "*()[]":
            pos += 1

        label = expr[start:pos].strip()
        if not label:
            raise ValueError(f"Empty genotype leaf near {expr[start:]!r}")

        suffix, pos = cls._parse_suffix(expr, pos)
        return label, {}, suffix, pos

    @classmethod
    def _parse_suffix(cls, expr, pos):
        pos = cls._skip_ws(expr, pos)

        if pos >= len(expr) or expr[pos] != "[":
            return None, pos

        end = expr.find("]", pos)
        if end < 0:
            raise ValueError(f"Unclosed sex label near {expr[pos:]!r}")

        label = expr[pos + 1:end].strip().lower()
        if label not in cls.VALID_SEX_LABELS:
            raise ValueError(
                f"Unsupported sex label {label!r}; expected 'f' or 'm'"
            )

        return label, end + 1

    @staticmethod
    def _skip_ws(expr, pos):
        while pos < len(expr) and expr[pos].isspace():
            pos += 1
        return pos

    @classmethod
    def _normalise_tree(cls, node):
        if isinstance(node, list):
            node = tuple(node)

        if isinstance(node, tuple):
            if len(node) != 2:
                raise ValueError(f"Genotype tree nodes must be binary; got {node!r}")
            return tuple(cls._normalise_tree(x) for x in node)

        if not isinstance(node, str):
            node = str(node)

        node = node.strip()
        if not node:
            raise ValueError("Founder labels cannot be empty")

        return node

    @classmethod
    def _normalise_sex_roles(cls, sex_roles):
        if sex_roles is None:
            return {}

        items = sex_roles.items() if isinstance(sex_roles, dict) else sex_roles
        role_map = {}

        for path, roles in items:
            path = tuple(path)
            roles = tuple(r.lower() for r in roles)

            if len(roles) != 2:
                raise ValueError(f"Sex role entry for path {path!r} must have length 2")

            if set(roles) != {"f", "m"}:
                raise ValueError(
                    f"Sex role entry for path {path!r} must contain one 'f' and one 'm'"
                )

            role_map[path] = roles

        return role_map

    @classmethod
    def _internal_paths(cls, node, path=()):
        if cls.is_leaf(node):
            return ()

        left, right = node
        return (
            path,
            *cls._internal_paths(left, path + (0,)),
            *cls._internal_paths(right, path + (1,)),
        )

    @classmethod
    def _fill_default_sex_roles(cls, tree, role_map):
        role_map = dict(role_map)

        for path in cls._internal_paths(tree):
            role_map.setdefault(path, ("f", "m"))

        return role_map

    @classmethod
    def _validate_sex_roles(cls, tree, role_map):
        internal_paths = set(cls._internal_paths(tree))

        for path, roles in role_map.items():
            if path not in internal_paths:
                raise ValueError(
                    f"Sex role path {path!r} does not refer to an internal cross node"
                )

            if set(roles) != {"f", "m"}:
                raise ValueError(
                    f"Sex role entry for path {path!r} must contain one 'f' and one 'm'"
                )

    @classmethod
    def _canonicalise_undefined_tree(cls, node):
        if cls.is_leaf(node):
            return node

        left, right = node
        left = cls._canonicalise_undefined_tree(left)
        right = cls._canonicalise_undefined_tree(right)

        return cls._order_pair(left, right)

    @classmethod
    def _canonicalise_sexed_tree(cls, node, role_map):
        if cls.is_leaf(node):
            return node, {}

        left, right = node
        left_tree, left_roles = cls._canonicalise_sexed_tree(
            left,
            cls._child_role_map(role_map, 0),
        )
        right_tree, right_roles = cls._canonicalise_sexed_tree(
            right,
            cls._child_role_map(role_map, 1),
        )

        left_role, right_role = role_map[()]

        if (left_role, right_role) == ("f", "m"):
            first_tree, second_tree = left_tree, right_tree
            first_roles, second_roles = left_roles, right_roles
        elif (left_role, right_role) == ("m", "f"):
            first_tree, second_tree = right_tree, left_tree
            first_roles, second_roles = right_roles, left_roles
        else:
            raise ValueError("Sexed cross must contain one female and one male parent")

        new_role_map = {(): ("f", "m")}
        new_role_map.update(cls._prefix_role_map(first_roles, 0))
        new_role_map.update(cls._prefix_role_map(second_roles, 1))

        return (first_tree, second_tree), new_role_map

    @classmethod
    def _order_pair(cls, left, right):
        if cls._canonical_sort_key(right) < cls._canonical_sort_key(left):
            return right, left
        return left, right

    @classmethod
    def _canonical_sort_key(cls, node):
        if cls.is_leaf(node):
            return (0, str(node))

        return (1, cls._tree_to_str(node))

    @staticmethod
    def _child_role_map(role_map, child_idx):
        return {
            path[1:]: roles
            for path, roles in role_map.items()
            if path and path[0] == child_idx
        }

    @staticmethod
    def _prefix_role_map(role_map, child_idx):
        return {
            (child_idx,) + path: roles
            for path, roles in role_map.items()
        }

    def _identity(self):
        if self.parental_roles == "sexed":
            return (
                self.parental_roles,
                self.genotype,
                tuple(sorted(self.sex_roles)),
            )

        return (
            self.parental_roles,
            self.genotype,
        )

    def __eq__(self, other):
        if not isinstance(other, GenotypeKey):
            return NotImplemented
        return self._identity() == other._identity()

    def __hash__(self):
        return hash(self._identity())

    @staticmethod
    def is_leaf(node):
        return not isinstance(node, tuple)

    @classmethod
    def is_simple_cross_node(cls, node):
        return (
            isinstance(node, tuple)
            and len(node) == 2
            and cls.is_leaf(node[0])
            and cls.is_leaf(node[1])
            and node[0] != node[1]
        )

    @classmethod
    def fixed_by_simple_cross_node(cls, node):
        if not isinstance(node, tuple) or len(node) != 2:
            return None

        left, right = node

        if cls.is_leaf(left) and cls.is_simple_cross_node(right):
            return left, right

        if cls.is_simple_cross_node(left) and cls.is_leaf(right):
            return right, left

        return None

    @classmethod
    def _leaves(cls, node):
        if cls.is_leaf(node):
            return (node,)

        left, right = node
        return cls._leaves(left) + cls._leaves(right)

    @classmethod
    def _leaf_paths(cls, node, path=()):
        if cls.is_leaf(node):
            return ((node, path),)

        left, right = node
        return (
            *cls._leaf_paths(left, path + (0,)),
            *cls._leaf_paths(right, path + (1,)),
        )

    @classmethod
    def _founder_sex_priority(cls, node, role_map):
        priority = {}
        for hap, path in cls._leaf_paths(node):
            if not path:
                continue
            roles = role_map.get(path[:-1])
            if roles is None:
                continue
            role_priority = 0 if roles[path[-1]] == "f" else 1
            priority[hap] = min(priority.get(hap, role_priority), role_priority)
        return priority

    @classmethod
    def _ordered_founders(cls, node, role_map=None):
        if role_map is None:
            role_map = {}
        leaves = cls._leaves(node)
        counts = {hap: leaves.count(hap) for hap in set(leaves)}
        sex_priority = cls._founder_sex_priority(node, role_map)
        return tuple(sorted(
            counts,
            key=lambda hap: (-counts[hap], sex_priority.get(hap, 2), str(hap)),
        ))

    @classmethod
    def _depth(cls, node):
        if cls.is_leaf(node):
            return 0

        left, right = node
        return 1 + max(cls._depth(left), cls._depth(right))

    @classmethod
    def _tree_to_str(cls, node, role_map=None, path=()):
        if cls.is_leaf(node):
            return str(node)

        left, right = node
        left_str = cls._tree_to_str(left, role_map, path + (0,))
        right_str = cls._tree_to_str(right, role_map, path + (1,))

        if role_map is not None:
            left_role, right_role = role_map[path]
            left_str = f"{left_str}[{left_role}]"
            right_str = f"{right_str}[{right_role}]"

        return f"({left_str}*{right_str})"

    @property
    def n_founders(self):
        return len(self.founders)

    def get_haplotype_index(self, hap):
        return self.haplotype_index[hap]

    @property
    def is_simple_cross(self):
        return self.is_simple_cross_node(self.genotype)

    @property
    def is_leaf_genotype(self):
        return self.is_leaf(self.genotype)

    @property
    def parent1(self):
        if self.is_leaf_genotype:
            raise ValueError("Leaf genotype has no parent1")

        role_map = dict(self.sex_roles)
        parent_roles = self._child_role_map(role_map, 0)
        name = f"{self.name}.p1" if self.name is not None else "p1"

        return GenotypeKey(
            self.genotype[0],
            name=name,
            parental_roles=self.parental_roles,
            sex_roles=tuple(parent_roles.items()),
        )

    @property
    def parent2(self):
        if self.is_leaf_genotype:
            raise ValueError("Leaf genotype has no parent2")

        role_map = dict(self.sex_roles)
        parent_roles = self._child_role_map(role_map, 1)
        name = f"{self.name}.p2" if self.name is not None else "p2"

        return GenotypeKey(
            self.genotype[1],
            name=name,
            parental_roles=self.parental_roles,
            sex_roles=tuple(parent_roles.items()),
        )

    @property
    def maternal_parent(self):
        if self.parental_roles != "sexed":
            raise AttributeError("maternal_parent is not defined")
        return self.parent1

    @property
    def paternal_parent(self):
        if self.parental_roles != "sexed":
            raise AttributeError("paternal_parent is not defined")
        return self.parent2

    def to_nested_tuple(self):
        return self.genotype

    def __str__(self):
        role_map = dict(self.sex_roles) if self.parental_roles == "sexed" else None
        genotype_str = self._tree_to_str(self.genotype, role_map=role_map)
        return f"{self.name}:{genotype_str}" if self.name is not None else genotype_str

    def __repr__(self):
        return str(self)

    def __getitem__(self, idx):
        if self.is_leaf_genotype:
            raise TypeError("Leaf genotype is not subscriptable")
        return self.genotype[idx]

    def __iter__(self):
        if self.is_leaf_genotype:
            return iter((self.genotype,))
        return iter(self.genotype)

    def __and__(self, other):
        if isinstance(other, GenotypeKey):
            return self.founder_set & other.founder_set
        if isinstance(other, (set, frozenset)):
            return self.founder_set & other
        return NotImplemented

    def __rand__(self, other):
        return self.__and__(other)


class PositionalGenotypes:

    def __init__(self, nbins, genotypes=None, genotyping_strategy='founder'):
        self.nbins = nbins
        if genotypes is None:
            self.genotypes = []
        else:
            self.genotypes = [GenotypeKey.from_any(g) for g in genotypes]
        self.genotyping_strategy = genotyping_strategy
        if genotyping_strategy == 'recombinant':
            self._geno_map = {
                (chrom, bin_idx): {}
                for chrom, n in nbins.items()
                for bin_idx in range(n)
            }
        else:
            self._geno_map = None

    def __getitem__(self, key):
        if self.genotyping_strategy != "recombinant":
            raise NotImplementedError()
        return self._geno_map[key]

    def add_genotype(self, genotype):
        self.genotypes.append(GenotypeKey.from_any(genotype))

    def get_bin_haplotypes(self, chrom, bin_idx):
        """
        Retrieve per-bin founder haplotype pairs for each candidate genotype.

        Parameters
        ----------
        chrom : str
            Chromosome identifier.
        bin_idx : int
            Zero-based bin index on the chromosome.

        Returns
        -------
        dict
            If ``genotyping_strategy='founder'``: ``{GenotypeKey -> GenotypeKey}`` (identity).
            If ``genotyping_strategy='recombinant'``: ``{GenotypeKey_overall -> GenotypeKey_positional}``
            for the requested bin.
        """
        if self.genotyping_strategy == 'founder':
            return {
                geno: geno for geno in self.genotypes
            }
        else:
            return self[(chrom, bin_idx)]