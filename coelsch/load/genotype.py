import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Literal
import itertools as it

import numpy as np

from joblib import Parallel, delayed

from .utils import weighted_chunks
from .counts import IntervalMarkerCounts
from coelsch.utils import spawn_child_rngs
from coelsch.records import PredictionRecords, NestedData
from coelsch.defaults import DEFAULT_RANDOM_SEED


log = logging.getLogger('coelsch')
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


@dataclass(frozen=True, eq=True)
class GenotypeKey:
    """
    Immutable key representing either a founder genotype (pair of haplotype tags)
    or a recombinant genotype (pair of founder genotypes).

    Parameters
    ----------
    genotype : tuple
        For ``mode='founder'``: a 2-tuple of haplotype tag strings,
        e.g. ``('A', 'B')``. For ``mode='recombinant'``: a 2-tuple of
        founder 2-tuples, e.g. ``(('A','B'), ('C','D'))``.
    name : str or None, optional
        Human-readable name/label for this genotype. If ``None``, the string
        form is just the canonical parenthesized expression, e.g. ``(A*B)``.
    mode : {'founder', 'recombinant'}, optional
        Whether this key represents a founder genotype or a recombinant.

    Attributes
    ----------
    genotype : tuple
        Canonicalized genotype tuple as described above.
    name : str or None
        The provided name or ``None``.
    mode : {'founder', 'recombinant'}
        The genotype mode.
    _geno_set : frozenset
        A frozenset of the two immediate terms in ``genotype`` used for
        set intersection operations.

    Notes
    -----
    - The string representation is recursive: founders print as ``(A*B)``,
      recombinants as ``((A*B)*(C*D))``; when a ``name`` is present it
      prefixes this as ``name:(...)``.
    """
    genotype: tuple
    name: str | None = None
    mode: Literal['founder','recombinant'] = 'founder'
    geno_set: frozenset = field(init=False)

    def __post_init__(self):
        if len(self.genotype) != 2:
            raise ValueError('genotype should consist of two items')

        if self.mode == 'founder':
            a, b = self.genotype
            if not (isinstance(a, str) and isinstance(b, str)):
                raise ValueError("founder needs ('hap1','hap2')")
        elif self.mode == 'recombinant':
            p1, p2 = self.genotype

            def _validate(term):
                if isinstance(term, tuple):
                    if len(term) != 2 or not all(isinstance(x, str) for x in term):
                        raise ValueError("bad recombinant term")
                    x, y = term
                else:
                    raise ValueError("bad recombinant term")
                return (x, y)

            object.__setattr__(self, 'genotype', (_validate(p1), _validate(p2)))
        else:
            raise ValueError("mode must be 'founder' or 'recombinant'")
        object.__setattr__(self, "geno_set", frozenset(self.genotype))

    def __getitem__(self, idx):
        return self.genotype[idx]

    @staticmethod
    def _geno_to_str(geno):
        hap1, hap2 = geno
        if isinstance(hap1, tuple):
            hap1 = GenotypeKey._geno_to_str(hap1)
        if isinstance(hap2, tuple):
            hap2 = GenotypeKey._geno_to_str(hap2)
        return f'({hap1}*{hap2})'

    def __str__(self):
        geno_str = self._geno_to_str(self.genotype)
        return f'{self.name}:{geno_str}' if self.name is not None else geno_str

    @classmethod
    def from_str(cls, geno_str, name=None):
        """
        Parse a string produced by ``GenotypeKey.__str__`` back into a GenotypeKey.

        Parameters
        ----------
        geno_str : str
            String of the form ``(A*B)``, ``name:(A*B)``, or
            ``((A*B)*(C*D))`` / ``name:((A*B)*(C*D))``.
        name : str or None, optional
            If provided, overrides any leading ``name:`` in ``geno_str``.

        Returns
        -------
        GenotypeKey
            Parsed genotype key.

        Raises
        ------
        ValueError
            If the string cannot be parsed or contains a mixed pair
            (e.g., ``(A*(B*C))``), which is not supported.
        """
        geno_str = geno_str.strip()
        i = geno_str.find('(')
        if i < 0: 
            raise ValueError('geno_str could not be parsed')
        if i and name is None:
            name = geno_str[:i].rstrip(': ') # strip whitespace and trailing ":"

        def _split_cross(expr):
            expr = expr.strip()
            if not expr.startswith('('):
                return expr, None
            inner = expr[1:-1]
            depth = 0
            for i, ch in enumerate(inner):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                elif ch == '*' and depth == 0:
                    parent1 = inner[:i].strip()
                    parent2 = inner[i+1:].strip()
                    break
            else:
                raise ValueError("geno_str could not be parsed")
            
            parent1, _ = _split_cross(parent1)
            parent2, _ = _split_cross(parent2)
            mode = 'recombinant' if isinstance(parent1, tuple) or isinstance(parent2, tuple) else 'founder'
            if mode == 'recombinant' and not (isinstance(parent1, tuple) and isinstance(parent2, tuple)):
                raise ValueError("mixed pair; expected ((a*b)*(c*d))")
            return (parent1, parent2), mode

        genotype, mode = _split_cross(geno_str[i:])
        return cls(genotype, name, mode)

    def cross(self, other, name=None):
        """
        Construct a recombinant genotype from two founder genotypes.

        Parameters
        ----------
        other : GenotypeKey
            The other parent genotype key (usually a founder).
        name : str or None, optional
            Optional label for the resulting recombinant.

        Returns
        -------
        GenotypeKey
            A recombinant genotype key with ``mode='recombinant'``.
        """
        return GenotypeKey(
            (self.genotype, other.genotype),
            name=name,
            mode='recombinant'
        )

    def __and__(self, other):
        if isinstance(other, GenotypeKey):
            return self.geno_set & other.geno_set
        elif isinstance(other, (set, frozenset)):
            return self.geno_set & other
        return NotImplemented

    def __rand__(self, other):
        return self.__and__(other)

    def __repr__(self):
        return str(self)


class GenotypesSet:
    """
    Container for a set of candidate genotypes and (optionally) their
    positional haplotypes along the genome.

    Parameters
    ----------
    genotypes : sequence of {GenotypeKey, tuple, str}
        Input genotypes. Each element may be:
        - ``GenotypeKey`` (used as-is),
        - a founder 2-tuple of haplotype tags (e.g., ``('A','B')``),
        - or a string parseable by ``GenotypeKey.from_str``.
    mode : {'founder', 'recombinant'}, optional
        Logical mode of this set; affects how positional haplotypes are
        returned (default ``'founder'``).
    positional_genotypes : dict or None, optional
        For recombinant mode, mapping
        ``(chrom, bin_idx) -> {GenotypeKey_overall -> GenotypeKey_positional}``
        where the positional ``GenotypeKey`` holds the two haplotypes present
        at that bin for the overall recombinant.

    Attributes
    ----------
    genotypes : list of GenotypeKey
        List of genotype keys.
    mode : {'founder', 'recombinant'}
        Mode of this set.
    positional_genotypes : dict or None
        Haplotypes at each genomic position when available.
    """

    def __init__(self, genotypes, mode='founder', positional_genotypes=None):
        genotypes_norm = []
        seen_genotypes_unordered = set()
        seen_names = set()
        for g in genotypes:
            if isinstance(g, str):
                g = GenotypeKey.from_str(g)
            elif isinstance(g, tuple):
                g = GenotypeKey(g)
            if isinstance(g, GenotypeKey):
                if mode == 'founder' and g.geno_set in seen_genotypes_unordered:
                    raise ValueError(
                        'Duplicate genotypes detected - reciprocal crossing combinations may have been supplied'
                    )
                if g.name is not None:
                    if g.name in seen_names:
                        raise ValueError(
                            f'Duplicate genotype name detected: {g.name}'
                        )
                    seen_names.add(g.name)
                genotypes_norm.append(g)
                seen_genotypes_unordered.add(g.geno_set)
            else:
                raise ValueError('genotypes should be a list of strings, tuples or GenotypeKeys')
        self.genotypes = genotypes_norm
        self.mode = mode
        self.positional_genotypes = positional_genotypes
        self._geno_dict = {
            g.name: g for g in self.genotypes if g.name is not None
        }
        self.idx = {g : i for i, g in enumerate(self.genotypes)}

    def __len__(self):
        return len(self.genotypes)

    def __iter__(self):
        return iter(self.genotypes)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._geno_dict[key]
        raise KeyError(key)

    def __contains__(self, key):
        if isinstance(key, GenotypeKey):
            return key in self.genotypes
        if isinstance(key, str):
            return key in self._geno_dict
        return NotImplemented

    @classmethod
    def from_haplotypes(cls, haplotypes):
        """
        Build all pairwise founder genotypes from a collection of haplotype tags.

        Parameters
        ----------
        haplotypes : Iterable[str]
            Unique haplotype tag strings.

        Returns
        -------
        GenotypesSet
            Set in founder mode containing all 2-combinations of the haplotypes.
        """
        return cls(list(it.combinations(haplotypes, r=2)), mode='founder')

    @classmethod
    def from_recombinant_parental_haplotypes(cls, recombinant_parental_haplotypes):
        """
        Build a recombinant-mode ``GenotypesSet`` from two parental
        ``PredictionRecords`` objects (one per parental side).

        Parameters
        ----------
        recombinant_parental_haplotypes : tuple
            A pair ``(left_parent, right_parent)`` of
            ``coelsch.records.PredictionRecords`` instances. Both must share
            identical barcode sets and genome binning. Each supplies founder
            genotypes per barcode and their positional assignment along the genome.

        Returns
        -------
        GenotypesSet
            In recombinant mode with ``positional_genotypes`` populated:
            for each ``(chrom, bin_idx)`` and overall recombinant genotype, the
            positional founder pair at that bin.

        Raises
        ------
        ValueError
            If the barcode/sample sets between the two parents do not match.
        """
        parental_hap1, parental_hap2 = recombinant_parental_haplotypes
        if sorted(parental_hap1.barcodes) != sorted(parental_hap2.barcodes):
            raise ValueError('Sample/Barcode names in recombinant_parental_haplotypes must match exactly')
        if (parental_hap1.chrom_sizes != parental_hap2.chrom_sizes) or (parental_hap1.bin_size != parental_hap2.bin_size):
            raise ValueError('Chromosomes and bin sizes in recombinant_parental_haplotypes must match exactly')
        if ('genotypes' not in parental_hap1.metadata) or ('genotypes' not in parental_hap2.metadata):
            raise ValueError('Both recombinant_parental_haplotypes objects require a genotypes metadata slot')
        genotypes = []
        for geno_name in parental_hap1.barcodes:
            # get the founder genotypes of the F1* parents
            hap1_geno = GenotypeKey.from_str(parental_hap1.metadata['genotypes'][geno_name])
            hap2_geno = GenotypeKey.from_str(parental_hap2.metadata['genotypes'][geno_name])
            # cross these to make an F1 parental genotype
            genotypes.append(hap1_geno.cross(hap2_geno, name=geno_name))

        # create with placeholder positional_genotypes
        genotypes = cls(genotypes, mode='recombinant', positional_genotypes=None)

        # create the positional_genotypes dict of (chrom, pos) -> GenotypeKey (overall) -> GenotypeKey (positional)
        positional_genotypes = defaultdict(dict)
        for geno_name, chrom, hap1 in parental_hap1.deep_items():
            hap2 = parental_hap2[geno_name, chrom]
            geno = genotypes[geno_name]
            for bin_idx, (p1, p2) in enumerate(zip(hap1, hap2)):
                positional_genotypes[(chrom, bin_idx)][geno] = GenotypeKey((
                    geno[0][int(p1 >= 0.5)], # hap1 or hap2 of left side
                    geno[1][int(p2 >= 0.5)], # hap1 or hap2 of right side
                ))
        # fill placeholder
        genotypes.positional_genotypes = positional_genotypes
        return genotypes

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
            If ``mode='founder'``: ``{GenotypeKey -> GenotypeKey}`` (identity).
            If ``mode='recombinant'``: ``{GenotypeKey_overall -> GenotypeKey_positional}``
            for the requested bin.
        """
        if self.mode == 'founder':
            return {
                geno: geno for geno in self.genotypes
            }
        else:
            if self.positional_genotypes is None:
                raise ValueError('mode == "recombinant" but positional_genotypes are not set')
            return self.positional_genotypes[(chrom, bin_idx)]


def em_assign(cb_markers, genotype_options, max_iter=100, min_delta=1e-3, error_rate_prior=0.01):
    """
    Estimate genotype probabilities for a single barcode using the Expectation-Maximization (EM) algorithm
    under a genotype-level error model.

    Each observed marker contributes support to a *group* (frozenset) of
    candidate genotypes (those consistent with the marker). Within a group,
    we divide the count equally among its members. Markers that support an
    empty group are counted as nonmatches against all genotypes.

    The EM algorithm iteratively updates genotype posterior probabilities and the global error rate until
    convergence or a maximum number of iterations is reached.

    Parameters
    ----------
    cb_markers : dict[int, int]
        Dictionary where keys are bitmasks encoding the genotypes supported by a marker,
        and values are the counts of such markers observed for the barcode.
    genotype_options : GenotypeSet
        GenotypeSet wrapper of list of candidate genotypes to evaluate. Each genotype is a GenotypeKey.
    max_iter : int, optional
        Maximum number of EM iterations. Default is 100.
    min_delta : float, optional
        Minimum change in genotype probabilities plus error rate for convergence. Default is 1e-3.
    error_rate_prior : float, optional
        Initial value for the error rate parameter. Default is 0.01.

    Returns
    -------
    probs : ndarray of shape (n_genotypes,)
        Posterior probabilities for each genotype in `genotypes`, in the same order.
    error_rate : float
        Estimated global error rate, i.e., the expected proportion of markers that do not match
        either haplotype of the true genotype.

    Notes
    -----
    - The model does not account for linkage or crossover structure; markers are assumed independent 
      given the genotype
    - Error rate is constrained to avoid numerical instability.
    - Convergence is determined by the sum of absolute changes in genotype probabilities and error rate.
    """
    n = len(genotype_options)
    probs = np.ones(n) / n  # genotype priors
    error_rate = error_rate_prior
    eps = 1e-12

    geno_matches = np.zeros(n)
    geno_nonmatches = np.zeros(n)
    all_idx = np.arange(n)
    tot = 0
    for geno_mask, count in cb_markers.items():
        tot += count
        ln_grp = geno_mask.bit_count()
        if ln_grp == 0:
            geno_nonmatches += count / n
            continue

        # vectorised: find indices of set bits
        match_idx = np.fromiter(
            (i for i in range(n) if geno_mask & (1 << i)), dtype=int
        )
        geno_matches[match_idx] += count / ln_grp

        if ln_grp != n:
            # add to all then subtract from matches
            nonmatch_weight = count / (n - ln_grp)
            geno_nonmatches += nonmatch_weight
            geno_nonmatches[match_idx] -= nonmatch_weight

    for _ in range(max_iter):
        error_rate = np.clip(error_rate, eps, 1 - eps)
        logh = np.log(1 - error_rate)
        loge = np.log(error_rate)

        geno_ll = geno_matches * logh + geno_nonmatches * loge
        max_log = np.max(geno_ll)
        geno_ll = np.exp(geno_ll - max_log)

        numerators = probs * geno_ll
        denom = numerators.sum()

        if denom <= eps:
            probs = np.ones(n) / n
            error_rate = error_rate_prior
            break

        post_probs = numerators / denom

        exp_nonmatches = (geno_nonmatches * post_probs).sum()
        exp_matches = (geno_matches * post_probs).sum()
        post_error_rate = exp_nonmatches / max(exp_matches + exp_nonmatches, eps)

        delta = np.abs(post_probs - probs).sum() + abs(post_error_rate - error_rate)
        error_rate = np.clip(post_error_rate, eps, 1.0 - eps)
        probs = post_probs

        if delta < min_delta:
            break

    return probs, error_rate


def random_resample_geno_markers(cb_geno_markers, n_resamples, rng, max_sample_size=1000):
    """
    Randomly resample genotype markers with replacement.

    Parameters
    ----------
    cb_geno_markers : dict
        A dictionary where keys are bitmasks encoding the genotypes supported by a marker,
        and values are the counts of such markers observed for the barcode.
    n_resamples : int
        The number of resamples to perform.
    rng : np.random.Generator
        Random number generator for reproducibility.
    max_sample_size : int, optional
        Upper bound on total draws per resample (default ``1000``). Uses
        ``min(total_count, max_sample_size)``

    Yields
    ------
    Counter
        A resampled set of genotype markers with counts, sampled with replacement.
    """
    genos = list(cb_geno_markers.keys())
    counts = np.array(list(cb_geno_markers.values()))
    tot = counts.sum()
    sample_size = min(tot, max_sample_size)
    p = counts / tot
    for _ in range(n_resamples):
        idx = rng.choice(np.arange(len(p)), size=sample_size, replace=True, p=p)
        yield Counter(genos[i] for i in idx)

        
class _GenotypingWorker:
    """
    Callable worker class to pickle/store genotype_options once per worker.
    """
    def __init__(self, func, genotype_options, **kwargs):
        self.func = func
        self.genotype_options = genotype_options
        self.kwargs = kwargs

    def __call__(self, chunk, **kwargs):
        self.kwargs.update(kwargs)
        return self.func(chunk, self.genotype_options, **self.kwargs)


def assign_genotype_with_em(geno_markers, genotype_options, *,
                            max_iter=1000, min_delta=1e-3, n_bootstraps=25,
                            rng=DEFAULT_RNG):
    """
    Assign a genotype to a cell barcode using Expectation-Maximization (EM) and bootstrap re-sampling.

    Parameters
    ----------
    geno_markers : dict[str, dict[int, int]]
        Observed counts per supported-genotype group for each barcode. cb -> geno -> counts mapping.
    genotype_options : GenotypeSet
        GenotypeSet of genotype combinations to consider for the EM algorithm.
    max_iter : int, optional
        The maximum number of iterations for the EM algorithm (default is 1000).
    min_delta : float, optional
        The minimum change in probabilities between iterations to stop the algorithm (default is 1e-3).
    n_bootstraps : int, optional
        The number of bootstrap samples to use for estimating genotype probabilities (default is 25).
    rng : np.random.Generator, optional
        Random number generator for reproducibility (default is `DEFAULT_RNG`).

    Returns
    -------
    geno_assignments : dict[str, GenotypeKey]
        The assigned genotype.
    geno_probabilities : dict[str, float]
        The probability of the assigned genotype.
    geno_nmarkers : int
        The number of markers used for genotyping.
    geno_error_rate : float
        Mean estimated error rate across bootstraps.
    """
    n_genos = len(genotype_options)
    geno_assignments = {}
    geno_probabilities = {}
    geno_nmarkers = {}
    geno_error_rate = {}
    for cb, cb_geno_markers in geno_markers.items():
        prob_bootstraps = []
        error_rate_bootstraps = []
        for marker_sample in random_resample_geno_markers(cb_geno_markers, n_bootstraps, rng):
            probs, error_rate = em_assign(
                marker_sample,
                genotype_options,
                max_iter=max_iter,
                min_delta=min_delta,
            )
            prob_bootstraps.append(probs)
            error_rate_bootstraps.append(error_rate)
        probs = {geno: p for geno, p in
                 zip(genotype_options, np.mean(prob_bootstraps, axis=0))}
        max_prob = max(probs.values())
        # when two or more genotypes are equally likely, select on at random
        genos_with_max_prob = [geno for geno, p in probs.items() if np.isclose(p, max_prob, atol=min_delta)]
        genos_with_max_prob.sort(key=lambda g: str(g))
        geno_assignments[cb] = rng.choice(genos_with_max_prob)
        geno_probabilities[cb] = float(max_prob)
        geno_nmarkers[cb] = int(sum(cb_geno_markers.values()))
        geno_error_rate[cb] = float(np.mean(error_rate_bootstraps))
    return geno_assignments, geno_probabilities, geno_nmarkers, geno_error_rate


def parallel_assign_genotypes(genotype_markers, genotype_options, *, processes=1, rng=DEFAULT_RNG, **kwargs):
    """
    Assign genotypes to multiple cell barcodes in parallel using EM and bootstrap sampling.

    Parameters
    ----------
    genotype_markers : dict[str, dict[int, int]]
        A dictionary where keys are cell barcodes and values are counters of haplotype markers 
        for each barcode.
    genotype_options : GenotypesSet
        Candidate genotype set (shared across barcodes).
    processes : int, optional
        The number of processes to use for parallel computation (default is 1).
    rng : np.random.Generator, optional
        Random number generator for reproducibility (default is `DEFAULT_RNG`).
    **kwargs : keyword arguments
        Additional parameters passed to the `assign_genotype_with_em` function.

    Returns
    -------
    geno_assignments : coelsch.records.NestedData
        The assigned genotype for each cell barcode
    geno_probabilities : coelsch.records.NestedData
        The probability that the assignment is correct, calculated using EM
    geno_nmarkers : coelsch.records.NestedData
        The number of markers that were used in genotype assignment
    geno_error_rates : coelsch.records.NestedData
        Mean background/error rate per barcode (``dtype=float``).
    """
    n_cb = len(genotype_markers)
    # use sorted cb list to ensure deterministic results across runs
    barcodes = sorted(genotype_markers)

    barcode_chunks = weighted_chunks(
        barcodes,
        weight_fn=lambda cb: sum(genotype_markers[cb].values()),
        processes=processes,
        oversubscription=10
    )

    worker = _GenotypingWorker(assign_genotype_with_em, genotype_options, **kwargs)
    res = Parallel(processes)(
        delayed(worker)({cb: genotype_markers[cb] for cb in cb_chunk}, rng=sp_rng)
        for cb_chunk, sp_rng in zip(barcode_chunks, spawn_child_rngs(rng))
    )
    geno_assignments = NestedData(levels=('cb',), dtype=GenotypeKey)
    geno_probabilities = NestedData(levels=('cb',), dtype=float)
    geno_nmarkers = NestedData(levels=('cb',), dtype=int)
    geno_error_rates = NestedData(levels=('cb',), dtype=float)

    for ga, gp, nm, er in res:
        geno_assignments.update(ga)
        geno_probabilities.update(gp)
        geno_nmarkers.update(nm)
        geno_error_rates.update(er)

    return geno_assignments, geno_probabilities, geno_nmarkers, geno_error_rates


def _parallel_convert_ic(inv_counts, genotype_options, threshold):
    genotype_markers = defaultdict(Counter)
    for ic in inv_counts:
        pos_parental_geno = genotype_options.get_bin_haplotypes(ic.chrom, ic.bin_idx)
        for cb, cb_counts in ic.counts.items():
            cb_pos_geno_markers = Counter()
            for hap_comb, count in cb_counts.items():
                supported_genotypes = 0
                for geno, pos_geno in pos_parental_geno.items():
                    if pos_geno & hap_comb:
                        supported_genotypes |= 1 << genotype_options.idx[geno]
                cb_pos_geno_markers[supported_genotypes] += min(count, threshold)
            genotype_markers[cb] += cb_pos_geno_markers
    return genotype_markers


def _calculate_ic_upper_threshold(inv_counts, threshold_percentile=95):
    count_values = np.fromiter(it.chain.from_iterable(ic.deep_values() for ic in inv_counts), dtype=int)
    return int(np.percentile(count_values, threshold_percentile))


def resolve_inv_counts_to_genotype_markers(inv_counts, genotype_options,
                                           threshold_percentile=95, processes=1):
    """
    Convert per-interval haplotype counts into genotype-support groups.

    For each interval and barcode, each haplotype-support combination is
    mapped to the set of candidate genotypes whose positional haplotypes at
    that bin intersect the observed haplotype combination. Counts are then
    aggregated over intervals.

    Parameters
    ----------
    inv_counts : list[IntervalMarkerCounts]
        Interval-wise haplotype marker counts by barcode.
    genotype_options : GenotypesSet
        Candidate genotype set; used to look up positional haplotypes per bin.
    processes : int, optional
        The number of processes to use for parallel computation (default is 1).

    Returns
    -------
    dict[str, collections.Counter]
        Mapping ``cb -> Counter({frozenset[GenotypeKey]: count, ...})``.
    """

    t = _calculate_ic_upper_threshold(inv_counts, threshold_percentile)
    ic_chunks = weighted_chunks(
        inv_counts,
        weight_fn=lambda ic: len(ic.counts),
        processes=processes,
        oversubscription=2,
    )

    genotype_markers = defaultdict(Counter)
    worker = _GenotypingWorker(_parallel_convert_ic, genotype_options, threshold=t)
    results = Parallel(processes, backend='loky')(
        delayed(worker)(ic_chunk) for ic_chunk in ic_chunks
    )
    for chunk_markers in results:
        for cb, cb_pos_geno_markers in chunk_markers.items():
            genotype_markers[cb] += cb_pos_geno_markers

    return genotype_markers


def resolve_inv_counts_to_co_markers(inv_counts, genotypes, genotype_options):
    """
    Resolve haplotype counts for a set of barcodes into a format compatible with crossover calling

    Parameters
    ----------
    inv_counts : list[IntervalMarkerCounts]
        Interval-wise haplotype marker counts by barcode.
    genotypes : coelsch.records.NestedData or dict
        Assigned genotype per barcode (``GenotypeKey`` or string convertible
        to ``GenotypeKey``).
    genotype_options : GenotypesSet
        Candidate set providing positional haplotypes per bin.

    Returns
    -------
    list of IntervalMarkerCounts
        A list of resolved `IntervalMarkerCounts` objects with markers uniquely identifying the two
        haplotypes of the assigned genotype for each cell barcode.
    """
    resolved_inv_counts = []
    for ic in inv_counts:
        resolved_ic = IntervalMarkerCounts.new_like(ic)
        pos_parental_geno = genotype_options.get_bin_haplotypes(ic.chrom, ic.bin_idx)
        for cb, cb_haplo_ic in ic.counts.items():
            try:
                geno = genotypes[cb]
            except KeyError:
                # cb was removed due to low markers
                continue
            geno_hap1, geno_hap2 = pos_parental_geno[geno].genotype
            if geno_hap1 == geno_hap2:
                # barcode has the same haplotype on both sides of the cross at this position
                # this could occur e.g. in an selfed F2 genotype. crossovers cannot be resolved at this bin
                continue
            for haps, count in cb_haplo_ic.items():
                if geno_hap1 in haps:
                    if geno_hap2 not in haps:
                        resolved_ic[cb][0] += count
                elif geno_hap2 in haps:
                    resolved_ic[cb][1] += count
        resolved_inv_counts.append(resolved_ic)
    return resolved_inv_counts


def genotype_from_inv_counts(inv_counts, recombinant_mode=False, recombinant_parental_haplotypes=None,
                             min_markers_per_cb=100, crossing_combinations=None, all_haplotypes=None,
                             processes=1, **kwargs):
    """
    Generate genotypes and crossover markers from interval counts using EM and bootstrap sampling.

    Parameters
    ----------
    inv_counts : list of IntervalMarkerCounts
        A list of `IntervalMarkerCounts` objects, each representing counts of markers
        across different cell barcodes.
    recombinant_mode : bool, optional
        If ``True``, parental genotypes are themselves recombinants whose
        positional haplotypes are supplied in ``recombinant_parental_haplotypes``.
    recombinant_parental_haplotypes : tuple or None, optional
        If ``recombinant_mode`` is ``True``, a 2-tuple of file paths to
        ``coelsch.records.PredictionRecords`` JSONs **or** already-loaded
        ``PredictionRecords`` objects; these describe the two parental sides.
    min_markers_per_cb : int, optional
        The minimum number of markers required for a barcode to be considered (default is 100).
    crossing_combinations : Iterable, optional
        In founder mode, optional whitelist of allowed founder pairs
        (e.g., ``[('A','B'), ('A','C'), ...]``). If ``None``, use all
        pairwise combinations of ``all_haplotypes``.
    all_haplotypes : Iterable[str] or None, optional
        Required if ``crossing_combinations`` is ``None`` in founder mode.
    **kwargs : keyword arguments
        Additional parameters passed to the `parallel_assign_genotypes` and `assign_genotype_with_em`
        functions.

    Returns
    -------
    genotypes : coelsch.records.NestedData
        Assigned genotype per barcode (as string form of ``GenotypeKey``).
    genotype_probs : coelsch.records.NestedData
        Posterior probability of the assigned genotype per barcode.
    genotype_nmarkers : coelsch.records.NestedData
        Number of markers used per barcode.
    genotype_error_rates : coelsch.records.NestedData
        Mean background/error rate per barcode.
    inv_counts : list[IntervalMarkerCounts]
        Interval counts collapsed to crossover markers for the assigned genotypes.
    """
    if not recombinant_mode or recombinant_parental_haplotypes is None:
        # founder mode - all parental genotypes are F1 hybrids of "founder" haplotype tags in bam file
        if crossing_combinations is None:
            # use all pairwise combinations of haplotypes
            if all_haplotypes is None:
                raise ValueError('need all_haplotypes when crossing_combinations is None')
            genotype_options = GenotypesSet.from_haplotypes(all_haplotypes)
        else:
            # limit genotypes to user specified combinations of haplotypes
            genotype_options = GenotypesSet(crossing_combinations, mode='founder')
    else:
        # recombinant mode - parental genotypes are F1 hybrids of F1 hybrids
        # i.e. themselves recombinants of the haplotype tags in bam file
        recombinant_parental_haplotypes = [
            PredictionRecords.read_json(fn) for fn in recombinant_parental_haplotypes
        ]
        genotype_options = GenotypesSet.from_recombinant_parental_haplotypes(recombinant_parental_haplotypes)

    log.info(f'Performing genotyping in {genotype_options.mode} mode with {len(genotype_options)} possible genotypes')

    # use genotype options to infer total support for each genotype for each barcode
    genotype_markers = resolve_inv_counts_to_genotype_markers(
        inv_counts, genotype_options, processes=min(processes, 10)
    )
    log.debug(f'Resolved interval bin counts to genotyping markers for {len(genotype_markers)} barcodes')

    if min_markers_per_cb > 0:
        genotype_markers = {cb: g for cb, g in genotype_markers.items()
                            if sum(g.values()) >= min_markers_per_cb}
        log.debug(f'Pre-filtered barcodes by marker count, retaining {len(genotype_markers)} barcodes')
    (genotypes, genotype_probs,
     genotype_nmarkers, genotype_error_rates) = parallel_assign_genotypes(
         genotype_markers, genotype_options=genotype_options, processes=processes, **kwargs
     )
    inv_counts = resolve_inv_counts_to_co_markers(inv_counts, genotypes, genotype_options)
    # convert genotype data from frozenset to str
    genotypes= NestedData(
        levels=('cb',),
        dtype=str,
        data={cb: str(geno) for cb, geno in genotypes.items()}
    )
    return genotypes, genotype_probs, genotype_nmarkers, genotype_error_rates, inv_counts
