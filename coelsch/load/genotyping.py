import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Literal
import itertools as it

import numpy as np

from joblib import Parallel, delayed

from .utils import weighted_chunks
from .counts import IntervalMarkerCounts
from coelsch.experiment import ExperimentalDesign
from coelsch.utils import spawn_child_rngs
from coelsch.records import PredictionRecords, NestedData
from coelsch.defaults import DEFAULT_RANDOM_SEED


log = logging.getLogger('coelsch')
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


def em_assign(cb_markers, n, max_iter=100, min_delta=1e-3, error_rate_prior=0.01):
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
    n : int
        The number of genotype possibilities to be evaluated
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
    def __init__(self, func, experimental_design, **kwargs):
        self.func = func
        self.experimental_design = experimental_design
        self.kwargs = kwargs

    def __call__(self, chunk, **kwargs):
        self.kwargs.update(kwargs)
        return self.func(chunk, self.experimental_design, **self.kwargs)


def assign_genotype_with_em(geno_markers, experimental_design, *,
                            max_iter=1000, min_delta=1e-3, n_bootstraps=25,
                            rng=DEFAULT_RNG):
    """
    Assign a genotype to a cell barcode using Expectation-Maximization (EM) and bootstrap re-sampling.

    Parameters
    ----------
    geno_markers : dict[str, dict[int, int]]
        Observed counts per supported-genotype group for each barcode. cb -> geno -> counts mapping.
    experimental_design : coelsch.experiment.ExperimentalDesign
        The experimental design of the dataset, including the possible genotypes
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
    genotype_options = experimental_design.genotypes
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
                n_genos,
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


def parallel_assign_genotypes(genotype_markers, experimental_design, *, processes=1, rng=DEFAULT_RNG, **kwargs):
    """
    Assign genotypes to multiple cell barcodes in parallel using EM and bootstrap sampling.

    Parameters
    ----------
    genotype_markers : dict[str, dict[int, int]]
        A dictionary where keys are cell barcodes and values are counters of haplotype markers 
        for each barcode.
    experimental_design : coelsch.experiment.ExperimentalDesign
        The experimental design of the dataset, including the possible genotypes
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

    worker = _GenotypingWorker(
        assign_genotype_with_em,
        experimental_design,
        **kwargs
    )
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


def _parallel_convert_ic(inv_counts, experimental_design, threshold):
    positional_genotypes = experimental_design.positional_genotypes
    genotype_markers = defaultdict(Counter)
    for ic in inv_counts:
        pos_parental_geno = positional_genotypes.get_bin_haplotypes(ic.chrom, ic.bin_idx)
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


def resolve_inv_counts_to_genotype_markers(inv_counts, experimental_design,
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
    experimental_design : coelsch.experiment.ExperimentalDesign
        The experimental design of the dataset, including the possible genotypes
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
    worker = _GenotypingWorker(
        _parallel_convert_ic,
        experimental_design,
        threshold=t
    )
    results = Parallel(processes, backend='loky')(
        delayed(worker)(ic_chunk) for ic_chunk in ic_chunks
    )
    for chunk_markers in results:
        for cb, cb_pos_geno_markers in chunk_markers.items():
            genotype_markers[cb] += cb_pos_geno_markers

    return genotype_markers


def resolve_inv_counts_to_co_markers(inv_counts, genotypes, experimental_design):
    """
    Resolve haplotype counts for a set of barcodes into a format compatible with crossover calling

    Parameters
    ----------
    inv_counts : list[IntervalMarkerCounts]
        Interval-wise haplotype marker counts by barcode.
    genotypes : coelsch.records.NestedData or dict
        Assigned genotype per barcode (``GenotypeKey`` or string convertible
        to ``GenotypeKey``).
    experimental_design : coelsch.experiment.ExperimentalDesign
        The experimental design of the dataset, including the possible genotypes

    Returns
    -------
    list of IntervalMarkerCounts
        A list of resolved `IntervalMarkerCounts` objects with markers uniquely identifying the two
        haplotypes of the assigned genotype for each cell barcode.
    """
    positional_genotypes = experimental_design.positional_genotypes
    resolved_inv_counts = []
    for ic in inv_counts:
        resolved_ic = IntervalMarkerCounts.new_like(ic)
        pos_parental_geno = positional_genotypes.get_bin_haplotypes(ic.chrom, ic.bin_idx)
        for cb, cb_haplo_ic in ic.counts.items():
            try:
                geno = genotypes[cb]
            except KeyError:
                # cb was removed due to low markers
                continue
            pg = pos_parental_geno[geno]
            if pg.n_founders < 2:
                continue

            for haps, count in cb_haplo_ic.items():
                supported = pg.founder_set.intersection(haps)

                # No support, or support for multiple genotype haplotypes, is ambiguous.
                if len(supported) != 1:
                    continue

                hap_idx = geno.get_hap_idx(supported[0])
                resolved_ic[cb][hap_idx] += count
        resolved_inv_counts.append(resolved_ic)
    return resolved_inv_counts


def genotype_from_inv_counts(inv_counts, experimental_design,
                             min_markers_per_cb=100, processes=1, **kwargs):
    """
    Generate genotypes and crossover markers from interval counts using EM and bootstrap sampling.

    Parameters
    ----------
    inv_counts : list of IntervalMarkerCounts
        A list of `IntervalMarkerCounts` objects, each representing counts of markers
        across different cell barcodes.
    experimental_design : coelsch.experiment.ExperimentalDesign
        The experimental design of the dataset, including the possible genotypes
    min_markers_per_cb : int, optional
        The minimum number of markers required for a barcode to be considered (default is 100).
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

    log.info(f'Performing genotyping in {experimental_design.genotyping_strategy} mode '
             f'with {len(experimental_design.genotypes)} possible genotypes')

    # use genotype options to infer total support for each genotype for each barcode
    genotype_markers = resolve_inv_counts_to_genotype_markers(
        inv_counts, experimental_design, processes=min(processes, 10)
    )
    log.debug(f'Resolved interval bin counts to genotyping markers for {len(genotype_markers)} barcodes')

    if min_markers_per_cb > 0:
        genotype_markers = {cb: g for cb, g in genotype_markers.items()
                            if sum(g.values()) >= min_markers_per_cb}
        log.debug(f'Pre-filtered barcodes by marker count, retaining {len(genotype_markers)} barcodes')
    (genotypes, genotype_probs,
     genotype_nmarkers, genotype_error_rates) = parallel_assign_genotypes(
         genotype_markers, experimental_design, processes=processes, **kwargs
     )
    inv_counts = resolve_inv_counts_to_co_markers(inv_counts, genotypes, experimental_design)
    # convert genotype data from frozenset to str
    genotypes= NestedData(
        levels=('cb',),
        dtype=str,
        data={cb: str(geno) for cb, geno in genotypes.items()}
    )
    return genotypes, genotype_probs, genotype_nmarkers, genotype_error_rates, inv_counts
