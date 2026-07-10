from collections import defaultdict
import numpy as np
from scipy.ndimage import convolve1d

from .records import NestedDataArray
from .defaults import DEFAULT_RANDOM_SEED


DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


def nonzero_range(arr, axis=-1):
    """
    Returns a boolean mask indicating the range of nonzero values along a given axis.

    This function identifies the continuous range of nonzero values along a specified axis
    by using forward and reverse cumulative logical OR operations. It returns `True` for positions
    where there are nonzero values at or flanking the position on both sides, and `False` elsewhere.

    Parameters
    ----------
    arr : ndarray
        Input array of any shape, containing the data to check for nonzero values.
    axis : int, optional
        The axis along which to identify nonzero ranges. The default is the last axis (-1).

    Returns
    -------
    ndarray
        A boolean mask of the same shape as the input array, with `True` where there are nonzero
        values at or flanking positions on both sides and `False` otherwise.

    Example
    -------
    Consider the following 1D array:

    >>> arr = np.array([0, 1, 0, 3, 0, 0, 4, 0])

    Applying `nonzero_range(arr)` will return a boolean mask for the nonzero values:

    >>> nonzero_range(arr)
    array([False,  True, False,  True, False, False,  True, False])

    The mask indicates the range of nonzero values along the array, marking `True` where nonzero
    values occur.
    """
    arr = arr.astype(bool)
    fwd = np.logical_or.accumulate(arr, axis=axis)
    rev = np.flip(
        np.logical_or.accumulate(
            np.flip(arr, axis=axis),
            axis=axis
        ),
        axis=axis
    )
    return np.logical_and(fwd, rev)


def calculate_cm_denominator(co_markers, apply_per_geno):
    """
    Calculate the denominator for recombination rate calculations based on marker data.

    This function calculates the denominators for each chromosome, which are used to scale
    the recombination landscape during computation. The denominators are based on the presence
    of markers flanking each position, calculated using the `nonzero_range` function.

    Parameters
    ----------
    co_markers : MarkerRecords
        Marker records object containing marker matrices per barcode and chromosome.

    Returns
    -------
    dict
        A dictionary where the keys are chromosome names and the values are the denominators
        for recombination calculations for each chromosome.
    """
    denom = NestedDataArray(levels=('genotype', 'chrom'))
    for geno, geno_co_markers in co_markers.groupby(by='genotype' if apply_per_geno else 'none'):
        for chrom in co_markers.chrom_sizes:
            denom[geno, chrom] = nonzero_range(
                geno_co_markers[:, chrom].stack_values().sum(axis=-1),
                axis=1
            ).astype(np.float32)
    return denom


def recombination_landscape(co_preds,
                            co_markers=None,
                            apply_per_geno=True,
                            rolling_mean_window_size=1_000_000,
                            nboots=100,
                            min_prob=5e-3,
                            rng=DEFAULT_RNG):
    """
    Calculate the recombination landscape for a PredictionRecords dataset.

    This function estimates the recombination rate per megabase for each chromosome by calculating
    the gradient of haplotype probabilities across bins, and bootstrapping the values to assess
    the uncertainty in the landscape. If a co_markers dataset is provided, it uses it to calculate 
    denominators to scale the recombination rate at chromosome ends, where lack of markers may
    limit sensitivity.

    Parameters
    ----------
    co_preds : PredictionRecords
        PredictionRecords object containing the haplotype predictions.
    co_markers : MarkerRecords, optional
        Marker records object used to scale the recombination rate. If None, no scaling is performed.
    apply_per_geno : bool, optional
        Whether to group barcodes by genotype for recombination landscape calculation. Default is True.
    rolling_mean_window_size : int, optional
        The size of the window for the rolling mean filter (default is 1,000,000).
    nboots : int, optional
        The number of bootstrap samples to draw for uncertainty estimation (default is 100).
    min_prob : float, optional
        The minimum threshold for crossover prediction gradients (default is 5e-3).
    rng : np.random.Generator, optional
        A random number generator for bootstrapping (default is `DEFAULT_RNG`).

    Returns
    -------
    dict
        A dictionary where the keys are chromosome names and the values are arrays of recombination
        rates per megabase, calculated from bootstrapped samples.

    Raises
    ------
    ValueError
        If the barcodes of the co-predictions and co-markers do not match.
    """
    nf = 1_000_000 // co_preds.bin_size
    ws = rolling_mean_window_size // co_preds.bin_size
    filt = np.ones(ws) / ws

    if co_markers is not None:
        if co_preds.barcodes != co_markers.barcodes:
            raise ValueError('Cell barcodes from marker-json-fn and predict-json-fn do not match')
        denominators = calculate_cm_denominator(co_markers, apply_per_geno)
    else:
        denominators = None

    cm_per_mb = NestedDataArray(levels=('genotype', 'chrom',))
    for geno, geno_co_preds in co_preds.groupby(by='genotype' if apply_per_geno else 'none'):
        N = len(geno_co_preds)
        for chrom, nbins in geno_co_preds.nbins.items():
            chrom_hap_probs = geno_co_preds[:, chrom].stack_values()
            chrom_co_probs = np.abs(np.diff(
                chrom_hap_probs,
                n=1,
                axis=1,
                prepend=chrom_hap_probs[:, 0].reshape(-1, 1)
            ))
            # filter gradients smaller than min_prob
            chrom_co_probs = np.where(
                chrom_co_probs >= min_prob, chrom_co_probs, 0
            )
            if co_preds.ploidy_type.startswith('diploid'):
                chrom_co_probs *= 2
            # filter gradients where there are no markers
            if denominators is not None:
                chrom_denom = denominators[geno, chrom]
                chrom_denom = convolve1d(
                    chrom_denom, filt,
                    axis=1,
                    mode='constant',
                    cval=0
                )
                chrom_co_probs = np.where(
                    chrom_denom > 0,
                    chrom_co_probs,
                    0,
                )
            else:
                chrom_denom = np.ones(shape=(N, nbins))
            # equivalent to rolling sum accounting for edge effects
            chrom_co_probs = convolve1d(
                chrom_co_probs, filt,
                axis=1,
                mode='constant',
                cval=0
            ) * nf
            chrom_cm_per_mb = []
            for _ in range(nboots):
                idx = rng.integers(0, N, size=N)
                chrom_cm_per_mb.append(
                    (chrom_co_probs[idx].sum(axis=0) / chrom_denom[idx].sum(axis=0)) * 100
                )
            cm_per_mb[geno, chrom] = np.stack(chrom_cm_per_mb)
    co_preds.add_metadata(recombination_landscape=cm_per_mb)
    return cm_per_mb


def _distances_observed(crossover_samples, only_adjacent=False):
    """
    Compute observed crossover–crossover distances.

    Parameters
    ----------
    crossover_samples : dict[str, list[np.ndarray]]
        Dictionary mapping chromosome IDs to lists of 1D arrays. Each array
        contains sorted crossover positions for one sample on that chromosome.
        All chromosomes should have the same number of samples.
    only_adjacent : bool, optional
        If True, compute only distances between adjacent crossovers.
        If False, compute all pairwise distances per sample.

    Returns
    -------
    obs : ndarray
        Flattened array of all observed distances across all chromosomes
        and all samples.
    obs_pairs : float
        Mean number of observed distance pairs per sample per chromosome,
        computed as ``len(obs) / (n_samples * n_chroms)``.
    """
    obs = {}
    obs_pairs = {}
    for chrom, chrom_samples in crossover_samples.items():
        n_samples = len(chrom_samples)
        chrom_obs = []
        for samp in chrom_samples:
            if len(samp) < 2:
                continue
            if only_adjacent:
                chrom_obs.append(np.diff(samp))
            else:
                chrom_obs.append((samp[None, :] - samp[:, None])[np.triu_indices(len(samp), 1)])
        obs[chrom] = np.concatenate(chrom_obs) if chrom_obs else np.array([], dtype=float)
        obs_pairs[chrom] = len(obs[chrom]) / n_samples
    return obs, obs_pairs


def _distances_expected(crossover_samples, max_pairs=200_000, only_adjacent=False, rng=DEFAULT_RNG):
    """
    Generate expected crossover–crossover distances under an independence model.

    Parameters
    ----------
    crossover_samples : dict[str, list[np.ndarray]]
        Dictionary mapping chromosome IDs to lists of 1D arrays of crossover
        positions. Used to estimate empirical position distributions.
    max_pairs : int, optional
        Maximum number of independent position pairs to sample when
        ``only_adjacent=False``.
    only_adjacent : bool, optional
        If True, simulate adjacent distances by drawing Poisson-distributed
        crossover counts per sample and resampling positions from the pooled
        empirical distribution. If False, generate absolute differences from
        independent draws with replacement from pooled positions.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    exp : ndarray
        Array of simulated expected distances (zero distances removed).
    exp_pairs : float
        Expected number of distance pairs per sample, computed as
        ``(mean_co ** 2) / 2`` where ``mean_co`` is the mean number of
        crossovers per chromosome across all samples/chromosomes.
    """
    
    exp = {}
    exp_pairs = {}
    cos_per_chrom = np.mean([len(samp) for chrom_samples in crossover_samples.values() for samp in chrom_samples])
    for chrom, chrom_samples in crossover_samples.items():
        n_samples = len(chrom_samples)
        lam = np.mean([len(o) for o in chrom_samples])
        chrom_samples = np.concatenate(chrom_samples)
        chrom_exp = []
        if only_adjacent:
            for n_co in rng.poisson(lam, size=n_samples):
                if n_co > 1:
                    q = np.sort(rng.choice(chrom_samples, size=n_co, replace=True))
                    chrom_exp.append(np.diff(q))
        else:
            # independent pairs
            m = min(max_pairs, chrom_samples.size * chrom_samples.size)
            a = rng.choice(chrom_samples, size=m, replace=True)
            b = rng.choice(chrom_samples, size=m, replace=True)
            chrom_exp.append(np.abs(b - a))
        chrom_exp = np.concatenate(chrom_exp) if chrom_exp else np.array([], dtype=float)
        exp[chrom] = chrom_exp[chrom_exp > 0]
        exp_pairs[chrom] = (lam ** 2) / 2.0
    return exp, exp_pairs


def _coc_curve_sample(crossover_samples, bins, chrom_nbins, only_adjacent=False, rng=DEFAULT_RNG):
    """
    Compute one coefficient-of-coincidence curve for a bootstrap sample.

    Parameters
    ----------
    crossover_samples : dict[str, list[np.ndarray]]
        Mapping of chromosome IDs to lists of crossover-position arrays.
    bins : array_like
        Bin edges at which distances are histogrammed.
    only_adjacent : bool, optional
        Use only adjacent distances for both observed and expected
        calculations.
    rng : numpy.random.Generator, optional
        Random number generator passed to expected-distance generation.

    Returns
    -------
    bin_mids : ndarray
        Midpoints of histogram bins.
    coc : ndarray
        Coefficient of coincidence per bin, computed as the ratio of
        normalised observed to normalised expected histogram counts.
    """
    obs, obs_pairs = _distances_observed(crossover_samples, only_adjacent=only_adjacent)
    exp, exp_pairs = _distances_expected(crossover_samples, only_adjacent=only_adjacent, rng=rng)
    bin_mids = {}
    coc = {}
    Lint = {}
    for chrom in obs:
        obs_h, edges = np.histogram(obs[chrom], bins=bins[chrom])
        exp_h, _ = np.histogram(exp[chrom], bins=bins[chrom])
        frac_obs_pairs = obs_pairs[chrom] / exp_pairs[chrom]
        exp_h = exp_h / len(exp[chrom])
        obs_h = frac_obs_pairs * obs_h / len(obs[chrom])

        with np.errstate(divide='ignore', invalid='ignore'):
            coc[chrom] = np.where(exp_h > 0, obs_h / exp_h, np.nan)
        L = chrom_nbins[chrom]
        d_obs = np.mean(obs[chrom]) / chrom_nbins[chrom]
        d_exp = np.mean(exp[chrom]) / chrom_nbins[chrom]
        if len(obs[chrom]) > 0:
            Lint[chrom] = (d_obs - 1) * frac_obs_pairs + 1 - d_exp
        elif len(exp[chrom]) > 0:
            Lint[chrom] =  1 - d_exp
        else:
            Lint[chrom] = 1
        bin_mids[chrom] = 0.5 * (edges[:-1] + edges[1:])
    return bin_mids, coc, Lint


def coefficient_of_coincidence(co_preds, nboots=100, min_dist=None, max_dist=None, step_size=1e6,
                               chroms=None, only_adjacent=False, rng=DEFAULT_RNG):
    """
    Bootstrap the coefficient-of-coincidence curve across samples.

    Parameters
    ----------
    co_preds : PredictionRecords
        haplotype predictions object with metadata slot crossover_samples
    nboots : int, optional
        Number of bootstrap replicates.
    min_dist : int or None, optional
        Minimum physical distance (in bp) for histogramming. If None, use the rfactor of the model
        used to make the haplotype predictions
    max_dist : int or None, optional
        Maximum physical distance (in bp). If None, use the chromosome length.
    step_size : int, optional
        Histogram step size in bp.
    chroms : iterable or None, optional
        Chromosome IDs to include. If None, include all chromosomes.
    only_adjacent : bool, optional
        Whether to use only adjacent distances when computing observed and
        expected distributions.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    bootstrap_coc : ndarray, shape (nboots, nbins - 1)
        Bootstrap samples of the coefficient-of-coincidence curve.
    mids : ndarray
        Bin midpoints expressed in base-pair coordinates.
    """
    nbarcodes = len(co_preds)
    bin_size = co_preds.bin_size
    if chroms is None:
        chroms = set(co_preds.chrom_sizes)
    if min_dist is None:
        min_dist = co_preds.metadata['rhmm_params']['rfactor']
    else:
        min_dist = min_dist // bin_size
    if max_dist is None:
        max_dist = co_preds.nbins
    else:
        max_dist = {chrom: max_dist // bin_size for chrom in chroms}
    step_size = step_size // bin_size
    bins = {
        chrom: np.arange(int(min_dist), int(max_dist[chrom]), int(step_size))
        for chrom in chroms
    }
    try:
        co_pos_samples = co_preds.metadata['crossover_samples']
    except KeyError:
        raise ValueError('co_preds object lacking crossover_samples data')
    barcodes = co_pos_samples.get_level_keys('cb')
    sample_ids = co_pos_samples.get_level_keys('sample')
    bootstrap_coc = {
        chrom: np.empty((nboots, len(bins[chrom]) - 1), dtype=float)
        for chrom in chroms
    }
    bootstrap_Lint = {
        chrom : np.empty(nboots, dtype=float) for chrom in chroms
    }
    for i in range(nboots):
        # resample barcodes with replacement
        cb_sample = rng.choice(barcodes, size=nbarcodes, replace=True)
        samp_idx = rng.choice(sample_ids, size=nbarcodes, replace=True)
        sample = defaultdict(list)
        for cb, s in zip(cb_sample, samp_idx):
            for chrom in chroms:
                sample[chrom].append(co_pos_samples[cb, chrom, s, :, 0])
        mids, coc, Lint = _coc_curve_sample(sample, bins, max_dist, only_adjacent=only_adjacent, rng=rng)
        for chrom in chroms:
            bootstrap_coc[chrom][i] = coc[chrom]
            bootstrap_Lint[chrom][i] = Lint[chrom] * co_preds.chrom_sizes[chrom]
    mids = {chrom: m * bin_size for chrom, m in mids.items()}
    return bootstrap_coc, mids, bootstrap_Lint