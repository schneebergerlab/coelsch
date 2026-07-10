import numpy as np
from pomegranate import distributions as pmd

from coelsch.records import MarkerRecords
from coelsch.predict.rhmm.dists import ZeroInflated


def _compute_bias_factor(co_markers, hap_bias_shrinkage=0.75, bc_haplotype=0):
    if co_markers.ploidy_type == 'diploid_bc1':
        expected_ratio = (3, 1) if bc_haplotype == 0 else (1, 3)
    else:
        expected_ratio = (1, 1)
    hap_totals = np.sum([m.sum(axis=0) for m in co_markers.deep_values()], axis=0)
    expected = np.array(expected_ratio, dtype=float)
    expected /= expected.sum()
    observed = hap_totals / hap_totals.sum()
    raw_factor = observed / expected
    bias_factor = 1.0 + hap_bias_shrinkage * (raw_factor - 1.0)
    return bias_factor


def normalise_bin_coverage(co_markers, shrinkage_q=0.99, allow_upweight=False, max_upweight=4.0,
                           binwise_hap_mode='shared', correct_hap_bias=True, hap_bias_shrinkage=0.75):
    """
    Normalise per-bin coverage across chromosomes by shrinking extreme coverage values.

    This function computes per-bin average coverage across all barcodes and applies a
    chromosome-specific normalisation factor. By default, bins with unusually high
    coverage are down-weighted to reduce bias from collapsed repeats, high marker
    density, or other technical artefacts, while typical bins are left largely unchanged.

    Optionally, when `allow_upweight=True`, bins with moderately lower-than-typical
    coverage may be up-weighted toward the medias. This mode is intended for
    non-sparse, high-coverage datasets and includes safeguards to avoid amplifying noise
    or inventing signal in bins with near-zero coverage.

    Normalisation is applied per chromosome and approximately preserves total coverage.
    Counts are scaled and rounded to integers.

    Parameters
    ----------
    co_markers : MarkerRecords
        Marker data structure containing count matrices for multiple barcodes and
        chromosomes. Each entry is a 2D NumPy array of shape (bins, haplotypes).
    shrinkage_q : float, optional, default=0.99
        Quantile used to define the shrinkage reference when `allow_upweight=False`.
        Higher values restrict shrinkage to extreme high-coverage bins; lower values
        apply stronger down-weighting.
    allow_upweight : bool, optional, default=False
        If False, only down-weighting of high-coverage bins is performed.
        If True, bins with moderately low coverage may also be up-weighted toward a
        representative depth (the median bin coverage), subject to caps and thresholds.
    max_upweight : float, optional, default=4.0
        Maximum multiplicative factor applied when up-weighting low-coverage bins.
        Only used when `allow_upweight=True`.
    binwise_hap_mode : "independent" or "shared", optional, default "shared"
        Whether to calculate shared or independent norm factors for each haplotype in each bin.
    correct_hap_bias : bool, optional, default=True
        Whether to correct for global reference/alternate haplotype count imbalance
        prior to per-bin normalisation.
    hap_bias_shrinkage : float, optional, default=0.75
        Shrinkage parameter used when estimating the haplotype bias correction.
    Notes
    -----
    Up-weighting low-coverage bins can amplify noise and should only be enabled for
    datasets with sufficiently high and relatively uniform coverage (e.g. ≥10×
    resequencing). For sparse data, the default down-weight-only behaviour is safer.
    """
    tot = {}
    n_cb = len(co_markers)

    shared = (binwise_hap_mode == "shared")
    if binwise_hap_mode not in ("shared", "independent"):
        raise ValueError("binwise_hap_mode must be 'shared' or 'independent'")

    for cb, chrom, m in co_markers.deep_items():
        mc = m.sum(axis=1, keepdims=True) if shared else m
        tot[chrom] = mc.copy() if chrom not in tot else (tot[chrom] + mc)
    bin_means = {chrom: t / n_cb for chrom, t in tot.items()}
    if correct_hap_bias:
        bias_factor = _compute_bias_factor(co_markers, hap_bias_shrinkage)
    else:
        bias_factor = np.ones(shape=2)

    target_q = shrinkage_q if not allow_upweight else 0.5
    lambdas = {
        chrom: np.quantile(bm[bm > 0].ravel(), target_q)
        for chrom, bm in bin_means.items()
    }

    norm_factor = {}
    for chrom, bm in bin_means.items():
        lam = lambdas[chrom]
        if not allow_upweight:
            f = lam / (bm + lam)
        else:
            f = lam / (bm + 1e-6)
            f = np.where(bm > 0.0, f, 0.0)
            f = np.minimum(max_upweight, f)

        original_sum = bm.sum()
        new_sum = (bm * f).sum()
        scale = original_sum / new_sum if new_sum > 0 else 1.0
        nf = np.minimum(1.0 if not allow_upweight else max_upweight, f * scale)
        norm_factor[chrom] = nf

    co_markers_n = MarkerRecords.new_like(co_markers)
    for cb, chrom, m in co_markers.deep_items():
        scaled = m * norm_factor[chrom] / bias_factor[None, :]
        co_markers_n[cb, chrom] = np.round(scaled).astype(int)

    return co_markers_n


def _estimate_zip_lambda(cb_co_markers):
    m = np.concatenate(list(cb_co_markers.values())).sum(axis=1)
    zip_ = ZeroInflated(
        pmd.Poisson([m[m!=0].mean(),]),
        priors=[(m == 0).mean()]
    ).fit(m.reshape(-1, 1))
    return zip_.distribution.lambdas.numpy()[0]


def normalise_barcode_depth(co_markers, min_norm_factor=2.0):
    """
    Normalise per-barcode marker depths using ZIP-estimated expected counts.

    This scales each barcode’s marker counts by an estimated Poisson mean
    (λ) obtained from a zero-inflated Poisson fit to its concatenated
    marker counts. All barcodes are rescaled to a common target depth
    defined as the median of the estimated λ values, with an enforced
    minimum (`min_norm_factor`). Counts are rounded to integers.

    Parameters
    ----------
    co_markers : MarkerRecords
        Input marker record object containing per-barcode marker count
        arrays.
    min_norm_factor : float, optional
        Lower bound on the global normalisation factor. Ensures that very
        low median λ estimates do not collapse all counts. Default is 2.0.

    Returns
    -------
    MarkerRecords
        A new ``MarkerRecords`` instance with normalised marker counts for
        each barcode and chromosome.

    Notes
    -----
    For each barcode, the expected depth λ is estimated from all
    concatenated marker counts using a zero-inflated Poisson model. The
    final scaling applied is:

        scaled = counts / λ_cb * max(median(λ), min_norm_factor)

    Where ``λ_cb`` is the estimated Poisson mean for barcode ``cb``.
    """
    lambdas = {cb: _estimate_zip_lambda(co_markers[cb]) for cb in co_markers.barcodes}
    norm_factor = np.maximum(np.median(list(lambdas.values())), min_norm_factor)
    co_markers_n = MarkerRecords.new_like(co_markers)
    for cb, chrom, m in co_markers.deep_items():
        scaled = m / lambdas[cb] * norm_factor
        co_markers_n[cb, chrom] = np.round(scaled).astype(int)
    return co_markers_n