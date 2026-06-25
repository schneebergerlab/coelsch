import logging

import numpy as np

from coelsch.signal import approximate_haplotype_patterns


log = logging.getLogger('coelsch')


def normalise_rows(arr, fallback):
    totals = arr.sum(axis=1, keepdims=True)
    return np.divide(
        arr,
        totals,
        out=np.repeat(fallback[None, :], arr.shape[0], axis=0),
        where=totals > 0,
    )


def initial_haplotype_bias(chrom_counts, expected_dosage, correction):
    hap_totals = np.sum([m.sum(axis=0) for m in chrom_counts], axis=0)
    observed = hap_totals + correction
    observed = observed / observed.sum()

    expected = np.asarray(expected_dosage, dtype=float)
    expected = expected / expected.sum()

    bias = observed / expected
    bias /= bias.mean()
    return bias


def estimate_haplotype_bias(co_markers, correction=1e-2, window=40,
                            min_bias=0.5, max_bias=2.0):
    """
    Estimate technical marker bias for each haplotype channel.

    Bias is estimated after accounting for the approximate haplotype dosage
    pattern in each bin. Pattern matching is performed once after correcting
    counts by a global observed-vs-expected bias prior. The returned vector has
    shape ``(n_haplotypes,)`` and is normalised to mean 1.
    """
    n_haplotypes = co_markers.n_haplotypes
    fallback = np.ones(n_haplotypes, dtype=float) / n_haplotypes
    dosage_patterns = np.asarray(
        co_markers.experiment_params.haplotype_state_dosage_patterns,
        dtype=float,
    )
    if dosage_patterns.shape[1] != n_haplotypes:
        raise ValueError(
            f'haplotype dosage patterns have {dosage_patterns.shape[1]} channels, '
            f'but MarkerRecords has {n_haplotypes}'
        )

    chrom_counts = []
    for chrom in co_markers.chrom_sizes:
        m = co_markers[:, chrom].stack_values()
        cb_totals = m.sum(axis=(1, 2))[:, np.newaxis, np.newaxis]
        with np.errstate(invalid='ignore', divide='ignore'):
            m_norm = np.where(cb_totals > 0, m / cb_totals, 0.0).sum(axis=0)
        chrom_counts.append(m_norm)

    if not any(np.any(m.sum(axis=1) > 0) for m in chrom_counts):
        return np.ones(n_haplotypes, dtype=float)

    bias_prior = initial_haplotype_bias(
        chrom_counts,
        co_markers.experiment_params.haplotype_dosage,
        correction,
    )
    bias_prior = np.clip(bias_prior, min_bias, max_bias)
    bias_prior /= bias_prior.mean()

    observed_chunks = []
    expected_chunks = []
    for m_norm in chrom_counts:
        marker_mask = m_norm.sum(axis=1) > 0
        if not np.any(marker_mask):
            continue

        observed_ratio = normalise_rows(m_norm + correction, fallback)
        pattern_counts = m_norm / bias_prior[None, :]
        pattern_weights = approximate_haplotype_patterns(
            [pattern_counts],
            dosage_patterns,
            window=window,
        )
        expected_dosage = pattern_weights @ dosage_patterns
        expected_ratio = normalise_rows(expected_dosage, fallback)

        observed_chunks.append(observed_ratio[marker_mask])
        expected_chunks.append(expected_ratio[marker_mask])

    if not observed_chunks:
        return np.ones(n_haplotypes, dtype=float)

    observed = np.concatenate(observed_chunks)
    expected = np.concatenate(expected_chunks)

    bias = np.ones(n_haplotypes, dtype=float)
    for h in range(n_haplotypes):
        valid = expected[:, h] > correction
        if np.any(valid):
            ratio = (observed[valid, h] + correction) / (
                expected[valid, h] + correction
            )
            bias[h] = np.median(ratio)

    bias = np.clip(bias, min_bias, max_bias)
    bias /= bias.mean()

    return bias


def compute_bias_factor(co_markers, hap_bias_correction_strength=0.75):
    raw_factor = estimate_haplotype_bias(co_markers)
    return 1.0 + hap_bias_correction_strength * (raw_factor - 1.0)

