import numpy as np
from scipy.ndimage import convolve1d


def smooth_counts_sum(m, window=40):
    """
    Smooth marker counts using a 1D convolution.

    Parameters
    ----------
    m : np.ndarray
        Marker count array with shape (bins, haplotypes).
    window : int, optional
        Width of the convolution window (default is 40).

    Returns
    -------
    np.ndarray
        Smoothed array of the same shape as `m`.
    """
    return convolve1d(m, np.ones(window), axis=0, mode='constant', cval=0)


def argmax_smoothed_haplotype(m, window=40):
    """
    Identify the index of the dominant (foreground) haplotype in each bin after smoothing.

    Parameters
    ----------
    m : np.ndarray
        Marker count array with shape (bins, haplotypes).
    window : int, optional
        Width of the smoothing window (default is 40).

    Returns
    -------
    np.ndarray
        Array of shape (bins,) containing the dominant haplotype index (0 or 1) per bin.
    """
    smoothed = smooth_counts_sum(m, window)
    return smoothed.argmax(axis=1)


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=axis, keepdims=True)


def approximate_haplotype_patterns(
    X,
    component_dosages,
    window=40,
    temperature=0.05,
    floor=0.02,
):
    """
    Approximate per-bin haplotype dosage patterns from marker counts.

    Marker counts are smoothed along bins, converted to haplotype proportions,
    and compared to each expected dosage pattern in ``component_dosages``. The
    returned array contains soft pattern weights with shape
    ``(total_bins, n_patterns)``. These weights can be used as rough labels for
    cleaning or as prior/responsibility estimates for downstream model fitting.

    Parameters
    ----------
    X : iterable of ndarray
        Marker count arrays with shape ``(bins, haplotypes)``. Masked arrays are
        supported.
    component_dosages : ndarray
        Expected dosage patterns with shape ``(n_patterns, haplotypes)``. Each
        row must have positive total dosage.
    window : int, optional
        Smoothing window in bins.
    temperature : float, optional
        Softmax temperature used to convert pattern distances to weights.
    floor : float, optional
        Uniform weight mixed into each row to prevent exact zero weights.

    Returns
    -------
    ndarray
        Soft pattern weights with one row per input bin.
    """
    component_dosages = np.asarray(component_dosages, dtype=float)

    dosage_totals = component_dosages.sum(axis=1, keepdims=True)
    if np.any(dosage_totals <= 0):
        raise ValueError("all component dosage vectors must have non-zero dosage")

    dosage_props = component_dosages / dosage_totals
    pattern_weights = []

    temperature = max(float(temperature), 1e-12)

    for x in X:
        x = np.ma.asarray(x, dtype=float)
        smoothed = np.ma.asarray(smooth_counts_sum(x, window), dtype=float)
        smoothed_data = smoothed.filled(0.0)

        row_sum = smoothed_data.sum(axis=1, keepdims=True)
        zero_rows = row_sum[:, 0] <= 0

        obs_props = np.divide(
            smoothed_data,
            row_sum,
            out=np.zeros_like(smoothed_data, dtype=float),
            where=row_sum > 0,
        )

        dist2 = ((obs_props[:, None, :] - dosage_props[None, :, :]) ** 2).sum(axis=2)
        p = _softmax(-dist2 / temperature, axis=1)

        if np.any(zero_rows):
            p[zero_rows] = 1.0 / component_dosages.shape[0]

        p = (1.0 - floor) * p + floor / component_dosages.shape[0]
        pattern_weights.append(p)

    return np.concatenate(pattern_weights, axis=0)
