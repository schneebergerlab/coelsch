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


def align_foreground_column(X, window=40):
    """
    Reorder marker counts so the dominant (foreground) haplotype is always in column 0.
    Supports masked arrays (mask is also aligned)

    Parameters
    ----------
    X : list of np.ndarray
        List of marker count arrays, each with shape (bins, 2).
    window : int, optional
        Width of the smoothing window for foreground detection (default is 40).

    Returns
    -------
    list of np.ndarray
        List of arrays with reordered haplotype columns.
    """
    X_reordered = []

    for x in X:
        fg_idx = argmax_smoothed_haplotype(x, window)
        idx = np.stack([fg_idx, 1 - fg_idx], axis=1)

        if isinstance(x, np.ma.MaskedArray):
            # reorder mask the same way
            x_reordered = np.take_along_axis(x.data, idx, axis=1)
            mask_reordered = np.take_along_axis(x.mask, idx, axis=1)
            X_reordered.append(np.ma.array(x_reordered, mask=mask_reordered))
        else:
            x_reordered = np.take_along_axis(x, idx, axis=1)
            X_reordered.append(x_reordered)

    return X_reordered


def detect_heterozygous_bins(X_ordered, window=40):
    """
    Detect heterozygous bins based on smoothed haplotype imbalance.

    Parameters
    ----------
    X_ordered : list of np.ndarray
        List of marker count arrays with shape (bins, 2). They should be ordered
        so that the dominant haplotype is in column 0.
    window : int, optional
        Width of the smoothing window (default is 40).

    Returns
    -------
    list of np.ndarray
        List of boolean masks indicating heterozygous bins for each input array.
    """
    heterozygous_mask = []
    for x in X_ordered:
        smoothed = smooth_counts_sum(x, window)
        # test the ratio of the fg and bg count sums.
        # In hom/het bins they should be 2:0 vs 1:1 respectively
        mask = smoothed[:, 0] < 2 * smoothed[:, 1]
        heterozygous_mask.append(mask)
    return heterozygous_mask
