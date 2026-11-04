import logging

import numpy as np
import pandas as pd

from coelsch.utils import load_json
from coelsch.signal import argmax_smoothed_haplotype
from coelsch.records import MarkerRecords, NestedData, NestedDataArray

log = logging.getLogger('coelsch')


def _estimate_marker_background(m, ws=100):
    """
    Estimate background signal by masking out predicted foreground signal.

    Parameters
    ----------
    m : np.ndarray
        Marker count matrix with shape (bins, haplotypes).
    ws : int, default=100
        Width of the convolution window in bins.

    Returns
    -------
    np.ndarray
        Background marker matrix with foreground masked with zeros.
    """
    fg_idx = argmax_smoothed_haplotype(m, ws)
    fg_masked = m.copy()
    fg_masked[np.arange(len(m)), fg_idx] = 0
    return fg_masked


def estimate_overall_background_signal(co_markers, conv_window_size, max_frac_bg,
                                       apply_per_geno=True):
    """
    Estimate and store background signal and barcode-level background contamination.

    Parameters
    ----------
    co_markers : MarkerRecords
        Marker records to process.
    conv_window_size : int
        Width of the background convolution window in base pairs.
    max_frac_bg : float
        Maximum tolerated background contamination fraction per barcode.
    apply_per_geno : bool, default=True
        Estimate background separately for each genotype.

    Returns
    -------
    MarkerRecords
        Filtered marker records with updated metadata.
    """
    conv_bins = conv_window_size // co_markers.bin_size
    background_signal = NestedDataArray(levels=('genotype', 'chrom'))
    estimated_background_fraction = NestedData(levels=('cb',), dtype=float)
    for geno, geno_co_markers in co_markers.groupby(by='genotype' if apply_per_geno else 'none'):
        bg_signal = {}
        frac_bg = {}
        for cb in geno_co_markers.barcodes:
            cb_co_markers = geno_co_markers[cb]
            bg_count = 0
            tot_count = 0
            for chrom, m in cb_co_markers.items():
                bg = _estimate_marker_background(m, conv_bins)
                if chrom not in bg_signal:
                    bg_signal[chrom] = bg
                else:
                    bg_signal[chrom] += bg

                bg_count += bg.sum(axis=None)
                tot_count += m.sum(axis=None)
            frac_bg[cb] = float(bg_count / tot_count)
            if frac_bg[cb] > max_frac_bg:
                co_markers.pop(cb)
        bg_signal = {
            chrom: sig / sig.sum() if sig.sum() > 0 else np.zeros_like(sig, dtype=float)
            for chrom, sig in bg_signal.items()
        }
        background_signal[geno] = bg_signal
        estimated_background_fraction.update(frac_bg)

    co_markers.add_metadata(
        background_signal=background_signal,
        estimated_background_fraction=estimated_background_fraction
    )
    return co_markers


def subtract_background(m, bg_signal, frac_bg, return_bg=False):
    """
    Deterministically subtract estimated background contamination from a marker count matrix.

    This function removes background signal from a barcode's marker counts by allocating
    the expected number of background markers proportionally to both the
    observed counts and a background probability model. Subtraction is capped so that
    no cell goes below zero.

    Parameters
    ----------
    m : ndarray of shape (bins, haplotypes)
        Observed marker count matrix for a single barcode.
    bg_signal : ndarray of shape (bins, haplotypes)
        Background probability matrix for the same chromosome, typically normalized
        so that its sum equals 1. Higher values indicate bins and haplotypes where
        background contamination is more likely.
    frac_bg : float
        Estimated background contamination fraction for this barcode. The function
        will subtract approximately `round(tot * frac_bg)` markers in total.

    Returns
    -------
    ndarray of shape (bins, haplotypes)
        Background-corrected marker count matrix, rounded to integers and guaranteed
        to have non-negative entries.
    """
    tot = m.sum()
    if tot == 0:
        return m.copy()
    weights = m * bg_signal
    w_sum = weights.sum()
    if w_sum == 0:
        return m.copy()
    bg_expected = weights * (tot * frac_bg / w_sum)
    bg = np.round(np.minimum(bg_expected, m)).astype(int)
    fg = m - bg
    if not return_bg:
        return fg
    return fg, bg


def clean_marker_background(co_markers, apply_per_geno=True):
    """
    Subtract estimated background signal from marker data for each barcode.

    Parameters
    ----------
    co_markers : MarkerRecords
        Marker records with background metadata.
    apply_per_geno : bool, default=True
        Whether to clean using genotype-specific background.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    MarkerRecords
        Background-cleaned marker records.
    """
    bg_signal = co_markers.metadata['background_signal']
    frac_bg = co_markers.metadata['estimated_background_fraction']
    if apply_per_geno:
        genotypes = co_markers.metadata['genotypes']
    co_markers_c = MarkerRecords.new_like(co_markers)
    for cb, chrom, m in co_markers.deep_items():
        geno = genotypes[cb] if apply_per_geno else 'ungrouped'
        co_markers_c[cb, chrom] = subtract_background(
            m, bg_signal[geno, chrom], frac_bg[cb]
        )
    return co_markers_c
