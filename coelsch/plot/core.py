"""
Shared plotting utilities for coelsch.
"""
import re
import itertools as it

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

from coelsch.defaults import DEFAULT_RANDOM_SEED


DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)
XLIM_OFFSET = 1e4


def chrom_subplots(chrom_sizes, nrows=1, figsize=(18, 5), xtick_every=1e7, xbuffer=5e5,
                   span_features=None, span_kwargs=None):
    """
    Create correctly proportioned subplots for individual chromosomes.

    Parameters
    ----------
    chrom_sizes : dict
        Dictionary with chromosome names as keys and chromosome sizes as values (in base pairs).
    nrows : int
        The number of rows of axes to produce.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (18, 5).
    xtick_every : float, optional
        Distance between x-ticks in base pairs. Default is 10 Mb (10e7).
    span_features : dict, optional
        Dictionary containing chromosome-specific features (e.g. centromeres) to highlight. Default is None.
    span_kwargs : dict, optional
        Additional keyword arguments for ax.axvspan. Default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the subplots.
    axes : list of matplotlib.axes.Axes
        List of axes for each subplot.
    """
    fig, axes = plt.subplots(
        figsize=figsize,
        ncols=len(chrom_sizes),
        nrows=nrows,
        width_ratios=list(chrom_sizes.values()),
        sharey='row',
        sharex='col',
        squeeze=False,
    )

    for i, row in enumerate(axes, 1):
        for chrom, ax in zip(chrom_sizes, row):
            ax.set_xlim(-xbuffer, chrom_sizes[chrom] + xbuffer)
            xticks = np.arange(0, chrom_sizes[chrom], xtick_every)
            ax.set_xticks(xticks)
            ax.set_xticklabels([int(i // 1e6) for i in xticks])
            if i == nrows:
                chrom_label = re.sub('^[Cc]hr', '', chrom)
                ax.set_xlabel(f'Chromosome {chrom_label} (Mb)')

    if span_features is not None:

        default_span_kwargs = {'alpha': 0.3, 'color': '#252525'}
        if span_kwargs is not None:
            span_kwargs.setdefault(default_span_kwargs)
        else:
            span_kwargs = default_span_kwargs

        for row in axes:
            for chrom, ax in zip(chrom_sizes, row):
                for s, e in span_features.get(chrom, []):
                    ax.axvspan(s, e, **span_kwargs)

    plt.tight_layout()
    return fig, axes


def chrom2d_subplots(chrom_sizes, figsize=(10, 10), xtick_every=1e7, xbuffer=5e5):
    """
    Create correctly proportioned 2D subplots for chromosome pairs.

    Parameters
    ----------
    chrom_sizes : dict
        Dictionary with chromosome names as keys and chromosome sizes as values (in base pairs).
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (10, 10).
    xtick_every : float, optional
        Distance between x-ticks in base pairs. Default is 10 Mb (10e7).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the subplots.
    axes : numpy.ndarray
        2D array of axes for the subplot grid.
    """
    fig, axes = plt.subplots(
        figsize=figsize,
        ncols=len(chrom_sizes),
        nrows=len(chrom_sizes),
        width_ratios=list(chrom_sizes.values()),
        height_ratios=list(chrom_sizes.values())[::-1],
        sharey='row',
        sharex='col'
    )
    if len(chrom_sizes) == 1 and isinstance(axes, mpl.axes.Axes):
        axes = [[axes,],]
    axes = axes[::-1]

    for chrom, ax in zip(chrom_sizes, axes[0]):
        ax.set_xlim(-xbuffer, chrom_sizes[chrom] + xbuffer)
        xticks = np.arange(0, chrom_sizes[chrom], xtick_every)
        ax.set_xticks(xticks)
        ax.set_xticklabels([int(i // 1e6) for i in xticks])
        chrom_label = re.sub('^[Cc]hr', '', chrom)
        ax.set_xlabel(f'Chr{chrom_label} (Mb)')

    for chrom, ax in zip(chrom_sizes, axes[:, 0]):
        ax.set_ylim(-xbuffer, chrom_sizes[chrom] + xbuffer)
        yticks = np.arange(0, chrom_sizes[chrom], xtick_every)
        ax.set_yticks(yticks)
        ax.set_yticklabels([int(i // 1e6) for i in yticks])
        chrom_label = re.sub('^[Cc]hr', '', chrom)
        ax.set_ylabel(f'Chr{chrom_label} (Mb)')

    plt.tight_layout()
    return fig, axes


def chrom2dtriangle_subplots(chrom_sizes, figsize=(10, 10), xtick_every=1e7):
    """
    Create triangular 2D subplots for chromosome pairs.

    Parameters
    ----------
    chrom_sizes : dict
        Dictionary with chromosome names as keys and chromosome sizes as values (in base pairs).
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (10, 10).
    xtick_every : float, optional
        Distance between x-ticks in base pairs. Default is 10 Mb (10e7).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the subplots.
    axes : numpy.ndarray
        2D array of axes for the triangular subplot grid.
    """
    assert len(chrom_sizes) > 1
    fig, axes = plt.subplots(
        figsize=figsize,
        ncols=len(chrom_sizes) - 1,
        nrows=len(chrom_sizes) - 1,
        width_ratios=list(chrom_sizes.values())[:-1],
        height_ratios=list(chrom_sizes.values())[1:][::-1],
        sharey='row',
        sharex='col'
    )
    chroms = list(chrom_sizes)
    axes = axes[::-1]
    for i, j in it.product(range(len(chroms) - 1), repeat=2):
        ax = axes[i, j]
        chrom_i, chrom_j = chroms[i], chroms[j + 1]
        if i < j:
            ax.set_axis_off()
            continue
        if i == j:
            ax.tick_params(
                which='both', bottom=True, top=True, left=True, right=True,
                labeltop=False, labelbottom=True, labelright=True, labelleft=False,
            )
            ax.set_xlim(0, chrom_sizes[chrom_i])
            xticks = np.arange(0, chrom_sizes[chrom_i], xtick_every)
            ax.set_xticks(xticks)
            ax.set_xticklabels([int(i // 1e6) for i in xticks])
            chrom_i_label = re.sub('^[Cc]hr', '', chrom_i)
            ax.set_xlabel(f'Chr{chrom_i_label} (Mb)', labelpad=0.1)

            ax.set_ylim(0, chrom_sizes[chrom_j])
            yticks = np.arange(0, chrom_sizes[chrom_j], xtick_every)
            ax.set_yticks(yticks)
            ax.set_yticklabels([int(i // 1e6) for i in yticks])
            chrom_j_label = re.sub('^[Cc]hr', '', chrom_j)
            ax.set_ylabel(f'Chr{chrom_j_label} (Mb)', labelpad=0.1)
            ax.yaxis.set_label_position('right')
        else:
            ax.tick_params(
                which='both', bottom=True, top=True, left=True, right=True,
                labeltop=False, labelbottom=False, labelright=False, labelleft=False,
            )

    plt.tight_layout(pad=0)
    return fig, axes


