"""
Dataset-level haplotype, recombination, and distortion plots.
"""
import itertools as it

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mpl

from coelsch.experiment.genotypes import GenotypeKey
from coelsch.recombination import recombination_landscape, coefficient_of_coincidence
from coelsch.distortion import segregation_distortion
from coelsch.defaults import DEFAULT_RANDOM_SEED

from .core import chrom_subplots, chrom2dtriangle_subplots


DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


def _append_track_label(group, track_label, apply_by):
    base_label = None
    if apply_by != 'none':
        base_label = str(group)
    if base_label and track_label:
        return f'{base_label}: {track_label}'
    if base_label:
        return base_label
    return track_label


def plot_recombination_landscape(co_preds, co_markers=None,
                                 apply_by='none',
                                 rolling_mean_window_size=1_000_000,
                                 nboots=100, ci=95,
                                 min_prob=5e-3,
                                 axes=None,
                                 figsize=(12, 4),
                                 rng=DEFAULT_RNG):
    """
    Plot the recombination landscape across chromosomes for multiple cell barcodes.

    Parameters
    ----------
    co_preds : PredictionRecords
        PredictionRecords object containing the haplotype predictions.
    co_markers : MarkerRecords, optional
        MarkerRecords object containing the marker data for the dataset. When provided, used for calculating edge
        effects only at chromosome ends only. Default is None.
    apply_by : str or func, optional
        how to group barcodes for recombination landscape calculation/plotting. Can be "none", "genotype" or a function
        that is passed to PredictionRecords.groupby. Default is "none".
    rolling_mean_window_size : int, optional
        The size of the window for computing the rolling mean (in base pairs). Default is 1,000,000.
    nboots : int, optional
        The number of bootstrap iterations for calculating confidence intervals. Default is 100.
    ci : int, optional
        The confidence interval percentage. Default is 95.
    min_prob : float, optional
        The minimum probability change to consider when counting crossovers. Default is 5e-3.
    axes : list of matplotlib.axes.Axes, optional
        The axes to plot on. If None, new axes are created. Default is None.
    figsize : tuple, optional
        The size of the figure (width, height) in inches. Default is (12, 4).
    rng : numpy.random.Generator, optional
        The random number generator to use for bootstrapping. Default is `DEFAULT_RNG`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure containing the recombination landscape plots.
    axes : list of matplotlib.axes.Axes
        The axes containing the recombination landscape plots.
    """
    if axes is None:
        fig, axes = chrom_subplots(co_preds.chrom_sizes, figsize=figsize)
        if axes.ndim == 2:
            axes = axes[0]
    else:
        fig = plt.gcf()
        assert len(axes) == len(co_preds.chrom_sizes)
        if len(co_preds.chrom_sizes) == 1 and isinstance(axes, mpl.axes.Axes):
            axes = [axes,]

    cm_per_mb = recombination_landscape(
        co_preds,
        co_markers=co_markers,
        apply_by=apply_by,
        rolling_mean_window_size=rolling_mean_window_size,
        nboots=nboots,
        min_prob=min_prob,
        rng=rng
    )

    lower = (100 - ci) / 2
    upper = 100 - lower

    legend_entries = {}
    for group, group_cm_per_mb in cm_per_mb.items():
        track_colours = {}
        for track_key, track_cm_per_mb in group_cm_per_mb.items():
            track_label = None if track_key == 'haplotype' else track_key
            plot_label = _append_track_label(group, track_label, apply_by)
            for chrom, ax in zip(track_cm_per_mb, axes):
                x = np.arange(0, co_preds.nbins[chrom]) * co_preds.bin_size
                c = track_cm_per_mb[chrom]
                curr_colour = track_colours.get(plot_label, None)
                line, = ax.step(x, np.nanmean(c, axis=0), color=curr_colour)
                curr_colour = line.get_color()
                track_colours[plot_label] = curr_colour
                ax.fill_between(
                    x=x,
                    y1=np.nanpercentile(c, lower, axis=0),
                    y2=np.nanpercentile(c, upper, axis=0),
                    alpha=0.25,
                    color=curr_colour
                )
                if plot_label is not None:
                    legend_entries[plot_label] = curr_colour
    if legend_entries:
        for plot_label, curr_colour in legend_entries.items():
            axes[-1].plot([], [], color=curr_colour, label=plot_label)
        axes[-1].legend()

    axes[0].set_ylabel('cM / Mb')
    plt.tight_layout()
    return fig, axes


def plot_allele_ratio(co_preds,
                      apply_by='none',
                      nboots=100, ci=95,
                      axes=None,
                      figsize=(12, 4),
                      rng=DEFAULT_RNG):
    """
    Plot the allele ratio across chromosomes for multiple cells with bootstrapped confidence intervals.

    Parameters
    ----------
    co_preds : PredictionRecords
        PredictionRecords object containing the haplotype predictions.
    apply_by : str or func, optional
        How to group barcodes for allele ratio calculation/plotting. Can be "none", "genotype" or a function
        that is passed to PredictionRecords.groupby. Default is "none".
    nboots : int, optional
        The number of bootstrap iterations for calculating confidence intervals. Default is 100.
    ci : int, optional
        The confidence interval percentage. Default is 95.
    axes : list of matplotlib.axes.Axes, optional
        The axes to plot on. If None, new axes are created. Default is None.
    figsize : tuple, optional
        The size of the figure (width, height) in inches. Default is (12, 4).
    rng : numpy.random.Generator, optional
        The random number generator to use for bootstrapping. Default is `DEFAULT_RNG`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure containing the marker coverage plots.
    axes : list of matplotlib.axes.Axes
        The axes containing the allele ratio plots.
    """
    if axes is None:
        fig, axes = chrom_subplots(co_preds.chrom_sizes, figsize=figsize)
        if axes.ndim == 2:
            axes = axes[0]
    else:
        fig = plt.gcf()
        assert len(axes) == len(co_preds.chrom_sizes)
        if len(co_preds.chrom_sizes) == 1 and isinstance(axes, mpl.axes.Axes):
            axes = [axes,]

    lower = (100 - ci) / 2
    upper = 100 - lower

    legend_entries = {}
    for group, group_co_preds in co_preds.groupby(apply_by):
        N = len(group_co_preds)
        track_colours = {}
        for chrom, ax in zip(group_co_preds.chrom_sizes, axes):
            x = np.arange(0, group_co_preds.nbins[chrom]) * group_co_preds.bin_size
            for track_label, haps in group_co_preds.iter_scalar_haplotypes(chrom):
                c = []
                for _ in range(nboots):
                    idx = rng.integers(0, N, size=N)
                    c.append(haps[idx].mean(axis=0))
                plot_label = _append_track_label(group, track_label, apply_by)
                curr_colour = track_colours.get(plot_label, None)
                line, = ax.step(x, np.nanmean(c, axis=0), color=curr_colour)
                curr_colour = line.get_color()
                track_colours[plot_label] = curr_colour
                ax.fill_between(
                    x=x,
                    y1=np.nanpercentile(c, lower, axis=0),
                    y2=np.nanpercentile(c, upper, axis=0),
                    alpha=0.25,
                    color=curr_colour
                )
                if plot_label is not None:
                    legend_entries[plot_label] = curr_colour
    if legend_entries:
        for plot_label, curr_colour in legend_entries.items():
            axes[-1].plot([], [], color=curr_colour, label=plot_label)
        axes[-1].legend()

    axes[0].set_ylabel('Allele ratio')

    if (
        co_preds.experiment_params.genotyping_strategy != 'recombinant'
        and co_preds.experiment_params.crossing_strategy == 'backcross'
    ):
        ylim = (0, 0.5)
    else:
        ylim = (0, 1)

    axes[0].set_ylim(*ylim)

    plt.tight_layout()
    return fig, axes


def _plot_segdist_1d(seg_dist, chrom_sizes, fig, axes, colour=None, label=None):
    for chrom, ax in zip(chrom_sizes, axes):
        chrom_sd = seg_dist.query('chrom_1 == @chrom')
        line, = ax.step(chrom_sd.pos_1.values, chrom_sd.lod_score.values, color=colour)
        colour = line.get_color()
    axes[0].set_ylabel('LOD score')
    axes[-1].plot([], [], color=colour, label=label)
    if label is not None:
        axes[-1].legend()
    plt.tight_layout()
    return axes


def _plot_segdist_2d(seg_dist, chrom_sizes, fig, axes, cmap='Blues',
                     vmin=None, vmax=None, label=None):
    chroms = list(chrom_sizes)
    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = seg_dist.lod_score.max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    for i, j in it.product(range(len(chroms) - 1), repeat=2):
        ax = axes[i, j]
        if i < j:
            continue
        chrom_1, chrom_2 = chroms[j], chroms[i + 1]
        chrom_sd = seg_dist.query('chrom_1 == @chrom_1 & chrom_2 == @chrom_2')
        chrom_sd = chrom_sd.pivot(columns='pos_1', index='pos_2', values='lod_score')
        im = ax.imshow(
            chrom_sd.values,
            norm=norm,
            extent=(0, chrom_sizes[chrom_1], chrom_sizes[chrom_2], 0),
            cmap=cmap
        )
    plt.colorbar(im, ax=axes[0, 1:], label='LOD score', orientation='horizontal')
    if label is not None:
        fig.suptitle(label)
    plt.tight_layout()
    return axes


def plot_segregation_distortion(co_preds, cb_whitelist=None,
                                order=1, resolution=250_000, processes=1,
                                axes=None, figwidth=12, figheight=4,
                                colour=None, cmap='Blues',
                                vmin=None, vmax=None, label=None):
    """
    Plot segregation distortion LOD scores for haplotype predictions.

    This function visualizes the segregation distortion of inherited haplotypes in either 1D or 2D plots.
    It calculates the segregation distortion using the specified order and plots it on a set of subplots. 
    The order of distortion can either be 1 (single locus, for single-chromosome plots)
    or 2 (two-locus, for pairwise chromosome plots).

    Parameters
    ----------
    co_preds : PredictionRecords
        PredictionRecords object containing the haplotype predictions.
    cb_whitelist : list of str, optional
        A list of cell barcodes to include in the analysis. If None, all barcodes are included.
        Default is None.
    order : int, optional
        The order of distortion to plot. Can either be 1 (single-chromosome distortion) or 2
        (pairwise chromosome distortion). Default is 1.
    resolution : int, optional
        The resolution of the plot in base pairs. Default is 250,000.
    processes : int, optional
        The number of processes to use for parallel computation of distortion. Default is 1.
    axes : list of matplotlib.axes.Axes or None, optional
        The axes to plot on. If None, new axes are created. Default is None.
    figwidth : int, optional
        The width of the figure in inches. Default is 12.
    figheight : int, optional
        The height of the figure in inches. Default is 4.
    colour : str, optional
        The color to use for the 1D plot lines and areas. If None, a colour is selected from the default palette.
        Default is None.
    cmap : str, optional
        The colormap to use for the 2D plot. Default is 'Blues'.
    vmin : float, optional
        The minimum value for the color scale in the 2D plot. If None, it defaults to 0. Default is None.
    vmax : float, optional
        The maximum value for the color scale in the 2D plot. If None, it defaults to the maximum LOD score.
        Default is None.
    label : str, optional
        The label for the plot legend or figure title. If None, no label is added. Default is None.

    Returns
    -------
    axes : list of matplotlib.axes.Axes
        The axes containing the segregation distortion plots.

    Raises
    ------
    ValueError
        If `order` is not 1 or 2, a ValueError is raised.
    
    Notes
    -----
    - For `order=1`, the function generates a 1D plot for each chromosome, showing the LOD score of the segregation distortion.
    - For `order=2`, the function generates a triangular 2D heatmap showing the pairwise segregation distortion between
      chromosome pairs.
    - The function calls the `coelsch.distortion.segregation_distortion()` function to compute the distortion values and uses the 
      appropriate plotting function depending on the specified `order`.

    See Also
    --------
    coelsch.distortion.segregation_distortion
    """
    if order not in (1, 2):
        raise ValueError('Can only generate plots for distortions of order 1 or 2')

    seg_dist = segregation_distortion(
        co_preds, order=order, resolution=resolution, processes=processes
    )

    if axes is None:
        if order == 1:
            fig, axes = chrom_subplots(co_preds.chrom_sizes, figsize=(figwidth, figheight))
            if axes.ndim == 2:
                axes = axes[0]
        elif order == 2:
            fig, axes = chrom2dtriangle_subplots(co_preds.chrom_sizes, figsize=(figwidth, figwidth))
    else:
        fig = plt.gcf()
        assert len(axes) == len(co_preds.chrom_sizes)
        if len(co_preds.chrom_sizes) == 1 and isinstance(axes, mpl.axes.Axes):
            axes = [axes,]

    if order == 1:
        _plot_segdist_1d(seg_dist, co_preds.chrom_sizes, fig, axes,
                         colour=colour, label=label)
    elif order == 2:
        _plot_segdist_2d(seg_dist, co_preds.chrom_sizes, fig, axes,
                         cmap=cmap, vmin=vmin, vmax=vmax, label=label)

    return axes


def plot_coefficient_of_coincidence(co_preds,
                                    apply_by='none',
                                    nboots=100, ci=95,
                                    min_dist=None,
                                    max_dist=None,
                                    step_size=1e6,
                                    only_adjacent=False,
                                    chroms=None,
                                    show_L_int=True,
                                    axes=None,
                                    figsize=(12, 4),
                                    rng=DEFAULT_RNG):
    """
    Plot the coefficient-of-coincidence (CoC) curve with bootstrap confidence intervals.

    Parameters
    ----------
    co_preds : PredictionRecords
        haplotype predictions object with metadata slot crossover_samples
    apply_by : str or func, optional
        How to group barcodes for allele ratio calculation/plotting. Can be "none", "genotype" or a function
        that is passed to PredictionRecords.groupby. Default is "none".
    nboots : int, optional
        Number of bootstrap replicates used to estimate the CoC distribution.
    ci : float, optional
        Confidence interval width (percent). For example, ``ci=95`` produces
        2.5th and 97.5th percentiles.
    min_dist : int or None, optional
        Minimum physical distance (in bp) used when constructing CoC bins. If
        None, this is set to the rigidity of the model used for haplotype predictions
    max_dist : int or None, optional
        Maximum physical distance (in bp). If None, determined from the length of the
        smallest chromosome
    step_size : int, optional
        Bin step size in bp for computing the CoC curve. Default 1 Mb.
    only_adjacent : bool, optional
        If True, compute CoC using only adjacent crossover distances.
    chroms : list or None
        If provided, limits the plot to a particular chromosome.
    show_L_int: bool, optional
        If True, plot the position of the interference length (defined by Ernst et al. 2024)
    axes : matplotlib.axes.Axes or None, optional
        Existing axes to draw on. If None, a new figure and axes are created.
    figsize : tuple, optional
        Size of the figure if ``ax`` is not provided.
    rng : numpy.random.Generator, optional
        Random number generator passed to bootstrap sampling.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot.
    ax : matplotlib.axes.Axes
        Axes on which the CoC curve and confidence intervals were drawn.

    Notes
    -----
    This function calls ``coefficient_of_coincidence`` to compute bootstrap
    CoC curves, then plots the mean curve and a shaded percentile band
    corresponding to the chosen confidence interval. Horizontal reference
    lines are drawn at CoC = 1 and CoC = 0.5. If ``apply_per_geno=True``,
    each genotype receives its own curve and legend entry.
    """
    if chroms is None:
        chrom_sizes = co_preds.chrom_sizes
    else:
        chrom_sizes = {chrom: co_preds.chrom_sizes[chrom] for chrom in chroms}
    if axes is None:
        if max_dist is None:
            fig, axes = chrom_subplots(chrom_sizes, figsize=figsize)
        else:
            fig, axes = chrom_subplots({chrom: max_dist for chrom in chrom_sizes}, figsize=figsize)
        if axes.ndim == 2:
            axes = axes[0]
    else:
        fig = plt.gcf()
        assert len(axes) == len(chrom_sizes)
        if len(co_preds.chrom_sizes) == 1 and isinstance(axes, mpl.axes.Axes):
            axes = [axes,]

    lower = (100 - ci) / 2
    upper = 100 - lower

    legend_entries = {}
    for group, group_co_preds in co_preds.groupby(apply_by):

        group_coc, x, l_int = coefficient_of_coincidence(
            group_co_preds, nboots=nboots, chroms=chroms, only_adjacent=only_adjacent,
            min_dist=min_dist, max_dist=max_dist, step_size=step_size,
        )

        meioses = sorted(group_coc) # list of (hap1, hap2) pairs
        if len(meioses) == 1:
            track_labels = [None,]
        elif len(meioses) == 2:
            track_labels = ['parent1', 'parent2']
        else:
            raise ValueError()

        track_colours = {}
        for chrom, ax in zip(chrom_sizes, axes):
            for meiosis, track_label in zip(meioses, track_labels):
                meiosis_coc = group_coc[meiosis][chrom]
                plot_label = _append_track_label(group, track_label, apply_by)
                if np.isnan(meiosis_coc).all():
                    log.warning(
                        'Skipping undefined CoC curve for group=%r track=%r chrom=%s',
                        group, track_label, chrom
                    )
                    continue
                curr_colour = track_colours.get(plot_label, None)
                line, = ax.plot(x[chrom], np.nanmean(meiosis_coc, axis=0), color=curr_colour)
                curr_colour = line.get_color()
                track_colours[plot_label] = curr_colour
                ax.fill_between(
                    x=x[chrom],
                    y1=np.nanpercentile(meiosis_coc, lower, axis=0),
                    y2=np.nanpercentile(meiosis_coc, upper, axis=0),
                    alpha=0.25,
                    color=curr_colour
                )
                if show_L_int:
                    ax.axvline(np.nanmean(l_int[meiosis][chrom]), color=curr_colour)
                    ax.axvspan(np.nanpercentile(l_int[meiosis][chrom], lower),
                               np.nanpercentile(l_int[meiosis][chrom], upper),
                               color=curr_colour,
                               alpha=0.25,
                               zorder=0)
                if plot_label is not None:
                    legend_entries[plot_label] = curr_colour

    if legend_entries:
        for plot_label, curr_colour in legend_entries.items():
            axes[-1].plot([], [], color=curr_colour, label=plot_label)
        axes[-1].legend()

    for chrom, ax in zip(chrom_sizes, axes):
        ax.axhline(1, color='#252525', zorder=-1)
        ax.axhline(0.5, color='#252525', zorder=-1, ls='--')
        ax.set_xlabel(f'{chrom} inter-CO distance (Mb)')

    axes[0].set_ylabel('Coefficient of coincidence')
    plt.tight_layout()
    return fig, axes


