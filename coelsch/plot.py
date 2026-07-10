"""
Plotting functions for visualizing recombination landscapes, allele ratios,
segregation distortion, and marker plots for coelsch datasets.

Functions
---------
chrom_subplots : Create correctly sized subplots for chromosomes of a genome.
chrom2d_subplots : Create correctly sized 2D subplots for chromosome pairs of a genome.
chrom2dtriangle_subplots : Create triangular 2D subplots for chromosome pairs of a genome.
chrom_markerplot : Plot marker coverage for a barcode for a single chromosome.
single_cell_markerplot : Plot marker coverage for a single cell barcode.
plot_recombination_landscape : Plot recombination landscape for a set of predictions.
plot_allele_ratio : Plot allele ratio for predictions.
plot_segregation_distortion : Plot segregation distortions in 1D or 2D for predictions.
run_plot : Generate a plot (e.g., markerplot, recombination) for a specific cell barcode (cli).
"""
import re
import logging
import itertools as it

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib as mpl

from .utils import load_json
from .recombination import recombination_landscape, coefficient_of_coincidence
from .distortion import segregation_distortion
from .load.genotype import GenotypeKey
from .defaults import DEFAULT_RANDOM_SEED


log = logging.getLogger('coelsch')
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)
XLIM_OFFSET = 1e4


def chrom_subplots(chrom_sizes, figsize=(18, 5), xtick_every=1e7, xbuffer=5e5,
                   span_features=None, span_kwargs=None):
    """
    Create correctly proportioned subplots for individual chromosomes.

    Parameters
    ----------
    chrom_sizes : dict
        Dictionary with chromosome names as keys and chromosome sizes as values (in base pairs).
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
        width_ratios=list(chrom_sizes.values()),
        sharey='row',
        sharex='col'
    )
    if len(chrom_sizes) == 1 and isinstance(axes, mpl.axes.Axes):
        axes = [axes,]

    for chrom, ax in zip(chrom_sizes, axes):
        ax.set_xlim(-xbuffer, chrom_sizes[chrom] + xbuffer)
        xticks = np.arange(0, chrom_sizes[chrom], xtick_every)
        ax.set_xticks(xticks)
        ax.set_xticklabels([int(i // 1e6) for i in xticks])
        chrom_label = re.sub('^[Cc]hr', '', chrom)
        ax.set_xlabel(f'Chromosome {chrom_label} (Mb)')

    if span_features is not None:

        default_span_kwargs = {'alpha': 0.3, 'color': '#252525'}
        if span_kwargs is not None:
            span_kwargs.setdefault(default_span_kwargs)
        else:
            span_kwargs = default_span_kwargs

        for chrom, ax in zip(chrom_sizes, axes):
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


def chrom_markerplot(co_markers, chrom_size, bin_size, ax=None, max_yheight=20,
                     ref_colour='#0072b2', alt_colour='#d55e00', ori_colour='#252525'):
    """
    Plot the marker coverage for a barcode for a single chromosome.

    Parameters
    ----------
    co_markers : numpy.ndarray
        The barcode marker data for a specific chromosome.
    chrom_size : int
        The size of the chromosome in base pairs.
    bin_size : int
        The size of each bin in base pairs.
    ax : matplotlib.axes.Axes, optional
        The axes to plot the markers on. Default is None, which creates new axes.
    max_yheight : int, optional
        The maximum y value for the plot. Default is 20.
    ref_colour : str, optional
        The color to use for reference markers. Default is '#0072b2'.
    alt_colour : str, optional
        The color to use for alternate markers. Default is '#d55e00'.
    ori_colour : str, optional
        The color to use for the origin. Default is '#252525'.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plotted markers.
    """
    ref_markers = co_markers[:, 0].copy()
    ref_markers[ref_markers > max_yheight] = max_yheight
    alt_markers = co_markers[:, 1].copy()
    alt_markers[alt_markers > max_yheight] = max_yheight
    alt_markers = np.negative(alt_markers)

    nbins = len(co_markers)
    pos = np.arange(nbins) * bin_size
    base = np.repeat(0, nbins)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.vlines(pos, base, ref_markers, color=ref_colour, zorder=0)
    ax.vlines(pos, base, alt_markers, color=alt_colour, zorder=0)
    ax.plot([0, chrom_size], [0, 0], ls='-', color=ori_colour)
    return ax


def _add_co_prob_colormesh(ax, hp, chrom_size, bin_size, ylims, cmap, norm):
    nbins = len(hp)
    xpos = np.insert(np.arange(nbins) * bin_size, nbins, chrom_size)
    ax.pcolormesh(
        xpos,
        ylims,
        hp.reshape(1, -1),
        cmap=cmap,
        norm=norm,
        alpha=0.5,
        zorder=-2,
        rasterized=True
    )
    return ax


def _add_gt_vlines(ax, gt, bin_size, ylims, colour='#eeeeee'):
    gt_pos = (np.where(np.diff(gt))[0] + 1) * bin_size
    if len(gt_pos):
        ax.vlines(
            gt_pos,
            np.repeat(ylims[0], len(gt_pos)),
            np.repeat(ylims[1], len(gt_pos)),
            ls='--',
            colors=colour,
            zorder=-1,
        )


def single_cell_markerplot(cb, co_markers, *, co_preds=None, figsize=(18, 4), chroms=None,
                           show_mesh_prob=True, annotate_co_number=True,
                           nco_min_prob_change=5e-3, show_gt=True,
                           max_yheight='auto', ref_colour='#0072b2', alt_colour='#d55e00'):
    """
    Plot the marker coverage and crossover probabilities for a single cell across chromosomes.

    Parameters
    ----------
    cb : str
        The barcode of the single cell to plot.
    co_markers : MarkerRecords
        MarkerRecords object containing the marker data for the dataset.
    co_preds : PredictionRecords, optional
        PredictionRecords object containing the haplotype predictions. Default is None.
    figsize : tuple, optional
        The size of the figure (width, height) in inches. Default is (18, 4).
    chroms : list of str, optional
        A list of chromosome names to plot. If None, all chromosomes will be plotted. Default is None.
    show_mesh_prob : bool, optional
        Whether to display the haplotype probability mesh. Default is True.
    annotate_co_number : bool, optional
        Whether to annotate the number of crossovers for each chromosome. Default is True.
    nco_min_prob_change : float, optional
        The minimum probability change to consider when counting crossovers for annotation. Default is 5e-3.
    show_gt : bool, optional
        Whether to show the ground truth crossover locations for simulated data, where available. Default is True.
    max_yheight : float or 'auto', optional
        The maximum y-axis height for the plots. If 'auto', the 99.5th percentile of all marker values 
        is used. Default is 'auto'.
    ref_colour : str, optional
        The color for the reference markers. Default is '#0072b2'.
    alt_colour : str, optional
        The color for the alternate markers. Default is '#d55e00'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure containing the marker coverage plots.
    axes : list of matplotlib.axes.Axes
        The list of axes objects for the subplots.
    
    Raises
    ------
    KeyError
        If the provided `cb` (cell barcode) is not found in `co_markers`.
    """
    if cb not in co_markers.barcodes:
        raise KeyError(f'cb {cb} not in co_marker object')

    chrom_sizes = co_markers.chrom_sizes
    if chroms is not None:
        chrom_sizes = {c: chrom_sizes[c] for c in chroms}
    fig, axes = chrom_subplots(chrom_sizes, figsize=figsize)
    try:
        hap1, hap2 = GenotypeKey.from_str(co_markers.metadata['genotypes'][cb]).genotype
    except (KeyError, ValueError):
        hap1, hap2 = 'hap1', 'hap2'
    axes[0].set_ylabel(f'Marker coverage ({hap1} vs {hap2})')
    if co_preds is not None:
        if co_preds.ploidy_type == 'haploid':
            cmap = LinearSegmentedColormap.from_list('haploid_cmap', [ref_colour, alt_colour])
            norm = Normalize(0, 1)
        elif co_preds.ploidy_type == 'diploid_bc1':
            cmap = LinearSegmentedColormap.from_list('bc1_cmap', [ref_colour, alt_colour])
            states = co_preds.metadata.get('rhmm_params', {}).get('states', None)
            if states is None:
                norm = Normalize(0, 1)
            else:
                states.sort()
                norm = Normalize(0, 0.5) if states[0][0] == states[0][1] == 0 else Normalize(0.5, 1)
        else:
            cmap = LinearSegmentedColormap.from_list('f2_cmap', [ref_colour, '#999999', alt_colour])
            norm = Normalize(0, 1)
    if show_gt:
        try:
            gt = co_markers.metadata['ground_truth'][cb]
        except KeyError:
            show_gt = False
            gt = None

    if max_yheight == 'auto':
        m = np.concatenate(list(co_markers[cb].values()))
        max_yheight = np.percentile(m, 99.5)
    ylim_offset = max_yheight * 0.05

    for chrom, ax in zip(chrom_sizes, axes):

        ax.set_xlim(-XLIM_OFFSET, chrom_sizes[chrom] + XLIM_OFFSET)
        ylims = np.array([-max_yheight - ylim_offset, max_yheight + ylim_offset])
        ax.set_ylim(*ylims)

        chrom_markerplot(
            co_markers[cb, chrom],
            chrom_sizes[chrom],
            co_markers.bin_size,
            ax=ax,
            max_yheight=max_yheight,
            ref_colour=ref_colour,
            alt_colour=alt_colour,
        )
        if co_preds is not None:
            hp = co_preds[cb, chrom]
            if show_mesh_prob:
                _add_co_prob_colormesh(
                    ax, hp, co_markers.chrom_sizes[chrom], co_markers.bin_size, ylims, cmap, norm
                )
            if annotate_co_number:
                p_co = np.abs(np.diff(hp))
                p_co = np.where(p_co > nco_min_prob_change, p_co, 0)
                n_co = p_co.sum()
                if co_preds.ploidy_type.startswith('diploid'):
                    n_co *= 2
                ax.annotate(text=f'{n_co.sum():.2f} COs', xy=(0.05, 0.05), xycoords='axes fraction')
        if show_gt:
            _add_gt_vlines(
                ax, gt[chrom], co_markers.bin_size, ylims
            )
    plt.tight_layout()
    return fig, axes


def plot_recombination_landscape(co_preds, co_markers=None,
                                 apply_per_geno=False,
                                 rolling_mean_window_size=1_000_000,
                                 nboots=100, ci=95,
                                 min_prob=5e-3,
                                 axes=None,
                                 colour=None,
                                 figsize=(12, 4),
                                 use_cached=False,
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
    apply_per_geno : bool, optional
        Whether to group barcodes by genotype for recombination landscape calculation/plotting. Default is True.
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
    use_cached : bool, optional
        Whether to use the metadata slot "recombination_landscape" of co_preds, if it exists
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
    else:
        fig = plt.gcf()
        assert len(axes) == len(co_preds.chrom_sizes)
        if len(co_preds.chrom_sizes) == 1 and isinstance(axes, mpl.axes.Axes):
            axes = [axes,]

    if not 'genotypes' in co_preds.metadata:
        apply_per_geno = False

    if use_cached and 'recombination_landscape' in co_preds.metadata:
        cm_per_mb = co_preds.metadata['recombination_landscape']
    else:
        cm_per_mb = recombination_landscape(
            co_preds,
            co_markers=co_markers,
            apply_per_geno=apply_per_geno,
            rolling_mean_window_size=rolling_mean_window_size,
            nboots=nboots,
            min_prob=min_prob,
            rng=rng
        )

    lower = (100 - ci) / 2
    upper = 100 - lower

    for geno, geno_cm_per_mb in cm_per_mb.items():
        curr_colour = colour
        for chrom, ax in zip(geno_cm_per_mb, axes):
            x = np.arange(0, co_preds.nbins[chrom]) * co_preds.bin_size
            c = geno_cm_per_mb[chrom]
            line, = ax.step(x, np.nanmean(c, axis=0), color=curr_colour)
            # when colour is None this guarantees shading and label colours are correct
            curr_colour = line.get_color()
            ax.fill_between(
                x=x,
                y1=np.nanpercentile(c, lower, axis=0),
                y2=np.nanpercentile(c, upper, axis=0),
                alpha=0.25,
                color=curr_colour
            )
        if apply_per_geno:
            axes[-1].plot([], [], color=colour, label=geno)
            axes[-1].legend()

    axes[0].set_ylabel('cM / Mb')
    plt.tight_layout()
    return fig, axes


def plot_allele_ratio(co_preds,
                      apply_per_geno=False,
                      nboots=100, ci=95,
                      axes=None,
                      figsize=(12, 4),
                      colour=None,
                      label=None,
                      rng=DEFAULT_RNG):
    """
    Plot the allele ratio across chromosomes for multiple cells with bootstrapped confidence intervals.

    Parameters
    ----------
    co_preds : PredictionRecords
        PredictionRecords object containing the haplotype predictions.
    apply_per_geno : bool, optional
        Whether to group barcodes by genotype for allele frequency calculation/plotting. Default is True.
    nboots : int, optional
        The number of bootstrap iterations for calculating confidence intervals. Default is 100.
    ci : int, optional
        The confidence interval percentage. Default is 95.
    axes : list of matplotlib.axes.Axes, optional
        The axes to plot on. If None, new axes are created. Default is None.
    figsize : tuple, optional
        The size of the figure (width, height) in inches. Default is (12, 4).
    colour : str, optional
        The colour to use for the plot lines and fills. If None, a colour is selected from the default palette.
        Default is None.
    label : str, optional
        The label for the plot legend. If None, no legend is added. Default is None.
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
    else:
        fig = plt.gcf()
        assert len(axes) == len(co_preds.chrom_sizes)
        if len(co_preds.chrom_sizes) == 1 and isinstance(axes, mpl.axes.Axes):
            axes = [axes,]

    if not 'genotypes' in co_preds.metadata:
        apply_per_geno = False

    lower = (100 - ci) / 2
    upper = 100 - lower

    for geno, geno_co_preds in co_preds.groupby('genotype' if apply_per_geno else 'none'):
        N = len(geno_co_preds)
        colour = None
        for chrom, ax in zip(geno_co_preds.chrom_sizes, axes):
            x = np.arange(0, geno_co_preds.nbins[chrom]) * geno_co_preds.bin_size
            haps = geno_co_preds[:, chrom].stack_values()
            c = []
            for _ in range(nboots):
                idx = rng.integers(0, N, size=N)
                c.append(haps[idx].mean(axis=0))
            line, = ax.step(x, np.nanmean(c, axis=0), color=colour)
            colour = line.get_color()
            ax.fill_between(
                x=x,
                y1=np.nanpercentile(c, lower, axis=0),
                y2=np.nanpercentile(c, upper, axis=0),
                alpha=0.25,
                color=colour
            )
        if apply_per_geno:
            axes[-1].plot([], [], color=colour, label=geno)
            axes[-1].legend()
    axes[0].set_ylabel('Allele ratio')

    if co_preds.ploidy_type == "diploid_bc1":
        rhmm_params = co_preds.metadata.get('rhmm_params', {})
        states = rhmm_params.get('states', None)
        if states is None:
            ylim = (0, 1)
        else:
            states.sort()
            ylim = (0, 0.5) if states[0][0] == states[0][1] == 0 else (0.5, 1)
    else:
        ylim = (0, 1)

    axes[0].set_ylim(*ylim)

    plt.tight_layout()
    return fig, axes


def _plot_segdist_1d(seg_dist, chrom_sizes, fig, axes, colour=None, label=None):
    for chrom, ax in zip(chrom_sizes, axes):
        chrom_sd = seg_dist.query('chrom_1 == @chrom')
        ax.step(chrom_sd.pos_1.values, chrom_sd.lod_score.values, color=colour)
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
                                    apply_per_geno=False, nboots=100, ci=95,
                                    min_dist=None, max_dist=None, step_size=1e6,
                                    only_adjacent=False, chroms=None, show_L_int=True,
                                    axes=None,
                                    figsize=(12, 4),
                                    rng=DEFAULT_RNG):
    """
    Plot the coefficient-of-coincidence (CoC) curve with bootstrap confidence intervals.

    Parameters
    ----------
    co_preds : PredictionRecords
        haplotype predictions object with metadata slot crossover_samples
    apply_per_geno : bool, optional
        If True, group ``co_preds`` by genotype and plot one curve per group.
        Ignored if genotype metadata is missing.
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
    else:
        fig = plt.gcf()
        assert len(axes) == len(chrom_sizes)
        if len(co_preds.chrom_sizes) == 1 and isinstance(axes, mpl.axes.Axes):
            axes = [axes,]


    if not 'genotypes' in co_preds.metadata:
        apply_per_geno = False

    lower = (100 - ci) / 2
    upper = 100 - lower

    for geno, geno_co_preds in co_preds.groupby('genotype' if apply_per_geno else 'none'):
        N = len(geno_co_preds)
        colour = None
        geno_coc, x, l_int = coefficient_of_coincidence(
            geno_co_preds, nboots=nboots, chroms=chroms, only_adjacent=only_adjacent,
            min_dist=min_dist, max_dist=max_dist, step_size=step_size,
        )
        for chrom, ax in zip(chrom_sizes, axes):
            line, = ax.plot(x[chrom], np.nanmean(geno_coc[chrom], axis=0), color=colour)
            colour = line.get_color()
            ax.fill_between(
                x=x[chrom],
                y1=np.nanpercentile(geno_coc[chrom], lower, axis=0),
                y2=np.nanpercentile(geno_coc[chrom], upper, axis=0),
                alpha=0.25,
                color=colour
            )
            if show_L_int:
                ax.axvline(np.nanmean(l_int[chrom]), color=colour)
                ax.axvspan(np.nanpercentile(l_int[chrom], lower),
                           np.nanpercentile(l_int[chrom], upper),
                           color=colour, alpha=0.25, zorder=0)

        if apply_per_geno:
            axes[-1].plot([], [], color=colour, label=geno)

    for chrom, ax in zip(chrom_sizes, axes):
        ax.axhline(1, color='#252525', zorder=-1)
        ax.axhline(0.5, color='#252525', zorder=-1, ls='--')
        ax.set_xlabel(f'{chrom} inter-CO distance (Mb)')

    axes[0].set_ylabel('Coefficient of coincidence')
    if apply_per_geno:
        axes[-1].legend(loc=2, title='Genotype')
    plt.tight_layout()
    return fig, axes


def run_plot(cell_barcode, marker_json_fn, pred_json_fn, output_fig_fn=None,
             cb_whitelist_fn=None, plot_type='markerplot', figsize=(18, 4), display_plot=False,
             show_pred=True, show_co_num=True, show_gt=True, max_yheight=20,
             window_size=1_000_000, nboots=100, confidence_intervals=95,
             ref_colour='#0072b2', alt_colour='#d55e00', rng=DEFAULT_RNG):
    """
    Generate and save a plot for the given cell barcode and crossover marker data.

    This function loads crossover marker data and optional prediction data, then generates a plot based
    on the specified plot type. It can create either a marker plot or a recombination landscape plot, 
    with various customizable options for visualizing the data.

    Parameters
    ----------
    cell_barcode : str
        The cell barcode to plot data for. Required if `plot_type` is 'markerplot'.
    marker_json_fn : str
        File path to the JSON file containing crossover marker data.
    pred_json_fn : str, optional
        File path to the JSON file containing crossover prediction data. If None, no predictions are used.
    output_fig_fn : str, optional
        File path to save the generated plot. If None, the plot is not saved. Default is None.
    cb_whitelist_fn : str, optional
        File path to a whitelist of cell barcodes to include. Default is None (no filtering).
    plot_type : str, optional
        The type of plot to generate. Can be 'markerplot' or 'recombination'. Default is 'markerplot'.
    figsize : tuple of (float, float), optional
        The size of the plot figure in inches. Default is (18, 4).
    display_plot : bool, optional
        Whether to display the plot using `plt.show()`. Default is False (do not display).
    show_pred : bool, optional
        Whether to display crossover prediction probability mesh in the plot. Default is True.
    show_co_num : bool, optional
        Whether to annotate the number of crossovers in the plot. Default is True.
    show_gt : bool, optional
        Whether to show genotype lines in the plot. Default is True.
    max_yheight : float, optional
        The maximum y-axis height for the plot. Default is 20.
    window_size : int, optional
        The rolling window size (in base pairs) for calculating recombination landscapes. Default is 1,000,000.
    nboots : int, optional
        The number of bootstrap iterations for calculating confidence intervals in recombination landscapes. Default is 100.
    confidence_intervals : int, optional
        The confidence interval percentage for recombination landscape plots. Default is 95.
    ref_colour : str, optional
        The color for the reference allele in the plot. Default is '#0072b2'.
    alt_colour : str, optional
        The color for the alternate allele in the plot. Default is '#d55e00'.
    rng : numpy.random.Generator, optional
        The random number generator for bootstrapping. Default is the global `DEFAULT_RNG`.

    Returns
    -------
    None
        This function generates and optionally saves a plot. No value is returned.

    Raises
    ------
    ValueError
    - If `plot_type` is 'markerplot' and `cell_barcode` is None, a ValueError is raised.
    - If `plot_type` is 'recombination' and `pred_json_fn` is None, a ValueError is raised.

    Notes
    -----
    - For `plot_type='markerplot'`, a single-cell marker plot is generated for the specified cell barcode.
    - For `plot_type='recombination'`, a recombination landscape plot is generated for the whole dataset,
      based on the haplotype predictions.
    """
    co_markers = load_json(marker_json_fn, cb_whitelist_fn=cb_whitelist_fn, bin_size=None)
    if pred_json_fn is not None:
        co_preds = load_json(
            pred_json_fn, cb_whitelist_fn=None, bin_size=None, data_type='predictions'
        )
    else:
        if plot_type == 'recombination':
            raise ValueError('Must specify a pred_json_fn for plot-type "recombination"')
        co_preds = None

    if plot_type == 'markerplot':
        if cell_barcode is None:
            raise ValueError('Must specify a cell barcode for plot-type "markerplot"')
        single_cell_markerplot(
            cell_barcode,
            co_markers,
            co_preds=co_preds,
            figsize=figsize,
            show_mesh_prob=show_pred,
            annotate_co_number=show_co_num,
            nco_min_prob_change=nco_min_prob_change,
            show_gt=show_gt,
            max_yheight=max_yheight,
            ref_colour=ref_colour,
            alt_colour=alt_colour
        )
    elif plot_type == 'recombination':
        plot_recombination_landscape(
            co_preds, co_markers,
            rolling_mean_window_size=window_size,
            nboots=nboots, ci=confidence_intervals,
            min_prob=nco_min_prob_change,
            figsize=figsize,
            colour=ref_colour,
            rng=rng
        )
    if output_fig_fn is not None:
        plt.savefig(output_fig_fn)
    if display_plot:
        #plt.switch_backend('TkAgg')
        plt.show()
