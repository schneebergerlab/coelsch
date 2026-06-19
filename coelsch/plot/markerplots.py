"""
Single-cell marker plotting utilities.
"""
import re

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap

from coelsch.experiment.genotypes import GenotypeKey
from coelsch.stats import n_crossovers

from .core import chrom_subplots, XLIM_OFFSET


def chrom_markerplot(co_markers, chrom_size, bin_size, ax=None, max_yheight=20, linewidths=1.0,
                     ref_colour='#0072b2', alt_colour='#d55e00', ori_colour='#252525',
                     ref_idx=0, alt_idx=1):
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
    ref_markers = co_markers[:, ref_idx].copy()
    ref_markers[ref_markers > max_yheight] = max_yheight
    alt_markers = co_markers[:, alt_idx].copy()
    alt_markers[alt_markers > max_yheight] = max_yheight
    alt_markers = np.negative(alt_markers)

    nbins = len(co_markers)
    pos = np.arange(nbins) * bin_size
    base = np.repeat(0, nbins)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.vlines(pos, base, ref_markers, color=ref_colour, zorder=0, linewidths=linewidths)
    ax.vlines(pos, base, alt_markers, color=alt_colour, zorder=0, linewidths=linewidths)
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


def chrom_markerplot_three_way(co_markers, chrom_size, bin_size, ax=None,
                               max_yheight=20, linewidths=1.0,
                               ref_colour='#0072b2', alt_colour='#d55e00',
                               alt2_colour='#009e73', ori_colour='#252525'):
    shared_markers = co_markers[:, 0].copy()
    shared_markers[shared_markers > max_yheight] = max_yheight
    alt1_markers = co_markers[:, 1].copy()
    alt1_markers[alt1_markers > max_yheight] = max_yheight
    alt2_markers = co_markers[:, 2].copy()
    alt2_markers[alt2_markers > max_yheight] = max_yheight

    nbins = len(co_markers)
    pos = np.arange(nbins) * bin_size
    base = np.repeat(0, nbins)
    alt1_base = np.negative(alt1_markers)
    alt2_base = np.negative(alt1_markers + alt2_markers)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.vlines(pos, base, shared_markers, color=ref_colour, zorder=0, linewidths=linewidths)
    ax.vlines(pos, base, alt1_base, color=alt_colour, zorder=0, linewidths=linewidths)
    ax.vlines(pos, alt1_base, alt2_base, color=alt2_colour, zorder=0, linewidths=linewidths)
    ax.plot([0, chrom_size], [0, 0], ls='-', color=ori_colour)
    return ax


def _three_way_prediction_probabilities(co_preds, cb, chrom):
    dosage = co_preds.get_haplotype_dosage(cb, chrom)
    m1_denom = dosage[:, 0] + dosage[:, 1]
    m2_denom = dosage[:, 0] + dosage[:, 2]
    m1 = dosage[:, 1] / np.maximum(m1_denom, 1e-12)
    m2 = dosage[:, 2] / np.maximum(m2_denom, 1e-12)
    return m1, m2


def _single_cell_markerplot_three_way(cb, co_markers, *, co_preds=None, figsize=(18, 4),
                                      chroms=None, show_mesh_prob=True,
                                      annotate_co_number=True, nco_min_prob_change=5e-3,
                                      show_gt=True, max_yheight='auto', linewidths=1.0,
                                      ref_colour='#0072b2', alt_colour='#d55e00',
                                      alt2_colour='#009e73'):
    chrom_sizes = co_markers.chrom_sizes
    if chroms is not None:
        chrom_sizes = {c: chrom_sizes[c] for c in chroms}
    fig, axes = chrom_subplots(chrom_sizes, figsize=figsize)

    hap0 = _haplotype_name(co_markers, cb, 0)
    hap1 = _haplotype_name(co_markers, cb, 1)
    hap2 = _haplotype_name(co_markers, cb, 2)
    axes[0].set_ylabel(f'Marker coverage\n{hap0} vs {hap1}+{hap2}')

    cmap_m1 = LinearSegmentedColormap.from_list('three_way_m1_cmap', [ref_colour, alt_colour])
    cmap_m2 = LinearSegmentedColormap.from_list('three_way_m2_cmap', [ref_colour, alt2_colour])
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

        chrom_markerplot_three_way(
            co_markers[cb, chrom],
            chrom_sizes[chrom],
            co_markers.bin_size,
            ax=ax,
            max_yheight=max_yheight,
            linewidths=linewidths,
            ref_colour=ref_colour,
            alt_colour=alt_colour,
            alt2_colour=alt2_colour,
        )
        if co_preds is not None:
            if show_mesh_prob:
                m1, m2 = _three_way_prediction_probabilities(co_preds, cb, chrom)
                _add_co_prob_colormesh(
                    ax, m1, co_markers.chrom_sizes[chrom], co_markers.bin_size,
                    np.array([0, ylims[1]]), cmap_m1, norm,
                )
                _add_co_prob_colormesh(
                    ax, m2, co_markers.chrom_sizes[chrom], co_markers.bin_size,
                    np.array([ylims[0], 0]), cmap_m2, norm,
                )
            if annotate_co_number:
                n_co = n_crossovers(
                    {chrom: co_preds[cb, chrom]},
                    min_co_prob=nco_min_prob_change,
                )
                ax.annotate(text=f'{n_co:.2f} COs', xy=(0.05, 0.05), xycoords='axes fraction')
        if show_gt:
            _add_gt_vlines(ax, gt[chrom], co_markers.bin_size, ylims)

    plt.tight_layout()
    return fig, axes


def _markerplot_comparisons(co_markers):
    params = co_markers.experiment_params

    if params.genotyping_strategy == 'recombinant':
        return (('haplotypes', 0, 1),)

    if params.crossing_strategy in {'f1', 'f2', 'backcross'}:
        return (('haplotypes', 0, 1),)

    if params.crossing_strategy == 'testcross':
        return (('segregating', 1, 2),)

    if params.crossing_strategy == 'four_way':
        return (
            ('meiosis 1', 0, 1),
            ('meiosis 2', 2, 3),
        )

    raise NotImplementedError(
        f'single-cell marker plots are not implemented for {params.crossing_strategy!r}'
    )


def _comparison_prediction_probability(co_preds, cb, chrom, ref_idx, alt_idx):
    dosage = co_preds.get_haplotype_dosage(cb, chrom)
    denom = dosage[:, ref_idx] + dosage[:, alt_idx]
    return dosage[:, alt_idx] / np.maximum(denom, 1e-12)


def _haplotype_name(co_markers, cb, hap_idx):
    genotypes = co_markers.metadata.get('genotypes', {})
    if cb not in genotypes:
        return f'hap{hap_idx}'

    genotype = GenotypeKey.from_any(genotypes[cb])
    return str(genotype.founders[hap_idx])


def single_cell_markerplot(cb, co_markers, *, co_preds=None, figsize=(18, 4), chroms=None,
                           show_mesh_prob=True, annotate_co_number=True,
                           nco_min_prob_change=5e-3, show_gt=True, max_yheight='auto',
                           linewidths=1.0, ref_colour='#0072b2', alt_colour='#d55e00'):
    """
    Plot marker coverage and prediction probabilities for one cell barcode.
    """
    if cb not in co_markers.barcodes:
        raise KeyError(f'cb {cb} not in co_marker object')

    params = co_markers.experiment_params
    if params.crossing_strategy == 'three_way' and params.genotyping_strategy != 'recombinant':
        return _single_cell_markerplot_three_way(
            cb,
            co_markers,
            co_preds=co_preds,
            figsize=figsize,
            chroms=chroms,
            show_mesh_prob=show_mesh_prob,
            annotate_co_number=annotate_co_number,
            nco_min_prob_change=nco_min_prob_change,
            show_gt=show_gt,
            max_yheight=max_yheight,
            linewidths=linewidths,
            ref_colour=ref_colour,
            alt_colour=alt_colour,
        )

    comparisons = _markerplot_comparisons(co_markers)
    chrom_sizes = co_markers.chrom_sizes
    if chroms is not None:
        chrom_sizes = {c: chrom_sizes[c] for c in chroms}

    nrows = len(comparisons)
    if nrows == 1:
        fig, axes = chrom_subplots(chrom_sizes, figsize=figsize)
        axes = np.asarray([axes], dtype=object)
    else:
        width, height = figsize
        fig, axes = plt.subplots(
            figsize=(width, height * nrows),
            nrows=nrows,
            ncols=len(chrom_sizes),
            width_ratios=list(chrom_sizes.values()),
            sharex='col',
            sharey='row',
            squeeze=False,
        )
        for row_axes in axes:
            for chrom, ax in zip(chrom_sizes, row_axes):
                ax.set_xlim(-XLIM_OFFSET, chrom_sizes[chrom] + XLIM_OFFSET)
                xticks = np.arange(0, chrom_sizes[chrom], 1e7)
                ax.set_xticks(xticks)
                ax.set_xticklabels([int(i // 1e6) for i in xticks])
                chrom_label = re.sub('^[Cc]hr', '', chrom)
                ax.set_xlabel(f'Chromosome {chrom_label} (Mb)')

    cmap = LinearSegmentedColormap.from_list('haplotype_cmap', [ref_colour, alt_colour])
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

    for row_idx, (comparison_label, ref_idx, alt_idx) in enumerate(comparisons):
        hap1 = _haplotype_name(co_markers, cb, ref_idx)
        hap2 = _haplotype_name(co_markers, cb, alt_idx)
        axes[row_idx, 0].set_ylabel(
            f'{comparison_label}\nMarker coverage ({hap1} vs {hap2})'
        )

        for chrom, ax in zip(chrom_sizes, axes[row_idx]):
            ax.set_xlim(-XLIM_OFFSET, chrom_sizes[chrom] + XLIM_OFFSET)
            ylims = np.array([-max_yheight - ylim_offset, max_yheight + ylim_offset])
            ax.set_ylim(*ylims)

            chrom_markerplot(
                co_markers[cb, chrom],
                chrom_sizes[chrom],
                co_markers.bin_size,
                ax=ax,
                max_yheight=max_yheight,
                linewidths=linewidths,
                ref_colour=ref_colour,
                alt_colour=alt_colour,
                ref_idx=ref_idx,
                alt_idx=alt_idx,
            )
            if co_preds is not None:
                if show_mesh_prob:
                    hp = _comparison_prediction_probability(co_preds, cb, chrom, ref_idx, alt_idx)
                    _add_co_prob_colormesh(
                        ax, hp, co_markers.chrom_sizes[chrom], co_markers.bin_size, ylims, cmap, norm
                    )
                if annotate_co_number and row_idx == 0:
                    n_co = n_crossovers(
                        {chrom: co_preds[cb, chrom]},
                        min_co_prob=nco_min_prob_change,
                    )
                    ax.annotate(text=f'{n_co:.2f} COs', xy=(0.05, 0.05), xycoords='axes fraction')
            if show_gt and row_idx == 0:
                _add_gt_vlines(
                    ax, gt[chrom], co_markers.bin_size, ylims
                )
    plt.tight_layout()
    return fig, axes[0] if nrows == 1 else axes


