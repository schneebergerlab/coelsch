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


def chrom_markerplot(co_markers, chrom_size, bin_size, ax=None, max_yheight='auto', linewidths=1.0,
                     palette=['#0072b2', '#d55e00', '#009e73', '#a783c9'],
                     ref_idx=0, alt_idx=1, alt2_idx=None):
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
        The maximum y value for the plot. Use 'auto' to scale to the chromosome.
    palette : str, optional
        The color palette to use

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plotted markers.
    """
    if max_yheight == 'auto':
        max_yheight = np.percentile(co_markers, 99.5)
    ref_markers = co_markers[:, ref_idx].copy()
    ref_markers[ref_markers > max_yheight] = max_yheight
    alt_markers = co_markers[:, alt_idx].copy()
    alt_markers[alt_markers > max_yheight] = max_yheight
    alt_markers = np.negative(alt_markers)

    if alt2_idx is not None:
        alt2_markers = co_markers[:, alt2_idx].copy()
        alt2_markers = alt_markers - alt2_markers # stack on top of alt_markers
        alt2_markers[alt2_markers < -max_yheight] = -max_yheight
    else:
        alt2_markers = None

    nbins = len(co_markers)
    pos = np.arange(nbins) * bin_size
    base = np.repeat(0, nbins)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.vlines(pos, base, ref_markers, color=palette[ref_idx], zorder=0, linewidths=linewidths)
    ax.vlines(pos, base, alt_markers, color=palette[alt_idx], zorder=0, linewidths=linewidths)
    if alt2_markers is not None:
        ax.vlines(pos, alt_markers, alt2_markers, color=palette[alt2_idx], zorder=0, linewidths=linewidths)
    ax.plot([0, chrom_size], [0, 0], ls='-', color='#252525')
    return ax


def _calculate_haplotype_prob(co_preds, cb, chrom, ref_idx, alt_idx):
    dosage = co_preds.get_haplotype_dosage(cb, chrom)
    if ref_idx is None:
        return dosage[:, alt_idx]
    denom = dosage[:, ref_idx] + dosage[:, alt_idx]
    return dosage[:, alt_idx] / np.maximum(denom, 1e-12)


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


def _markerplot_comparisons(co_markers, palette):
    params = co_markers.experiment_params

    if len(palette) < 2:
        raise ValueError('Need at least 2 colour palette for markerplots')

    cmap = LinearSegmentedColormap.from_list('cmap_standard', palette[:2])

    if params.genotyping_strategy == 'recombinant':
        return ((0, 1, None, cmap, None),)

    if params.crossing_strategy in {'f1', 'f2', 'backcross'}:
        return ((0, 1, None, cmap, None),)

    if params.crossing_strategy == 'testcross':
        return ((1, 2, None, cmap, None),)

    if params.crossing_strategy == 'three_way':
        if len(palette) < 2:
            raise ValueError('Need at least 3 colour palette for three-way markerplots')
        cmap2 = LinearSegmentedColormap.from_list('cmap_three_way', [palette[0], palette[2]])
        return (
            (0, 1, 2, cmap, cmap2),
        )

    if params.crossing_strategy == 'four_way':
        if len(palette) < 4:
            raise ValueError('Need at least 4 colour palette for four-way markerplots')
        cmap3 = LinearSegmentedColormap.from_list('cmap_four_way', palette[2:4])
        return (
            (0, 1, None, cmap, None),
            (2, 3, None, cmap3, None),
        )

    raise NotImplementedError(
        f'single-cell marker plots are not implemented for {params.crossing_strategy!r}'
    )


def _markerplot_ylabel(co_markers, cb, ref_idx, alt_idx, alt2_idx=None):
    genotypes = co_markers.metadata.get('genotypes', {})
    if cb not in genotypes:
        label = f'hap{ref_idx} vs hap{alt_idx}'
        if alt2_idx is not None:
            label += f'/hap{alt2_idx}'
    else:
        genotype = GenotypeKey.from_any(genotypes[cb])
        label =f'{genotype.founders[ref_idx]} vs {genotype.founders[alt_idx]}'
        if alt2_idx is not None:
            label += f'/{genotype.founders[alt2_idx]}'
    return f'Informative read coverage\n{label}'


def single_cell_markerplot(cb, co_markers, *, co_preds=None, figsize=(18, 4), chroms=None,
                           show_mesh_prob=True, annotate_co_number=True,
                           nco_min_prob_change=5e-3, show_gt=True, max_yheight='auto',
                           linewidths=1.0, palette=['#0072b2', '#d55e00', '#009e73', '#663399']):
    """
    Plot marker coverage and prediction probabilities for one cell barcode.
    """
    if cb not in co_markers.barcodes:
        raise KeyError(f'cb {cb} not in co_marker object')

    comparisons = _markerplot_comparisons(co_markers, palette)
    chrom_sizes = co_markers.chrom_sizes
    if chroms is not None:
        chrom_sizes = {c: chrom_sizes[c] for c in chroms}

    nrows = len(comparisons)
    fig, axes = chrom_subplots(chrom_sizes, nrows=nrows, figsize=figsize)

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
    ylims = np.array([-max_yheight - ylim_offset, max_yheight + ylim_offset])

    for row_idx, (ref_idx, alt_idx, alt2_idx, cmap, alt_cmap) in enumerate(comparisons):
        axes[row_idx, 0].set_ylabel(
            _markerplot_ylabel(co_markers, cb, ref_idx, alt_idx, alt2_idx)
        )
        for chrom, ax in zip(chrom_sizes, axes[row_idx]):
            ax.set_xlim(-XLIM_OFFSET, chrom_sizes[chrom] + XLIM_OFFSET)
            ax.set_ylim(*ylims)

            chrom_markerplot(
                co_markers[cb, chrom],
                chrom_sizes[chrom],
                co_markers.bin_size,
                ax=ax,
                max_yheight=max_yheight,
                linewidths=linewidths,
                ref_idx=ref_idx,
                alt_idx=alt_idx,
                alt2_idx=alt2_idx,
                palette=palette,
            )
            if co_preds is not None:
                if show_mesh_prob:
                    if alt2_idx is None:
                        hp = _calculate_haplotype_prob(co_preds, cb, chrom, ref_idx, alt_idx)
                    else:
                        # three way special case
                        hp = _calculate_haplotype_prob(co_preds, cb, chrom, None, alt_idx)
                    _add_co_prob_colormesh(
                        ax, hp, co_markers.chrom_sizes[chrom],
                        co_markers.bin_size,
                        ylims if alt2_idx is None else [0, ylims[1]],
                        cmap, norm
                    )
                    if alt2_idx is not None:
                        # three way special case
                        hp2 = _calculate_haplotype_prob(co_preds, cb, chrom, None, alt2_idx)
                        _add_co_prob_colormesh(
                            ax, hp2, 
                            co_markers.chrom_sizes[chrom],
                            co_markers.bin_size,
                            [ylims[0], 0],
                            alt_cmap, norm
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


