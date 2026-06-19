"""
Plotting API for coelsch.
"""
from .core import chrom_subplots, chrom2d_subplots, chrom2dtriangle_subplots
from .markerplots import (
    chrom_markerplot,
    chrom_markerplot_three_way,
    single_cell_markerplot,
)
from .happlots import (
    plot_recombination_landscape,
    plot_allele_ratio,
    plot_segregation_distortion,
    plot_coefficient_of_coincidence,
)
from .commands import run_plot

__all__ = [
    'chrom_subplots',
    'chrom2d_subplots',
    'chrom2dtriangle_subplots',
    'chrom_markerplot',
    'chrom_markerplot_three_way',
    'single_cell_markerplot',
    'plot_recombination_landscape',
    'plot_allele_ratio',
    'plot_segregation_distortion',
    'plot_coefficient_of_coincidence',
    'run_plot',
]
