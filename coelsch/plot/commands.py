"""
CLI dispatcher for plot commands.
"""
from matplotlib import pyplot as plt

from coelsch.utils import load_json

from .core import DEFAULT_RNG
from .markerplots import single_cell_markerplot
from .happlots import plot_recombination_landscape


def run_plot(cell_barcode, marker_json_fn, pred_json_fn, output_fig_fn=None,
             cb_whitelist_fn=None, plot_type='markerplot', figsize=(18, 4), display_plot=False,
             show_pred=True, show_co_num=True, show_gt=True, max_yheight=20,
             window_size=1_000_000, nboots=100, confidence_intervals=95,
             nco_min_prob_change=2.5e-3,
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
    nco_min_prob_change : float, optional
        Minimum probability change for crossover annotations and recombination landscapes.
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
