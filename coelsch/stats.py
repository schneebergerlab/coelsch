import logging
import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d

from .utils import load_json


log = logging.getLogger('coelsch')


def total_markers(cb_co_markers):
    """
    Calculates the total number of markers for a cell barcode.

    Parameters
    ----------
    cb_co_markers : dict
        A dictionary where keys are chromosomes and values are arrays representing marker counts.

    Returns
    -------
    float
        The log-transformed total number of markers (base 10).
    """
    tot = 0
    for m in cb_co_markers.values():
        tot += m.sum(axis=None)
    return np.log10(tot) if tot else 0


def n_crossovers(cb_co_preds, min_co_prob=5e-3):
    """
    Count expected crossovers from haplotype dosage changes along the genome.

    Scalar two-haplotype records store haplotype-1 dosage, so absolute changes
    directly count changed inherited copies. Multi-haplotype records store full
    dosage rows, where half the L1 dosage change counts the number of changed
    inherited copies.
    """
    nco = 0
    for p in cb_co_preds.values():
        if p.ndim == 1:
            p_co = np.abs(np.diff(p, axis=0))
        elif p.ndim == 2:
            p_co = 0.5 * np.abs(np.diff(p, axis=0)).sum(axis=1)
        else:
            raise ValueError('prediction arrays must be 1D or 2D')

        p_co = np.where(p_co >= min_co_prob, p_co, 0)
        nco += p_co.sum(axis=None)
    return nco



def _error_rate(n, d, pseudo=0.5):
    return (d - n + pseudo) / (d + pseudo + pseudo)
    

def _chrom_agreement(m, dosage):
    if dosage.ndim != 2:
        raise ValueError('prediction dosage array must be 2D')

    if m.shape != dosage.shape:
        raise ValueError(
            f"marker and prediction shapes do not match: {m.shape} != {dosage.shape}"
        )

    total = m.sum(axis=None)
    if total <= 0:
        return 0.0, 0.0

    row_totals = dosage.sum(axis=1, keepdims=True)
    hap_probs = dosage / np.maximum(row_totals, 1e-12)
    agreement = (m * hap_probs).sum(axis=None)
    return agreement, total


def marker_agreement_score(cb_co_markers, cb_co_preds, max_score=10, pseudo=0.5):
    agreement = 0.0
    total = 0.0

    for chrom, m in cb_co_markers.items():
        n, d = _chrom_agreement(m, cb_co_preds[chrom])
        agreement += n
        total += d

    if total <= 0:
        return np.nan

    error_rate = _error_rate(agreement, total, pseudo)
    with np.errstate(divide='ignore'):
        return np.minimum(-np.log2(error_rate), max_score)


def aneuploidy_score(cb_co_markers, cb_co_preds, pseudo=0.5):
    noms = []
    denoms = []
    max_error_idx = None
    max_error_rate = 0.0
    for i, (chrom, m) in enumerate(cb_co_markers.items()):
        p = cb_co_preds[chrom]
        n, d = _chrom_agreement(m, p)
        noms.append(n)
        denoms.append(d)
        e = _error_rate(n, d, pseudo)
        if e > max_error_rate:
            max_error_idx = i
            max_error_rate = e
    max_error_nom = noms.pop(max_error_idx)
    max_error_denom = denoms.pop(max_error_idx)
    bg_error_nom = sum(noms)
    bg_error_denom = sum(denoms)

    return np.log2(
        _error_rate(max_error_nom, max_error_denom, pseudo) / 
        _error_rate(bg_error_nom, bg_error_denom, pseudo)
    )
    

def prediction_uncertainty_score(cb_co_pred_dosage):
    """
    Calculates uncertainty as deviation from hard haplotype dosage calls.
    """
    auc = 0
    for p in cb_co_pred_dosage.values():
        hu = np.abs(p - np.round(p)).sum(axis=1) / p.sum(axis=1)
        auc += np.trapz(hu)
    with np.errstate(divide='ignore'):
        return np.maximum(np.log10(auc), 0)


def marker_span_score(cb_co_markers, max_score=10):
    """
    Calculates a marker span score for a cell barcode.

    Parameters
    ----------
    cb_co_markers : dict
        A dictionary where keys are chromosomes and values are arrays representing marker data.
    max_score : int, optional
        The maximum score for coverage (default is 10).

    Returns
    -------
    float
        The marker span score on a phred-like scale, capped at the provided `max_score`.
    """
    cov = 0
    tot = 0
    for m in cb_co_markers.values():
        idx, = np.nonzero(m.sum(axis=1))
        if idx.size > 0:
            cov += idx[-1] - idx[0] + 1
        tot += len(m)
    ratio = cov / tot if tot > 0 else 0
    delta = 1 - ratio
    if delta <= 2 ** -max_score:
        return max_score
    return -np.log2(delta)


def haplotype_dosage_bias(cb_co_pred_dosage, expected_dosage):
    dosage = np.concatenate(list(cb_co_pred_dosage.values()), axis=0)
    observed = dosage.mean(axis=0)
    expected_dosage = np.asarray(expected_dosage, dtype=np.float32)
    return np.abs(observed - expected_dosage).sum() / expected_dosage.sum()


def calculate_prediction_metrics(co_markers, co_preds, nco_min_prob=2.5e-3, max_phred_score=10):
    """
    Calculates prediction metrics for each cell barcode's marker and prediction data.

    Parameters
    ----------
    co_markers : MarkerRecords
        A MarkerRecords object representing observed marker data.
    co_preds : PredictionRecords
        A PredictionRecords object representing predicted haplotypes for the dataset.
    nco_min_prob : float, optional
        The minimum difference in haplotype probability between two bins for calculating crossovers (default is 2.5e-3).
    max_phred_score : int, optional
        The maximum score for metrics calculated on phred-like scale (default is 10).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the calculated prediction metrics for each cell barcode.
    """
    qual_metrics = []

    genotypes = co_markers.metadata.get('genotypes', {})
    genotype_probs = co_markers.metadata.get('genotype_probability', {})
    genotype_nmarkers = co_markers.metadata.get('genotyping_nmarkers', {})
    bg_frac = co_markers.metadata.get('estimated_background_fraction', {})
    doublet_rate = co_preds.metadata.get('doublet_probability', {})
    expected_dosage = co_preds.experiment_params.haplotype_dosage

    for cb, cb_co_markers in co_markers.items():
        cb_co_preds = co_preds[cb]
        cb_co_pred_dosage = {
            chrom: co_preds.get_haplotype_dosage(cb, chrom)
            for chrom in co_preds.chrom_sizes
        }
        qual_metrics.append([
            cb,
            genotypes.get(cb, None),
            genotype_probs.get(cb, np.nan),
            np.log10(genotype_nmarkers.get(cb, np.nan)),
            total_markers(cb_co_markers),
            bg_frac.get(cb, np.nan),
            n_crossovers(cb_co_preds, min_co_prob=nco_min_prob),
            marker_agreement_score(cb_co_markers, cb_co_pred_dosage, max_score=max_phred_score),
            prediction_uncertainty_score(cb_co_pred_dosage),
            doublet_rate.get(cb, np.nan),
            marker_span_score(cb_co_markers),
            haplotype_dosage_bias(cb_co_pred_dosage, expected_dosage)
        ])
    qual_metrics = pd.DataFrame(
        qual_metrics,
        columns=['cb', 'geno_pred', 'geno_prob', 'geno_n_marker_reads',
                 'co_n_marker_reads', 'bg_fraction', 'n_crossovers',
                 'marker_agreement_score', 'prediction_uncertainty_score',
                 'doublet_probability',
                 'marker_span_score', 'haplotype_dosage_bias']
    )
    return qual_metrics


def gt_haplotype_accuracy_score(cb_co_preds, cb_co_gt, thresholded=False, max_score=10):
    """
    Calculates the accuracy score for genotypes based on predicted and ground truth haplotypes.

    Parameters
    ----------
    cb_co_preds : dict
        A dictionary where keys are chromosomes and values are arrays representing haplotype probabilities.
    cb_co_gt : dict
        A dictionary where keys are chromosomes and values are arrays representing ground truth haplotypes.
    thresholded : bool, optional
        If True, thresholds the predictions at 0.5 before calculating the accuracy score (default is False).
    max_score : int, optional
        The maximum score for accuracy (default is 10).

    Returns
    -------
    float
        The haplotype accuracy score on a phred-like scale, capped at the provided `max_score`.
    """
    dev = 0
    nbins = 0
    for chrom, p in cb_co_preds.items():
        if thresholded:
            p = (p > 0.5).astype(np.float32)
        gt = cb_co_gt[chrom]
        dev += np.abs(p - gt).sum(axis=None)
        nbins += len(p)
    with np.errstate(divide='ignore'):
        return np.minimum(-np.log2(dev / nbins), max_score)


def _co_score(p, gt, ws=40):
    assert not ws % 2
    filt = np.ones(ws) / ws
    filt[: ws // 2] = np.negative(filt[: ws // 2])
    gt_c = convolve1d((gt - 0.5) * 2, filt, mode='nearest')
    p_c = convolve1d((p - 0.5) * 2, filt, mode='nearest')
    return np.trapz(gt_c * p_c)


def gt_crossover_score(cb_co_preds, cb_co_gt, window_size=40):
    """
    Calculates the crossover score between predicted and ground truth haplotypes.

    Parameters
    ----------
    cb_co_preds : dict
        A dictionary where keys are chromosomes and values are arrays representing haplotype probabilities.
    cb_co_gt : dict
        A dictionary where keys are chromosomes and values are arrays representing ground truth haplotypes.
    window_size : int, optional
        The window size for the filter (default is 40).

    Returns
    -------
    float
        The calculated crossover score, or NaN if no crossovers are detected.
    """
    n_co = n_crossovers(cb_co_gt)
    if not n_co:
        return np.nan
    co = 0
    for chrom, p in cb_co_preds.items():
        gt = cb_co_gt[chrom]
        co += _co_score(p, gt, window_size)
    return np.log10(np.maximum(co / n_co, 1))


def _max_detectable_cos(m, gt):
    co_idx = np.where(np.diff(gt))[0] + 1
    m_seg = np.array_split(m, co_idx, axis=0)
    seg_haps = gt[np.insert(co_idx, 0, 0)].astype(int)
    supported_haps = []
    for seg, h in zip(m_seg, seg_haps):
        support = seg[:, h].sum()
        if support:
            supported_haps.append(h)
    return len(np.where(np.diff(supported_haps))[0])


def gt_detectable_crossovers(cb_co_markers, cb_co_gt):
    """
    Calculates the maximum number of crossovers from the ground truth that could possibly be detected using
    the given distribution of markers - some crossovers are invisible due to lack of markers in segments.

    Parameters
    ----------
    cb_co_markers : dict
        A dictionary where keys are chromosomes and values are arrays representing marker data.
    cb_co_gt : dict
        A dictionary where keys are chromosomes and values are arrays representing ground truth data.

    Returns
    -------
    int
        The total number of detectable crossovers across all barcodes.
    """
    dcos = 0
    for chrom, m in cb_co_markers.items():
        gt = cb_co_gt[chrom]
        dcos += _max_detectable_cos(m, gt)
    return dcos


def calculate_ground_truth_metrics(co_markers, co_preds, ground_truth, max_phred_score=10):
    """
    Calculates ground-truth benchmarking metrics for each cell barcode.

    Parameters
    ----------
    co_markers : MarkerRecords
        A MarkerRecords object representing observed marker data.
    co_preds : PredictionRecords
        A PredictionRecords object representing predicted haplotypes for the dataset.
    ground_truth : dict
        A PredictionRecords object representing ground truth data.
    max_phred_score : int, optional
        The maximum phred score (default is 10).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the calculated ground-truth metrics for each cell barcode.
    """
    score_metrics = []
    for cb, cb_co_preds in co_preds.items():
        cb_co_markers = co_markers[cb]
        if cb.split(':')[0] != 'doublet':
            cb_co_gt = ground_truth[cb]
            score_metrics.append([
                cb,
                int(n_crossovers(cb_co_gt)),
                gt_detectable_crossovers(cb_co_markers, cb_co_gt),
                gt_haplotype_accuracy_score(cb_co_preds, cb_co_gt, max_score=max_phred_score),
                gt_haplotype_accuracy_score(
                    cb_co_preds, cb_co_gt, thresholded=True, max_score=max_phred_score
                ),
                gt_crossover_score(cb_co_preds, cb_co_gt),
            ])
        else:
            score_metrics.append([
                cb, np.nan, np.nan, np.nan, np.nan, np.nan
            ])
    score_metrics = pd.DataFrame(
        score_metrics,
        columns=['cb', 'gt_n_crossovers', 'gt_detectable_crossovers',
                 'gt_haplotype_accuracy_score', 'gt_hardcall_accuracy_score', 'gt_crossover_score']
    )
    return score_metrics


def _write_metric_tsv(output_tsv_fn, qual_metrics, score_metrics=None, precision=3):
    '''
    Write the statistics to a tsv file using pandas
    '''
    if score_metrics is not None:
        qual_metrics = qual_metrics.merge(score_metrics, on='cb', how='outer')
    qual_metrics.to_csv(output_tsv_fn, sep='\t', index=False, float_format=f'%.{precision}g')


def run_stats(marker_json_fn, pred_json_fn, output_tsv_fn, *,
              co_markers=None, co_preds=None,
              cb_whitelist_fn=None, bin_size=25_000,
              nco_min_prob_change=2.5e-3, output_precision=3):
    """
    Scores the quality of data and predictions for a set of haplotype calls 
    generated with `predict`. This function computes prediction metrics and, if 
    available, benchmarking metrics using ground truth data, and writes the 
    results to a TSV file.

    Parameters
    ----------
    marker_json_fn : str
        Path to the JSON file containing the marker data.
    pred_json_fn : str
        Path to the JSON file containing the predicted data.
    output_tsv_fn : str
        Path where the output TSV file will be saved.
    co_markers : MarkerRecords, optional
        A MarkerRecords object (default is None, in which case it is loaded from `marker_json_fn`).
    co_preds : PredictionRecords, optional
        A PredictionRecords object (default is None, in which case  it is loaded from `pred_json_fn`).
    cb_whitelist_fn : str, optional
        Path to a file containing a whitelist of cell barcodes (default is None).
    bin_size : int, optional
        The size of the genomic bins for data (default is 25,000).
    nco_min_prob_change : float, optional
        The minimum crossover probability change (default is 2.5e-3).
    output_precision : int, optional
        The precision for floating point numbers in the output TSV (default is 3).

    Raises
    ------
    ValueError
        If the barcodes in `co_markers` and `co_preds` do not match.
    """
    if co_markers is None:
        co_markers = load_json(marker_json_fn, cb_whitelist_fn, bin_size)
    if co_preds is None:
        co_preds = load_json(
            pred_json_fn, cb_whitelist_fn, bin_size, data_type='predictions'
        )

    if set(co_preds.barcodes) != set(co_markers.barcodes):
        raise ValueError('Cell barcodes from marker-json-fn and predict-json-fn do not match')

    log.info('Calculating prediction metrics')
    qual_metrics = calculate_prediction_metrics(co_markers, co_preds)
    if 'ground_truth' in co_markers.metadata:
        # TODO: Rework sim.py ground-truth records and benchmarking metrics for
        # marginal haplotype dosage predictions before re-enabling this branch.
        log.warning(
            'Ground truth benchmarking metrics are temporarily disabled while '
            'simulation ground-truth records are updated for marginal haplotype '
            'dosage predictions'
        )
        score_metrics = None
    else:
        score_metrics = None

    log.info(f'Writing stats to {output_tsv_fn}')
    _write_metric_tsv(output_tsv_fn, qual_metrics, score_metrics, precision=output_precision)
