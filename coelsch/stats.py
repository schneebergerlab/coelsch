import logging
import numpy as np
import pandas as pd
from .utils import load_json
from .records import PredictionRecords


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


def _marker_agreement_totals(cb_co_markers, cb_co_preds, thresholded=False):
    agreement = 0.0
    total = 0.0

    for chrom, m in cb_co_markers.items():
        p = cb_co_preds[chrom]
        if thresholded:
            p = np.round(p)
        n, d = _chrom_agreement(m, p)
        agreement += n
        total += d

    return agreement, total


def marker_agreement_fraction(cb_co_markers, cb_co_preds, thresholded=False):
    agreement, total = _marker_agreement_totals(
        cb_co_markers, cb_co_preds, thresholded=thresholded
    )
    if total <= 0:
        return np.nan
    return agreement / total


def marker_agreement_score(cb_co_markers, cb_co_preds, max_score=10, pseudo=0.5):
    agreement, total = _marker_agreement_totals(cb_co_markers, cb_co_preds)

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


def gt_haplotype_mae_score(cb_co_preds, cb_co_gt, thresholded=False, max_score=10):
    """
    Calculate a phred-like score from mean absolute haplotype dosage error.

    Parameters
    ----------
    cb_co_preds : dict
        A dictionary where keys are chromosomes and values are haplotype dosage arrays.
    cb_co_gt : dict
        A dictionary where keys are chromosomes and values are ground-truth haplotype dosage arrays.
    thresholded : bool, optional
        If True, round both predicted and ground-truth dosage before scoring.
    max_score : int, optional
        The maximum score (default is 10).

    Returns
    -------
    float
        The haplotype MAE score, capped at the provided `max_score`.
    """
    abs_error = 0.0
    total_dosage = 0.0

    for chrom, p in cb_co_preds.items():
        gt = cb_co_gt[chrom]

        if p.ndim != 2 or gt.ndim != 2:
            raise ValueError('gt_haplotype_mae_score requires dosage matrices')

        if p.shape != gt.shape:
            raise ValueError(
                f'prediction and ground truth shapes do not match: {p.shape} != {gt.shape}'
            )

        if thresholded:
            p = np.round(p)
            gt = np.round(gt)

        abs_error += np.abs(p - gt).sum(axis=None)
        total_dosage += gt.sum(axis=None)

    if total_dosage <= 0:
        return np.nan

    mae = abs_error / total_dosage
    if mae <= 0:
        return max_score

    with np.errstate(divide='ignore'):
        return np.minimum(-np.log2(mae), max_score)


def _dosage_edge_signal(dosage):
    """
    Convert haplotype dosage into crossover edge mass.

    Each edge is the expected number of inherited haplotype-copy switches
    between adjacent bins.
    """
    if dosage.ndim != 2:
        raise ValueError('crossover scoring requires dosage matrices')

    return 0.5 * np.abs(np.diff(dosage, axis=0)).sum(axis=1)


def _edge_window(edge, window_size):
    """
    Mark positions close enough to crossover edges to receive credit.

    The returned array is clipped to [0, 1], so overlapping windows do not give
    extra credit and perfect overlap remains bounded at precision/recall = 1.
    """
    if window_size <= 0:
        raise ValueError('window_size must be > 0')

    target = np.zeros_like(edge, dtype=float)
    half = window_size // 2

    for idx, mass in enumerate(edge):
        if mass <= 0:
            continue

        # Clip the local window at chromosome boundaries.
        start = max(0, idx - half)
        end = min(len(edge), idx + half + 1)
        target[start:end] = 1.0

    return target


def gt_crossover_precision_recall(cb_co_preds, cb_co_gt, window_size=40):
    """
    Calculate crossover precision and recall from haplotype dosage edges.

    Both inputs must be dictionaries of chrom -> haplotype dosage matrix. Recall
    asks what fraction of true crossover edge mass was recovered nearby, while
    precision asks what fraction of predicted crossover edge mass is near truth.
    """
    recall_overlap = 0.0
    precision_overlap = 0.0
    true_mass = 0.0
    pred_mass = 0.0

    for chrom, pred in cb_co_preds.items():
        gt = cb_co_gt[chrom]

        if pred.ndim != 2 or gt.ndim != 2:
            raise ValueError('crossover scoring requires dosage matrices')

        if pred.shape != gt.shape:
            raise ValueError(
                f'prediction and ground truth shapes do not match: {pred.shape} != {gt.shape}'
            )

        pred_edge = _dosage_edge_signal(pred)
        gt_edge = _dosage_edge_signal(gt)

        chrom_true_mass = gt_edge.sum()
        chrom_pred_mass = pred_edge.sum()
        true_mass += chrom_true_mass
        pred_mass += chrom_pred_mass

        if chrom_true_mass > 0:
            # True-edge windows give predicted edges partial credit when close to truth.
            gt_window = _edge_window(gt_edge, window_size)
            recall_overlap += np.sum(pred_edge * gt_window)

        if chrom_pred_mass > 0:
            # Predicted-edge windows give true edges partial credit when close to prediction.
            pred_window = _edge_window(pred_edge, window_size)
            precision_overlap += np.sum(gt_edge * pred_window)

    recall = np.nan if true_mass <= 0 else recall_overlap / true_mass
    precision = np.nan if pred_mass <= 0 else precision_overlap / pred_mass

    return (
        np.clip(precision, 0.0, 1.0) if not np.isnan(precision) else np.nan,
        np.clip(recall, 0.0, 1.0) if not np.isnan(recall) else np.nan,
    )


def _max_detectable_cos(m, gt):
    if gt.ndim != 2:
        raise ValueError('ground truth must be a dosage matrix')
    if m.shape != gt.shape:
        raise ValueError(
            f'marker and ground truth shapes do not match: {m.shape} != {gt.shape}'
        )

    co_idx = np.where(np.any(np.diff(gt, axis=0), axis=1))[0] + 1
    if co_idx.size == 0:
        return 0

    m_seg = np.array_split(m, co_idx, axis=0)
    seg_gt = gt[np.insert(co_idx, 0, 0)]
    supported_gt = []

    for seg, dosage in zip(m_seg, seg_gt):
        inherited = dosage > 0
        if inherited.any() and seg[:, inherited].sum() > 0:
            supported_gt.append(dosage)

    if len(supported_gt) < 2:
        return 0

    supported_gt = np.asarray(supported_gt)
    return np.any(np.diff(supported_gt, axis=0), axis=1).sum()


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


def _nanmean(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.isnan(values).all():
        return np.nan
    return np.nanmean(values)


def co_sample_assignment_metrics(cb_co_assignments, bin_size):

    precision = []
    recall = []
    fdr = []
    matched_distances = []
    for _, sample in cb_co_assignments.deep_items():

        has_gt = np.isfinite(sample[:, 0])
        has_pred = np.isfinite(sample[:, 1])
        matched = has_gt & has_pred
        n_gt = has_gt.sum()
        n_pred = has_pred.sum()
        n_matched = matched.sum()

        precision.append(np.nan if n_pred == 0 else n_matched / n_pred)
        recall.append(np.nan if n_gt == 0 else n_matched / n_gt)
        fdr.append(np.nan if n_pred == 0 else (n_pred - n_matched) / n_pred)
        if n_matched:
            matched_distances.append(sample[matched, 4])

    if matched_distances:
        mean_distance_bp = np.concatenate(matched_distances).mean() * bin_size
    else:
        mean_distance_bp = np.nan

    return _nanmean(precision), _nanmean(recall), _nanmean(fdr), mean_distance_bp


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
    co_sample_gt_assignment = co_preds.metadata.get('co_sample_gt_assignment')
    columns = [
        'cb', 'gt_n_crossovers', 'gt_detectable_crossovers',
        'gt_haplotype_mae_score', 'gt_crossover_precision', 'gt_crossover_recall',
    ]
    if co_sample_gt_assignment is not None:
        columns += [
            'gt_sample_co_precision', 'gt_sample_co_recall',
            'gt_sample_co_fdr', 'gt_sample_co_mean_distance_bp'
        ]

    score_metrics = []
    for cb, cb_co_preds in co_preds.items():
        if cb.startswith('doublet'):
            score_metrics.append([cb] + [np.nan] * (len(columns) - 1))
            continue

        cb_co_markers = co_markers[cb]
        cb_co_pred_dosage = {
            chrom: co_preds.get_haplotype_dosage(cb, chrom)
            for chrom in co_preds.chrom_sizes
        }
        cb_co_gt_dosage = {
            chrom: ground_truth.get_haplotype_dosage(cb, chrom)
            for chrom in ground_truth.chrom_sizes
        }
        gt_co_precision, gt_co_recall = gt_crossover_precision_recall(
            cb_co_pred_dosage,
            cb_co_gt_dosage,
        )
        row = [
            cb,
            n_crossovers(cb_co_gt_dosage),
            gt_detectable_crossovers(cb_co_markers, cb_co_gt_dosage),
            gt_haplotype_mae_score(
                cb_co_pred_dosage, cb_co_gt_dosage, max_score=max_phred_score
            ),
            gt_co_precision,
            gt_co_recall,
        ]
        if co_sample_gt_assignment is not None:
            row += co_sample_assignment_metrics(
                co_sample_gt_assignment[cb],
                co_preds.bin_size,
            )
        score_metrics.append(row)

    return pd.DataFrame(score_metrics, columns=columns)


def _ground_truth_from_marker_records(co_markers):
    ground_truth = PredictionRecords.new_like(co_markers)
    for cb, chrom_data in co_markers.metadata['ground_truth'].items():
        for chrom, arr in chrom_data.items():
            ground_truth[cb, chrom] = np.asarray(arr)
    return ground_truth


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
        ground_truth = _ground_truth_from_marker_records(co_markers)
        score_metrics = calculate_ground_truth_metrics(
            co_markers,
            co_preds,
            ground_truth,
        )
    else:
        score_metrics = None

    log.info(f'Writing stats to {output_tsv_fn}')
    _write_metric_tsv(output_tsv_fn, qual_metrics, score_metrics, precision=output_precision)
