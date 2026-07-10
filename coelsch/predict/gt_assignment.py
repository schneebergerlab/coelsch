"""
Ground-truth assignment helpers for sampled crossover events.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from coelsch.records import NestedDataArray, PredictionRecords
from .utils import co_switch_resolver

ASSIGNMENT_COLUMNS = (
    'gt_bin',
    'pred_bin',
    'hap1',
    'hap2',
    'sign',
    'distance',
)


def _build_cost_matrix(gt_pos, gt_sign, pred_pos, pred_sign, max_dist):
    dist = np.abs(gt_pos[:, None] - pred_pos[None, :])
    dist = np.where(dist > max_dist, np.inf, dist)

    # Opposite-direction transitions should not be assigned to each other.
    sign_match = gt_sign[:, None] * pred_sign[None, :]
    dist = np.where(sign_match < 0, np.inf, dist)
    return dist


def _assign_positions(gt_pos, gt_sign, pred_pos, pred_sign, max_dist):
    n_gt = len(gt_pos)
    n_pred = len(pred_pos)
    if not n_gt or not n_pred:
        return np.full(n_gt, -1, dtype=int), np.full(n_pred, -1, dtype=int)

    cost = _build_cost_matrix(gt_pos, gt_sign, pred_pos, pred_sign, max_dist)
    gt_assignable = np.isfinite(cost).any(axis=1)
    pred_assignable = np.isfinite(cost).any(axis=0)
    if not gt_assignable.any() or not pred_assignable.any():
        return np.full(n_gt, -1, dtype=int), np.full(n_pred, -1, dtype=int)

    gt_idx = np.arange(n_gt)[gt_assignable]
    pred_idx = np.arange(n_pred)[pred_assignable]
    gt_local, pred_local = linear_sum_assignment(
        cost[np.ix_(gt_assignable, pred_assignable)]
    )

    gt_assignment = np.full(n_gt, -1, dtype=int)
    pred_assignment = np.full(n_pred, -1, dtype=int)
    gt_assignment[gt_idx[gt_local]] = pred_idx[pred_local]
    pred_assignment[pred_idx[pred_local]] = gt_idx[gt_local]
    return gt_assignment, pred_assignment


def _group_events(events):
    grouped = {}
    events = np.asarray(events)
    if events.size == 0:
        return grouped
    for i, event in enumerate(events):
        pos, from_hap, to_hap = event[:3]
        sign = 1 if from_hap < to_hap else -1
        meiosis = frozenset((int(from_hap), int(to_hap)))
        grouped.setdefault(meiosis, []).append(
            (i, int(pos), sign)
        )
    return {
        meiosis: np.asarray(vals, dtype=np.int32)
        for meiosis, vals in grouped.items()
    }


def _assignment_rows(gt_events, pred_events, max_dist):
    gt_grouped = _group_events(gt_events)
    pred_grouped = _group_events(pred_events)
    rows = []

    meioses = sorted(
        set(gt_grouped) | set(pred_grouped),
        key=lambda x: tuple(sorted(x))
    )

    for meiosis in meioses:
        gt = gt_grouped.get(meiosis, np.empty((0, 3), dtype=np.int32))
        pred = pred_grouped.get(meiosis, np.empty((0, 3), dtype=np.int32))
        gt_assignment, _ = _assign_positions(
            gt[:, 1], gt[:, 2], pred[:, 1], pred[:, 2], max_dist=max_dist
        )
        hap1, hap2 = sorted(meiosis)
        assigned_pred = set()
        for gt_local_idx, pred_local_idx in enumerate(gt_assignment):
            gt_idx, gt_bin, gt_sign = gt[gt_local_idx]
            if pred_local_idx >= 0:
                pred_idx, pred_bin, pred_sign = pred[pred_local_idx]
                assigned_pred.add(int(pred_local_idx))
                rows.append((
                    float(gt_bin),
                    float(pred_bin),
                    float(hap1),
                    float(hap2),
                    float(gt_sign),
                    float(abs(pred_bin - gt_bin)),
                ))
            else:
                rows.append((
                    float(gt_bin),
                    np.nan,
                    float(hap1),
                    float(hap2),
                    float(gt_sign),
                    np.nan,
                ))

        for pred_local_idx, (pred_idx, pred_bin, pred_sign) in enumerate(pred):
            if pred_local_idx in assigned_pred:
                continue
            rows.append((
                np.nan,
                float(pred_bin),
                float(hap1),
                float(hap2),
                float(pred_sign),
                np.nan,
            ))

    if not rows:
        return np.empty((0, len(ASSIGNMENT_COLUMNS)), dtype=float)
    return np.asarray(rows, dtype=float)


# TODO: expose max_dist in cli in bases and convert at the same time as o other genomic distances
def assign_co_samples_to_gt(co_samples, ground_truth, max_dist=200, experiment_params=None):
    """
    Assign sampled crossover events to ground-truth crossover events.

    Parameters
    ----------
    co_samples : NestedDataArray
        Sampled crossover event metadata with levels ``('cb', 'chrom', 'sample')``
        and event rows ``[bin_idx, from_hap, to_hap]``.
    ground_truth : PredictionRecords or NestedDataArray
        Ground-truth haplotype calls. PredictionRecords are preferred. Nested
        metadata arrays may be scalar haplotype tracks or dosage matrices.
    max_dist : int, optional
        Maximum assignment distance in bins.
    experiment_params : ExperimentParams, optional
        Experimental design used to infer ground-truth meiosis. Required when
        ``ground_truth`` is a NestedDataArray; inferred from PredictionRecords.

    Returns
    -------
    NestedDataArray
        Assignment rows per ``cb/chrom/sample``. Columns are listed in
        ``ASSIGNMENT_COLUMNS``. Unmatched ground-truth or predicted events are
        retained with missing fields represented as ``np.nan``.
    """
    assignments = NestedDataArray(levels=('cb', 'chrom', 'sample'))
    if experiment_params is None:
        if isinstance(ground_truth, PredictionRecords):
            experiment_params = ground_truth.experiment_params
        else:
            raise ValueError(
                'experiment_params is required when ground_truth is not a PredictionRecords object'
            )

    co_iter = co_switch_resolver(experiment_params)

    for cb, chrom, chrom_co_samples in co_samples.deep_items(max_depth=2):
        try:
            gt = ground_truth[cb, chrom]
        except KeyError:
            continue
        gt_events = list(co_iter(gt))
        for sample_id, pred_events in chrom_co_samples.items():
            assignments[cb, chrom, sample_id] = _assignment_rows(
                gt_events,
                pred_events,
                max_dist=max_dist,
            )

    return assignments
