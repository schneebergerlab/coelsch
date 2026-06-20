"""
Ground-truth assignment helpers for sampled crossover events.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from coelsch.records import NestedDataArray, PredictionRecords


ASSIGNMENT_COLUMNS = (
    'gt_bin',
    'pred_bin',
    'meiosis',
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


def _dosage_from_ground_truth(ground_truth, cb, chrom, ploidy=None):
    if isinstance(ground_truth, PredictionRecords):
        return ground_truth.get_haplotype_dosage(cb, chrom)

    arr = np.asarray(ground_truth[cb, chrom])
    if arr.ndim == 2:
        return arr
    if arr.ndim != 1:
        raise ValueError('ground-truth arrays must be scalar or dosage matrices')
    if ploidy is None:
        ploidy = 1
    return np.stack([ploidy - arr, arr], axis=1)


def _event_meiosis(changed, experiment_params):
    strategy = experiment_params.crossing_strategy
    if strategy == 'f2':
        return -1
    if experiment_params.ploidy == 1 or experiment_params.genotyping_strategy == 'recombinant':
        return 0
    if strategy in {'backcross', 'testcross'}:
        return 1
    if strategy == 'four_way':
        changed = set(changed)
        if changed.issubset({0, 1}):
            return 0
        if changed.issubset({2, 3}):
            return 1
    if strategy == 'three_way':
        changed = set(changed)
        if 1 in changed and 2 not in changed:
            return 0
        if 2 in changed and 1 not in changed:
            return 1
    return 0


def _events_from_dosage(dosage, experiment_params):
    """
    Convert hard/soft dosage changes into event rows [bin_idx, meiosis, sign].

    Meiosis is inferred from the experimental design. F2 events remain
    unresolved and are emitted with meiosis -1.
    """
    delta = np.diff(dosage, axis=0)
    positions = np.nonzero(np.any(delta != 0, axis=1))[0]
    events = []
    for pos in positions:
        d = delta[pos]
        gained = np.flatnonzero(d > 0)
        lost = np.flatnonzero(d < 0)
        changed = np.flatnonzero(d != 0)
        n_events = int(max(d[d > 0].sum(), -d[d < 0].sum()))
        if n_events <= 0:
            continue
        sign = 0
        if gained.size and lost.size:
            sign = int(np.sign(gained[0] - lost[0]))
        elif gained.size:
            sign = 1
        elif lost.size:
            sign = -1
        meiosis = _event_meiosis(changed, experiment_params)
        for _ in range(n_events):
            events.append((pos, meiosis, sign))
    if not events:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray(events, dtype=np.int32)


def _group_events(events):
    grouped = {}
    events = np.asarray(events)
    if events.size == 0:
        return grouped
    events = events.reshape(-1, events.shape[-1])
    for i, event in enumerate(events):
        pos, meiosis, sign = event[:3]
        grouped.setdefault(int(meiosis), []).append((i, int(pos), int(sign)))
    return {
        meiosis: np.asarray(vals, dtype=np.int32)
        for meiosis, vals in grouped.items()
    }


def _assignment_rows(gt_events, pred_events, max_dist):
    gt_grouped = _group_events(gt_events)
    pred_grouped = _group_events(pred_events)
    rows = []

    # F2 ground truth has unresolved meiosis. In that case, assign against all
    # predicted meioses together rather than requiring a meiosis match.
    if set(gt_grouped) == {-1}:
        gt_grouped = {-1: gt_grouped[-1]}
        if pred_grouped:
            pred_grouped = {-1: np.concatenate(list(pred_grouped.values()), axis=0)}
        else:
            pred_grouped = {}

    for meiosis in sorted(set(gt_grouped) & set(pred_grouped)):
        gt = gt_grouped[meiosis]
        pred = pred_grouped[meiosis]
        gt_assignment, _ = _assign_positions(
            gt[:, 1], gt[:, 2], pred[:, 1], pred[:, 2], max_dist=max_dist
        )
        assigned_pred = set()
        for gt_local_idx, pred_local_idx in enumerate(gt_assignment):
            gt_idx, gt_bin, gt_sign = gt[gt_local_idx]
            if pred_local_idx >= 0:
                pred_idx, pred_bin, pred_sign = pred[pred_local_idx]
                assigned_pred.add(int(pred_local_idx))
                rows.append((
                    float(gt_bin),
                    float(pred_bin),
                    float(meiosis),
                    float(gt_sign if gt_sign else pred_sign),
                    float(abs(pred_bin - gt_bin)),
                ))
            else:
                rows.append((
                    float(gt_bin),
                    np.nan,
                    float(meiosis),
                    float(gt_sign),
                    np.nan,
                ))

        for pred_local_idx, (pred_idx, pred_bin, pred_sign) in enumerate(pred):
            if pred_local_idx in assigned_pred:
                continue
            rows.append((
                np.nan,
                float(pred_bin),
                float(meiosis),
                float(pred_sign),
                np.nan,
            ))

    if not rows:
        return np.empty((0, len(ASSIGNMENT_COLUMNS)), dtype=float)
    return np.asarray(rows, dtype=float)


def assign_co_samples_to_gt(co_samples, ground_truth, max_dist=5_000_000 // 25_000,
                            ploidy=None, experiment_params=None):
    """
    Assign sampled crossover events to ground-truth crossover events.

    Parameters
    ----------
    co_samples : NestedDataArray
        Sampled crossover event metadata with levels ``('cb', 'chrom', 'sample')``
        and event rows ``[bin_idx, meiosis, sign]``.
    ground_truth : PredictionRecords or NestedDataArray
        Ground-truth haplotype calls. PredictionRecords are preferred. Nested
        metadata arrays may be scalar haplotype tracks or dosage matrices.
    max_dist : int, optional
        Maximum assignment distance in bins.
    ploidy : int, optional
        Ploidy used to convert scalar NestedData ground truth into dosage.
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

    for cb in co_samples.get_level_keys('cb'):
        for chrom in co_samples.get_level_keys('chrom'):
            try:
                gt_dosage = _dosage_from_ground_truth(ground_truth, cb, chrom, ploidy=ploidy)
            except KeyError:
                continue
            gt_events = _events_from_dosage(
                np.round(gt_dosage),
                experiment_params=experiment_params,
            )
            for sample in co_samples.get_level_keys('sample'):
                pred_events = co_samples.get((cb, chrom, sample))
                if pred_events is None:
                    continue
                assignments[cb, chrom, sample] = _assignment_rows(
                    gt_events,
                    pred_events,
                    max_dist=max_dist,
                )

    return assignments
