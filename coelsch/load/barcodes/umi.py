from copy import deepcopy
from collections import defaultdict
import itertools as it
from .utils import edit_dist


def umi_dedup_directional(
    umi_hap_counts: dict[str, dict[str, int] | int],
    has_haplotype: bool = True
) -> dict:
    '''
    Deduplicate UMIs using the directional method (UMItools)
    for a collection of UMIs aligning to the same gene/genomic bin,
    maintaining information about which haplotype each read supports
    
    Parameters
    ----------
    umi_hap_counts : dict
        Dictionary of UMI -> haplotype -> count mappings if `has_haplotype` is True.
        Otherwise, a dictionary of UMI -> count.
    has_haplotype : bool, default: True
        Indicates whether haplotype information is present in the input.

    Returns
    -------
    dict
        Dictionary of deduplicated UMI counts. If haplotypes were used, each UMI maps
        to a Counter of haplotype counts.
    '''
    edges = defaultdict(set)
    if has_haplotype:
        umi_counts = {umi: hap_counts.total() for umi, hap_counts in umi_hap_counts.items()}
    else:
        umi_counts = umi_hap_counts
    nodes = sorted(umi_counts, key=umi_counts.__getitem__, reverse=True)
    for umi_i, umi_j in it.combinations(nodes, r=2):
        if edit_dist(umi_i, umi_j) <= 1:
            umi_i_count = umi_counts[umi_i]
            umi_j_count = umi_counts[umi_j]
            if umi_i_count >= (2 * umi_j_count - 1):
                edges[umi_i].add(umi_j)

    deduped = {}
    visited = set()

    for parent in nodes:
        if parent in visited:
            continue

        stack = [parent]
        merged = deepcopy(umi_hap_counts[parent])

        while stack:
            umi = stack.pop()
            if umi in visited:
                continue
            visited.add(umi)

            if umi != parent:
                merged += umi_hap_counts[umi]

            stack.extend(edges[umi])

        deduped[parent] = merged

    return deduped