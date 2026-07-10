import logging
import numpy as np
import torch

from coelsch.defaults import DEFAULT_RANDOM_SEED
from .model import RigidHMM
from .independent import IndependentMeiosesHMM
from .estimate import estimate_emissions
from .utils import mask_array_zeros, format_params

log = logging.getLogger('coelsch')
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)
DEFAULT_DEVICE = torch.device('cpu')


def _calculate_co_shrinkage(co_markers, min_shrink=0.15):
    # shrinkage factor for crossover rate based on the sparsity of the input data.
    # helps to prevent the model wandering in low information regions
    informative_bins = np.concatenate([
        (m == 0).all(axis=1) for m in co_markers.deep_values()
    ])
    return np.clip(1 - np.mean(informative_bins), min_shrink, 1.0)


def train_rhmm(co_markers, cm_per_mb=4.5,
               segment_size=1_000_000, terminal_segment_size=50_000, interference_half_life=100_000,
               dist_type='poisson', mask_empty_bins=True, independent_meioses='auto',
               max_training_sequences=5_000, device=DEFAULT_DEVICE, rng=DEFAULT_RNG):
    """
    Constructs a RigidHMM instance and fits it to the crossover marker data.

    Parameters
    ----------
    co_markers : MarkerRecords
        MarkerRecords dataset with haplotype specific read/variant information.
    cm_per_mb : float, optional
        Recombination rate in centimorgans per megabase (default: 4.5).
    segment_size : int, optional
        Size of internal genomic segments, i.e. soft-enforced distance between crossovers
        (default: 1,000,000 bp).
    terminal_segment_size : int, optional
        Size of terminal genomic segments, i.e. minimum distance between crossovers and
        ends of chromosomes (default: 50,000 bp).
    interference_half_life : int, optional
        Distance below segment_size at which recombination rate halves to enforce crossover-interference
    dist_type : str, optional
        Type of distribution to use in rHMM - can be either poisson or nb.
    max_training_sequences : int, optional
        The maximum number of sequences to use for training. Datasets with more sequences are randomly undersampled.
    mask_empty_bins : bool, optional
        Use masked arrays/tensors for bins which are empty in all barcodes (i.e. have no markers/are masked)
    device : torch.device, optional
        Device to initialize the model on (default: cpu).
    rng : np.random.Generator, optional
        Random number generator instance.

    Returns
    -------
    RigidHMM
        Initialized RigidHMM model with emission and transition parameters estimated or supplied.
    """
    bin_size = co_markers.bin_size
    rfactor = segment_size // bin_size
    term_rfactor = terminal_segment_size // bin_size
    trans_prob = cm_per_mb * (bin_size / 1e10) * _calculate_co_shrinkage(co_markers)
    if interference_half_life < 1:
        interference_half_life = interference_half_life * segment_size
    trans_prob_decay_rate = interference_half_life / bin_size / np.log(2)

    if not mask_empty_bins:
        X = list(co_markers.deep_values())
    else:
        X = []
        for chrom in co_markers.chrom_sizes:
            X_chrom = co_markers[:, chrom].stack_values()
            X += [m for m in mask_array_zeros(X_chrom, axis=1)]

    if len(X) > max_training_sequences:
        training_idx = rng.choice(len(X), size=max_training_sequences, replace=False)
        X = [X[i] for i in training_idx]

    params = co_markers.experiment_params
    if independent_meioses == 'auto':
        independent_meioses = (
            params.crossing_strategy in ('three_way', 'four_way')
            and params.genotyping_strategy == 'founder'
        )
    if independent_meioses and not (
        params.crossing_strategy in ('three_way', 'four_way')
        and params.genotyping_strategy == 'founder'
    ):
        raise ValueError(
            'independent_meioses is only supported for three_way and four_way founder designs'
        )

    states = params.haplotype_states
    n_haplotypes = co_markers.n_haplotypes
    fg_params, bg_params = estimate_emissions(
        X, params, rfactor, dist_type=dist_type
    )

    columns = [f'hap{i}' for i in range(1, n_haplotypes + 1)]
    log.debug(
        "Estimated model parameters from data:\n"
        f"Foreground:\n{format_params(fg_params, columns)}\n"
        f"Background:\n{format_params(bg_params, columns)}"
    )

    if independent_meioses:
        log.info(
            'Using two independent rHMMs to model male/female meioses, '
            f'matching crossing_strategy "{params.crossing_strategy}"'
        )

        return IndependentMeiosesHMM(
            crossing_strategy=params.crossing_strategy,
            rfactor=rfactor,
            term_rfactor=term_rfactor,
            trans_prob=trans_prob,
            fg_params=fg_params,
            bg_params=bg_params,
            dist_type=dist_type,
            trans_prob_decay_rate=trans_prob_decay_rate,
            device=device
        )

    return RigidHMM(
        states=states,
        n_haplotypes=n_haplotypes,
        rfactor=rfactor,
        term_rfactor=term_rfactor,
        trans_prob=trans_prob,
        fg_params=fg_params,
        bg_params=bg_params,
        dist_type=dist_type,
        trans_prob_decay_rate=trans_prob_decay_rate,
        device=device
    )
