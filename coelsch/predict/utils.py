import numpy as np


def co_switch_resolver(experiment_params):

    recombining_haplotypes = experiment_params.recombining_haplotypes
    allowed_switches = set()
    for hap1, hap2 in recombining_haplotypes:
        allowed_switches.add((hap1, hap2))
        allowed_switches.add((hap2, hap1))

    def _err(b, before, after):
        return ValueError(
            f"Cannot resolve switch at bin {b}: "
            f"{before.tolist()} -> {after.tolist()}"
        )

    if experiment_params.crossing_strategy == "three_way":

        def _resolve_complex_switch(b, delta, before, after):
            lost = np.where(delta < 0)[0]
            gained = np.where(delta > 0)[0]

            # Handles apparent 1 <-> 2 switches by routing through 0.
            #
            # Example:
            # [1, 0, 1] -> [1, 1, 0]
            # delta = [0, +1, -1]
            #
            # First emit 2 -> 0, giving delta = [-1, +1, 0]
            # Then normal allowed-switch logic emits 0 -> 1.
            if len(lost) == 1 and len(gained) == 1:
                h_from = lost[0]
                h_to = gained[0]

                if {h_from, h_to} == {1, 2}:
                    return [(h_from, 0)]

            raise _err(b, before, after)

    else:

        def _resolve_complex_switch(b, delta, before, after):
            raise _err(b, before, after)

    def _iter_switches(x):
        x = np.asarray(x)
        d = np.diff(x, axis=0)

        for b in np.where(np.any(d != 0, axis=1))[0]:
            delta = d[b].copy()

            if delta.sum() != 0:
                raise ValueError(
                    f"Unbalanced dosage change at bin {b}: "
                    f"{x[b].tolist()} -> {x[b + 1].tolist()}"
                )

            while np.any(delta < 0):
                lost = np.where(delta < 0)[0]
                gained = np.where(delta > 0)[0]

                candidates = [
                    (h_from, h_to)
                    for h_from in lost
                    for h_to in gained
                    if (h_from, h_to) in allowed_switches
                ]

                if not candidates:
                    candidates = _resolve_complex_switch(
                        b,
                        delta,
                        x[b],
                        x[b + 1],
                    )

                candidates = sorted(candidates)
                h_from, h_to = candidates[0]

                yield b, h_from, h_to

                delta[h_from] += 1
                delta[h_to] -= 1

    return _iter_switches