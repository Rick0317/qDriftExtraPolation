import numpy as np
import statistics

# ════════════════════════════════════════════════════════════════════════════
# post-processing helpers
# ════════════════════════════════════════════════════════════════════════════
def analyse_counts(counts: dict[str,int],
                   t: float,
                   m: int,
                   wrapp_around_correction: bool=True) -> tuple[str, float,float,float,float,float,float]:
    """
    Analyse the counts dictionary from a QPE experiment.
    Args:
        counts: dictionary mapping bitstrings to their observed counts
        t: evolution time used in the QPE experiment
        m: number of qubits used in the QPE experiment
        wrapp_around_correction: whether to apply a correction for phases > 0.5
    Returns:
        A tuple containing:
        - most likely bitstring (str)
        - median energy (float)
        - mean energy (float)
        - standard deviation of energies (float)
        - minimum energy (float)
        - maximum energy (float)
    """
    ml_bitstr  = max(counts, key=counts.get)
    # translate bitstrings ↦ energies
    energies_weighted = []
    for bs, c in counts.items():
        phase = int(bs, 2) / 2**m
        if phase > .5 and wrapp_around_correction:                         # map to (-.5,.5]
            phase -= 1
        energy = 2*np.pi*phase / t
        energies_weighted += [energy]*c
    e_med  = statistics.median(energies_weighted)
    e_mean = statistics.mean  (energies_weighted)
    e_std  = statistics.stdev (energies_weighted) if len(energies_weighted)>1 else 0
    e_min, e_max = min(energies_weighted), max(energies_weighted)
    return ml_bitstr, e_med, e_mean, e_std, e_min, e_max


def int_to_bitstring(value: int, m: int) -> str:
    """
    Convert an integer to a binary bitstring of fixed length `m`.

    Args:
        value (int): The integer to convert. Must be non-negative and less than 2**m.
        m (int): The desired length of the output bitstring.

    Returns:
        str: Binary representation of `value` as a zero-padded bitstring of length `m`.

    Raises:
        ValueError: If `value` is negative or cannot be represented with `m` bits.
    """
    if value < 0:
        raise ValueError("value must be non-negative")
    if value >= 2**m:
        raise ValueError(f"value {value} cannot be represented with {m} bits")

    return format(value, f"0{m}b")
