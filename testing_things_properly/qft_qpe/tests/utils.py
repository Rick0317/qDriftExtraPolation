def phase_to_bin(phase, precision_bits):
    """Convert a phase (0 ≤ phase < 2π) to binary with given precision bits."""
    from math import pi
    decimal = phase / (2 * pi)
    binary = bin(int(decimal * (2 ** precision_bits)))[2:].zfill(precision_bits)
    return binary
