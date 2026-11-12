"""Lambert's problem: find the orbit connecting two position vectors in a given time.

Universal-variable formulation (Bate/Mueller/White, Curtis Alg. 5.2), solved by bisection.
"""
import math

import numpy as np

from kepler import stumpff_c, stumpff_s


def solve_lambert(r1, r2, tof, mu):
    """Prograde (short-way) transfer. Returns (v1, v2) in km/s; raises ValueError on failure."""
    r1n, r2n = np.linalg.norm(r1), np.linalg.norm(r2)
    cos_dth = np.dot(r1, r2) / (r1n * r2n)
    dth = math.acos(max(-1.0, min(1.0, cos_dth)))
    if np.cross(r1, r2)[2] < 0:           # prograde = counter-clockwise seen from ecliptic north
        dth = 2 * math.pi - dth
    A = math.sin(dth) * math.sqrt(r1n * r2n / (1 - math.cos(dth)))
    sq = math.sqrt(mu)

    def y_of(z):
        return r1n + r2n + A * (z * stumpff_s(z) - 1) / math.sqrt(stumpff_c(z))

    def tof_error(z):
        y = y_of(z)
        if y < 0:
            return -1.0                    # z too small: push the bracket upward
        return ((y / stumpff_c(z)) ** 1.5 * stumpff_s(z) + A * math.sqrt(y)) / sq - tof

    lo, hi = -4 * math.pi ** 2, 4 * math.pi ** 2 - 1e-3
    if tof_error(hi) < 0:
        raise ValueError("no Lambert solution in bracket")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if tof_error(mid) < 0:
            lo = mid
        else:
            hi = mid
    z = 0.5 * (lo + hi)
    y = y_of(z)

    f = 1 - y / r1n
    g = A * math.sqrt(y / mu)
    gdot = 1 - y / r2n
    v1 = (r2 - f * r1) / g
    v2 = (gdot * r2 - r1) / g
    return v1, v2
