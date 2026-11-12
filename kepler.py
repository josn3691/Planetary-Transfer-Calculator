"""Universal-variable two-body tools (Stumpff functions and Kepler propagation)."""
import math

import numpy as np


def stumpff_c(z):
    if z > 1e-6:
        return (1 - math.cos(math.sqrt(z))) / z
    if z < -1e-6:
        return (math.cosh(math.sqrt(-z)) - 1) / -z
    return 0.5 - z / 24


def stumpff_s(z):
    if z > 1e-6:
        s = math.sqrt(z)
        return (s - math.sin(s)) / s ** 3
    if z < -1e-6:
        s = math.sqrt(-z)
        return (math.sinh(s) - s) / s ** 3
    return 1 / 6 - z / 120


def propagate(r0, v0, dt, mu):
    """Advance a state (r0, v0) by dt seconds along its Kepler orbit."""
    r0n = np.linalg.norm(r0)
    vr0 = np.dot(r0, v0) / r0n
    alpha = 2 / r0n - np.dot(v0, v0) / mu          # 1/semi-major axis
    sq = math.sqrt(mu)

    chi = sq * abs(alpha) * dt                      # initial guess
    for _ in range(50):
        z = alpha * chi ** 2
        C, S = stumpff_c(z), stumpff_s(z)
        F = r0n * vr0 / sq * chi ** 2 * C + (1 - alpha * r0n) * chi ** 3 * S + r0n * chi - sq * dt
        dF = r0n * vr0 / sq * chi * (1 - z * S) + (1 - alpha * r0n) * chi ** 2 * C + r0n
        step = F / dF
        chi -= step
        if abs(step) < 1e-9:
            break

    z = alpha * chi ** 2
    C, S = stumpff_c(z), stumpff_s(z)
    f = 1 - chi ** 2 / r0n * C
    g = dt - chi ** 3 / sq * S
    r = f * r0 + g * v0
    rn = np.linalg.norm(r)
    fdot = sq / (rn * r0n) * (z * S - 1) * chi
    gdot = 1 - chi ** 2 / rn * C
    return r, fdot * r0 + gdot * v0


def orbit_period(r, v, mu):
    a = 1 / (2 / np.linalg.norm(r) - np.dot(v, v) / mu)
    return 2 * math.pi * math.sqrt(a ** 3 / mu)
