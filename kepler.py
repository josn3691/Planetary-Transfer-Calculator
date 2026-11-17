"""Universal-variable two-body tools (Stumpff functions and Kepler propagation)."""
import math #provides standard mathematical functions

import numpy as np #Imports NumPy for vector and linear algebra operations


def stumpff_c(z): #Stumpff function is a special mathematical function used in orbital mechanics to make one set of equations work for elliptical, parabolic, and hyperbolic orbits.
    if z > 1e-6: #Elliptic orbit case: z positive 
        return (1 - math.cos(math.sqrt(z))) / z
    if z < -1e-6: #Hyperbolic orbit case: z negative 
        return (math.cosh(math.sqrt(-z)) - 1) / -z
    return 0.5 - z / 24


def stumpff_s(z):
    if z > 1e-6: #Elliptic orbit case
        s = math.sqrt(z)
        return (s - math.sin(s)) / s ** 3
    if z < -1e-6: #Hyperbolic orbit case 
        s = math.sqrt(-z)
        return (math.sinh(s) - s) / s ** 3
    return 1 / 6 - z / 120


def propagate(r0, v0, dt, mu):
    """Advance a state (r0, v0) by dt seconds along its Kepler orbit."""
    r0n = np.linalg.norm(r0) #Initial distance from central body 
    vr0 = np.dot(r0, v0) / r0n #Initial radial velocity
    alpha = 2 / r0n - np.dot(v0, v0) / mu          # 1/semi-major axis
    sq = math.sqrt(mu)

    chi = sq * abs(alpha) * dt                      #Initial guess for universal anomaly χ
    for _ in range(50):
        z = alpha * chi ** 2
        C, S = stumpff_c(z), stumpff_s(z)
        F = r0n * vr0 / sq * chi ** 2 * C + (1 - alpha * r0n) * chi ** 3 * S + r0n * chi - sq * dt
        dF = r0n * vr0 / sq * chi * (1 - z * S) + (1 - alpha * r0n) * chi ** 2 * C + r0n
        step = F / dF
        chi -= step
        if abs(step) < 1e-9:
            break

    z = alpha * chi ** 2 #Final z after convergence 
    C, S = stumpff_c(z), stumpff_s(z) #Final stumpff function values 
    f = 1 - chi ** 2 / r0n * C #Lagrange f coefficient 
    g = dt - chi ** 3 / sq * S #Lagrange g coefficient 
    r = f * r0 + g * v0 #New position vector 
    rn = np.linalg.norm(r) #Magnitude of propagated position 
    fdot = sq / (rn * r0n) * (z * S - 1) * chi #Time derivative of f
    gdot = 1 - chi ** 2 / rn * C #Time derivative of g 
    return r, fdot * r0 + gdot * v0 #Returns propagated position vector r and velocity vector v


def orbit_period(r, v, mu): #Computes orbital period for a bound (ellipitcal orbit).
    a = 1 / (2 / np.linalg.norm(r) - np.dot(v, v) / mu)
    return 2 * math.pi * math.sqrt(a ** 3 / mu) #Kepler's third law: T = 2π√(a³/μ)
