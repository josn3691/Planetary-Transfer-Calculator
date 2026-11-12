"""Sweep departure date x time-of-flight, solving Lambert's problem for every pair."""
import math
from dataclasses import dataclass
from datetime import timedelta

import numpy as np

from config import (G0, ISP, MU_EARTH, MU_MARS, MU_SUN, PARKING_ALT_EARTH,
                    PARKING_ALT_MARS, R_EARTH, R_MARS)
from lambert import solve_lambert


def burn_dv(v_inf, mu, r_orbit):
    """Delta-v to leave (or be captured into) a circular orbit with hyperbolic excess v_inf."""
    return math.sqrt(v_inf ** 2 + 2 * mu / r_orbit) - math.sqrt(mu / r_orbit)


def propellant_fraction(dv):
    """Tsiolkovsky: fraction of initial mass that must be propellant."""
    return 1 - math.exp(-dv / (ISP * G0))


@dataclass
class SweepResult:
    dep_dates: list          # departure dates (length n)
    tofs: np.ndarray         # times of flight in days (length m)
    dv_dep: np.ndarray       # (n, m) departure burn [km/s], NaN where no solution
    dv_arr: np.ndarray       # (n, m) capture burn [km/s]
    dv_total: np.ndarray     # (n, m)
    vinf_dep: np.ndarray
    vinf_arr: np.ndarray
    prop_frac: np.ndarray    # (n, m) propellant mass fraction


def run_sweep(earth, mars, start, end, dep_step, tof_min, tof_max, tof_step):
    dep_dates = []
    d = start
    while d <= end:
        dep_dates.append(d)
        d += timedelta(days=dep_step)
    tofs = np.arange(tof_min, tof_max + 1, tof_step)

    shape = (len(dep_dates), len(tofs))
    out = {k: np.full(shape, np.nan) for k in
           ("dv_dep", "dv_arr", "dv_total", "vinf_dep", "vinf_arr", "prop_frac")}
    r_park_e, r_park_m = R_EARTH + PARKING_ALT_EARTH, R_MARS + PARKING_ALT_MARS

    for i, dep in enumerate(dep_dates):
        r_e, v_e = earth.at(dep)
        for j, tof in enumerate(tofs):
            r_m, v_m = mars.at(dep + timedelta(days=int(tof)))
            try:
                v1, v2 = solve_lambert(r_e, r_m, tof * 86400.0, MU_SUN)
            except ValueError:
                continue
            vinf_d = np.linalg.norm(v1 - v_e)      # excess speed leaving Earth
            vinf_a = np.linalg.norm(v2 - v_m)      # excess speed arriving at Mars
            dv_d = burn_dv(vinf_d, MU_EARTH, r_park_e)
            dv_a = burn_dv(vinf_a, MU_MARS, r_park_m)
            out["vinf_dep"][i, j], out["vinf_arr"][i, j] = vinf_d, vinf_a
            out["dv_dep"][i, j], out["dv_arr"][i, j] = dv_d, dv_a
            out["dv_total"][i, j] = dv_d + dv_a
            out["prop_frac"][i, j] = propellant_fraction(dv_d + dv_a)
        if (i + 1) % 10 == 0 or i == len(dep_dates) - 1:
            print(f"  swept {i + 1}/{len(dep_dates)} departure dates")
    return SweepResult(dep_dates, tofs, **out)


def find_optimum(result):
    """Index (i, j) of the minimum total delta-v transfer."""
    return np.unravel_index(np.nanargmin(result.dv_total), result.dv_total.shape)
