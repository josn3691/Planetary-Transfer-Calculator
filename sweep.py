"""Sweep departure date x time-of-flight, solving Lambert's problem for every pair."""

import math
from dataclasses import dataclass
from datetime import timedelta

import numpy as np

from config import (G0, ISP, MU_EARTH, MU_MARS, MU_SUN, PARKING_ALT_EARTH,
                    PARKING_ALT_MARS, R_EARTH, R_MARS)
from lambert import solve_lambert


def burn_dv(v_inf, mu, r_orbit):  # Calculates delta-v needed to leave or enter a circular parking orbit
    """Delta-v to leave (or be captured into) a circular orbit with hyperbolic excess v_inf."""
    return math.sqrt(v_inf ** 2 + 2 * mu / r_orbit) - math.sqrt(mu / r_orbit)  # Difference between transfer and circular orbit speeds


def propellant_fraction(dv):  # Calculates the fraction of spacecraft mass needed for fuel
    """Tsiolkovsky: fraction of initial mass that must be propellant."""
    return 1 - math.exp(-dv / (ISP * G0))  # Uses the rocket equation to calculate propellant fraction


@dataclass
class SweepResult:  # Stores the calculated results for every departure date and flight time
    dep_dates: list          # List of possible Earth departure dates
    tofs: np.ndarray         # Possible times of flight in days
    dv_dep: np.ndarray       # Delta-v required to depart Earth in km/s
    dv_arr: np.ndarray       # Delta-v required to enter Mars orbit in km/s
    dv_total: np.ndarray     # Total delta-v required for the transfer
    vinf_dep: np.ndarray     # Hyperbolic excess speed leaving Earth
    vinf_arr: np.ndarray     # Hyperbolic excess speed arriving at Mars
    prop_frac: np.ndarray    # Fraction of initial mass needed as propellant


def run_sweep(earth, mars, start, end, dep_step, tof_min, tof_max, tof_step):  # Tests combinations of departure dates and travel times
    dep_dates = []  # Stores all departure dates being tested
    d = start  # Begins at the first departure date

    while d <= end:  # Continues until the final departure date
        dep_dates.append(d)  # Adds the current date to the list
        d += timedelta(days=dep_step)  # Moves forward by the departure step

    tofs = np.arange(tof_min, tof_max + 1, tof_step)  # Creates the range of flight times in days

    shape = (len(dep_dates), len(tofs))  # Sets the dimensions of the results arrays
    out = {k: np.full(shape, np.nan) for k in
           ("dv_dep", "dv_arr", "dv_total", "vinf_dep", "vinf_arr", "prop_frac")}  # Initializes arrays with NaN for unsolved transfers

    r_park_e, r_park_m = R_EARTH + PARKING_ALT_EARTH, R_MARS + PARKING_ALT_MARS  # Calculates Earth and Mars parking orbit radii

    for i, dep in enumerate(dep_dates):  # Loops through each possible departure date
        r_e, v_e = earth.at(dep)  # Gets Earth's position and velocity at departure

        for j, tof in enumerate(tofs):  # Tests each possible time of flight
            r_m, v_m = mars.at(dep + timedelta(days=int(tof)))  # Gets Mars position and velocity at arrival

            try:
                v1, v2 = solve_lambert(r_e, r_m, tof * 86400.0, MU_SUN)  # Solves for transfer velocities using Lambert's problem
            except ValueError:
                continue  # Skips combinations where no Lambert solution is found

            vinf_d = np.linalg.norm(v1 - v_e)  # Calculates excess velocity leaving Earth
            vinf_a = np.linalg.norm(v2 - v_m)  # Calculates excess velocity arriving at Mars

            dv_d = burn_dv(vinf_d, MU_EARTH, r_park_e)  # Calculates Earth departure burn
            dv_a = burn_dv(vinf_a, MU_MARS, r_park_m)  # Calculates Mars orbit capture burn

            out["vinf_dep"][i, j], out["vinf_arr"][i, j] = vinf_d, vinf_a  # Stores excess velocities
            out["dv_dep"][i, j], out["dv_arr"][i, j] = dv_d, dv_a  # Stores departure and arrival delta-v
            out["dv_total"][i, j] = dv_d + dv_a  # Calculates total delta-v for this transfer
            out["prop_frac"][i, j] = propellant_fraction(dv_d + dv_a)  # Calculates the required propellant fraction

        if (i + 1) % 10 == 0 or i == len(dep_dates) - 1:  # Displays progress every 10 dates or at completion
            print(f"  swept {i + 1}/{len(dep_dates)} departure dates")

    return SweepResult(dep_dates, tofs, **out)  # Returns all calculated transfer results


def find_optimum(result):  # Finds the departure date and flight time with the lowest total delta-v
    """Index (i, j) of the minimum total delta-v transfer."""
    return np.unravel_index(np.nanargmin(result.dv_total), result.dv_total.shape)  # Returns the row and column of the best transfer
