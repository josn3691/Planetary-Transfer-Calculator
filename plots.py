"""Porkchop plot, travel-time-vs-fuel plot, and 3D transfer plot."""
import math
from datetime import timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from config import AU, MU_SUN
from kepler import orbit_period, propagate


def porkchop(result, best, path):
    """Departure date vs arrival date, filled by total delta-v, with travel-time lines."""
    i_best, j_best = best
    dep = mdates.date2num(result.dep_dates)[:, None] + 0 * result.tofs[None, :]
    arr = dep + result.tofs[None, :]
    dv = np.ma.masked_invalid(result.dv_total)

    fig, ax = plt.subplots(figsize=(10, 7))
    levels = np.linspace(np.nanmin(dv), np.nanmin(dv) + 6, 25)
    cf = ax.contourf(dep, arr, dv, levels=levels, cmap="viridis_r", extend="max")
    cs = ax.contour(dep, arr, dv, levels=levels[::3], colors="k", linewidths=0.5)
    ax.clabel(cs, fmt="%.1f", fontsize=7)
    tof_lines = ax.contour(dep, arr, np.broadcast_to(result.tofs, dep.shape),
                           levels=range(150, 500, 50), colors="white",
                           linewidths=0.6, linestyles="--")
    ax.clabel(tof_lines, fmt="%d d", fontsize=7)
    ax.plot(dep[i_best, j_best], arr[i_best, j_best], "r*", ms=18, mec="w",
            label=f"Optimum: {result.dv_total[i_best, j_best]:.2f} km/s")
    ax.xaxis_date(); ax.yaxis_date()
    fig.autofmt_xdate()
    ax.set_xlabel("Departure date (Earth)")
    ax.set_ylabel("Arrival date (Mars)")
    ax.set_title("Earth-Mars porkchop plot: total delta-v (km/s)")
    fig.colorbar(cf, label="Total delta-v [km/s]")
    ax.legend(loc="upper left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def tof_vs_fuel(result, best, path):
    """Every (departure, TOF) pair: travel time vs propellant fraction."""
    i_best, j_best = best
    tof = np.broadcast_to(result.tofs, result.prop_frac.shape)
    dep_num = np.broadcast_to(mdates.date2num(result.dep_dates)[:, None], tof.shape)
    ok = np.isfinite(result.prop_frac)

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(tof[ok], 100 * result.prop_frac[ok], c=dep_num[ok], s=6, cmap="plasma")
    cb = fig.colorbar(sc, label="Departure date")
    cb.ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    # Lower envelope: cheapest departure for each travel time
    env = np.nanmin(result.prop_frac, axis=0)
    ax.plot(result.tofs, 100 * env, "k-", lw=1.5, label="Best fuel at each travel time")
    ax.plot(result.tofs[j_best], 100 * result.prop_frac[i_best, j_best], "r*", ms=18, mec="w",
            label="Optimum")
    ax.set_xlabel("Travel time [days]")
    ax.set_ylabel("Propellant mass fraction [%]")
    ax.set_title("Travel time vs fuel use (every departure date in the window)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _orbit_points(r, v, n=300):
    T = orbit_period(r, v, MU_SUN)
    return np.array([propagate(r, v, t, MU_SUN)[0] for t in np.linspace(0, T, n)]) / AU


def transfer_3d(earth_state, mars_state, v_transfer, dep, tof_days, path):
    """3D heliocentric view: planet orbits, Sun, and the transfer arc (all in AU)."""
    r_e, v_e = earth_state
    r_m, v_m = mars_state                      # Mars at departure
    arc = np.array([propagate(r_e, v_transfer, t, MU_SUN)[0]
                    for t in np.linspace(0, tof_days * 86400, 200)]) / AU
    earth_orbit = _orbit_points(r_e, v_e)
    mars_orbit = _orbit_points(r_m, v_m)

    # Planet positions at arrival
    earth_arr = propagate(r_e, v_e, tof_days * 86400, MU_SUN)[0] / AU
    mars_arr = propagate(r_m, v_m, tof_days * 86400, MU_SUN)[0] / AU

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    ax.plot(*earth_orbit.T, color="tab:blue", lw=1, label="Earth orbit")
    ax.plot(*mars_orbit.T, color="tab:red", lw=1, label="Mars orbit")
    ax.plot(*arc.T, color="gold", lw=3, label=f"Transfer ({tof_days:.0f} days)")
    ax.scatter(0, 0, 0, s=200, c="orange", label="Sun")
    ax.scatter(*(r_e / AU), s=60, c="tab:blue", marker="o", label=f"Earth at departure ({dep})")
    ax.scatter(*earth_arr, s=40, c="tab:blue", marker="x", label="Earth at arrival")
    ax.scatter(*(r_m / AU), s=40, c="tab:red", marker="x", label="Mars at departure")
    arr_date = dep + timedelta(days=int(round(tof_days)))
    ax.scatter(*mars_arr, s=60, c="tab:red", marker="o", label=f"Mars at arrival ({arr_date})")

    lim = 1.8
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-0.6, 0.6)
    ax.set_box_aspect((1, 1, 0.35))
    ax.set_xlabel("x [AU]"); ax.set_ylabel("y [AU]"); ax.set_zlabel("z [AU]")
    ax.set_title("Optimal Earth-Mars transfer (heliocentric ecliptic frame)")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
