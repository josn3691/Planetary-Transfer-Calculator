"""Find the optimal Earth->Mars transfer window in a date range.

Example:
    python main.py --start 2026-09-01 --end 2027-02-28
"""
import argparse
from datetime import date, timedelta
from pathlib import Path

from config import EARTH_ID, MARS_ID, MU_SUN
from horizons import fetch_ephemeris
from lambert import solve_lambert
from plots import porkchop, tof_vs_fuel, transfer_3d
from sweep import find_optimum, run_sweep


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=date.fromisoformat, default=date(2026, 9, 1), help="first departure date")
    p.add_argument("--end", type=date.fromisoformat, default=date(2027, 2, 28), help="last departure date")
    p.add_argument("--tof-min", type=int, default=120, help="min travel time [days]")
    p.add_argument("--tof-max", type=int, default=400, help="max travel time [days]")
    p.add_argument("--dep-step", type=int, default=2, help="departure date step [days]")
    p.add_argument("--tof-step", type=int, default=4, help="travel time step [days]")
    p.add_argument("--out", type=Path, default=Path("output"), help="output directory")
    args = p.parse_args()
    args.out.mkdir(exist_ok=True)

    print("1/4 Fetching JPL Horizons data")
    earth = fetch_ephemeris(EARTH_ID, args.start, args.end)
    # Mars is needed from the first departure (for the plot) through the latest arrival.
    mars = fetch_ephemeris(MARS_ID, args.start, args.end + timedelta(days=args.tof_max))

    print("2/4 Sweeping departure date x travel time (Lambert solver)")
    result = run_sweep(earth, mars, args.start, args.end,
                       args.dep_step, args.tof_min, args.tof_max, args.tof_step)

    i, j = find_optimum(result)
    dep = result.dep_dates[i]
    tof = int(result.tofs[j])
    arr = dep + timedelta(days=tof)
    print("\n=== Optimal transfer (minimum total delta-v) ===")
    print(f"Departure date : {dep}")
    print(f"Arrival date   : {arr}")
    print(f"Travel time    : {tof} days")
    print(f"Departure burn : {result.dv_dep[i, j]:.2f} km/s  (v_inf {result.vinf_dep[i, j]:.2f} km/s)")
    print(f"Capture burn   : {result.dv_arr[i, j]:.2f} km/s  (v_inf {result.vinf_arr[i, j]:.2f} km/s)")
    print(f"Total delta-v  : {result.dv_total[i, j]:.2f} km/s")
    print(f"Propellant     : {100 * result.prop_frac[i, j]:.1f}% of initial mass")

    print("\n3/4 Making porkchop and travel-time/fuel plots")
    porkchop(result, (i, j), args.out / "porkchop.png")
    tof_vs_fuel(result, (i, j), args.out / "tof_vs_fuel.png")

    print("4/4 Making 3D transfer plot")
    r_e, v_e = earth.at(dep)
    r_m, v_m = mars.at(dep)
    v1, _ = solve_lambert(r_e, mars.at(arr)[0], tof * 86400.0, MU_SUN)
    transfer_3d((r_e, v_e), (r_m, v_m), v1, dep, tof, args.out / "transfer_3d.png")
    print(f"\nSaved plots to {args.out}/")


if __name__ == "__main__":
    main()
