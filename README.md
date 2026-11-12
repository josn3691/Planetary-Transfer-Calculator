# Earth -> Mars Transfer Window Finder

Finds the minimum-delta-v Earth->Mars transfer in a departure date range, using real
planetary state vectors from NASA JPL Horizons and a Lambert's problem solver.

## Run
```bash
pip install -r requirements.txt
python main.py --start 2026-09-01 --end 2027-02-28
```
Options: `--tof-min/--tof-max` (travel time bounds, days), `--dep-step/--tof-step` (grid
resolution, days), `--out` (output folder). The first run downloads data from Horizons
(internet needed) and caches it in `data/`.

## Outputs (in `output/`)
- `porkchop.png` - departure date vs arrival date, filled by total delta-v, with travel-time lines
- `tof_vs_fuel.png` - travel time vs propellant fraction for every date pair
- `transfer_3d.png` - 3D view of the Sun, Earth/Mars orbits and the transfer arc
- The optimal departure/arrival date, delta-v and propellant fraction are printed to the console

## How it works
| File | Role |
|---|---|
| `config.py` | constants (GM values, radii, parking orbits, Isp) |
| `horizons.py` | downloads/caches heliocentric position + velocity from JPL Horizons |
| `kepler.py` | Stumpff functions and universal-variable Kepler propagation |
| `lambert.py` | Lambert solver (universal variables, bisection) |
| `sweep.py` | for each departure date x travel time: Lambert -> v_inf at both ends -> burns -> fuel |
| `plots.py` | porkchop, travel-time-vs-fuel, and 3D plots |
| `main.py` | command-line entry point tying it together |

## Assumptions
- Departure from a 300 km circular Earth orbit; capture into a 300 km circular Mars orbit
  (edit `config.py`; an elliptical capture orbit would lower delta-v).
- Fuel = Tsiolkovsky propellant fraction at Isp = 450 s.
- Prograde, single-revolution, two-body (Sun-only) transfers; planets are patched-conic point masses.
