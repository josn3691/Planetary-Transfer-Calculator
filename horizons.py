"""Download real state vectors (position + velocity) from NASA JPL Horizons."""
import json
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from config import SUN_CENTER

API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
CACHE_DIR = Path(__file__).parent / "data"


class Ephemeris:
    """Daily heliocentric state vectors for one body (ecliptic J2000, km and km/s)."""

    def __init__(self, start, pos, vel):
        self.start = start
        self.pos = pos
        self.vel = vel

    @property
    def end(self):
        return self.start + timedelta(days=len(self.pos) - 1)

    def at(self, day):
        """Return (r, v) on a given calendar day."""
        i = (day - self.start).days
        if not 0 <= i < len(self.pos):
            raise ValueError(f"{day} is outside the downloaded range {self.start}..{self.end}")
        return self.pos[i], self.vel[i]


def _download(body, start, stop):
    params = {
        "format": "json",
        "COMMAND": f"'{body}'",
        "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTORS'",
        "CENTER": f"'{SUN_CENTER}'",
        "START_TIME": f"'{start.isoformat()}'",
        "STOP_TIME": f"'{stop.isoformat()}'",
        "STEP_SIZE": "'1 d'",
        "OUT_UNITS": "'KM-S'",
        "REF_PLANE": "'ECLIPTIC'",
        "REF_SYSTEM": "'ICRF'",
        "VEC_TABLE": "'2'",       # state vector only: x y z vx vy vz
        "CSV_FORMAT": "'YES'",
    }
    url = API_URL + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)["result"]


def _parse(text):
    if "$$SOE" not in text:
        raise RuntimeError("Horizons returned no data:\n" + text[:500])
    block = text.split("$$SOE")[1].split("$$EOE")[0]
    rows = [line.split(",") for line in block.strip().splitlines()]
    # CSV columns: JDTDB, calendar date, X, Y, Z, VX, VY, VZ
    data = np.array([[float(v) for v in r[2:8]] for r in rows])
    return data[:, :3], data[:, 3:]


def fetch_ephemeris(body, start, stop):
    """Daily heliocentric states for `body` from start to stop (inclusive). Cached on disk."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"{body}_{start}_{stop}.json"
    if cache.exists():
        text = json.loads(cache.read_text())["result"]
    else:
        print(f"  downloading Horizons data for body {body}: {start} -> {stop}")
        text = _download(body, start, stop)
        cache.write_text(json.dumps({"result": text}))
    pos, vel = _parse(text)
    return Ephemeris(start, pos, vel)
