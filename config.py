"""Physical constants and default mission settings (km, s, kg units)."""

# Gravitational parameters [km^3/s^2]
MU_SUN = 1.32712440018e11
MU_EARTH = 398600.4418
MU_MARS = 42828.375

# Body radii [km]
R_EARTH = 6378.137
R_MARS = 3396.19

AU = 149597870.7  # km

# JPL Horizons body IDs and heliocentric center
EARTH_ID = "399"
MARS_ID = "499"
SUN_CENTER = "500@10"

# Circular parking orbit altitudes [km] used for the departure / capture burns
PARKING_ALT_EARTH = 300.0
PARKING_ALT_MARS = 300.0

# Engine assumptions for converting delta-v into propellant use
ISP = 450.0        # specific impulse [s] (hydrogen/oxygen class engine)
G0 = 9.80665e-3    # standard gravity [km/s^2]
