"""
class_thresholds.py — end-of-run orbit classification for the Earth-Moon CR3BP.

Every simulation in `mission.py` ends with one outcome label. This file is the
single place where those labels and the numbers behind them are defined, so
AIAA and ADSC always agree on what "captured", "escaped", etc. mean.

Conventions match `dynamics.py`: nondimensional rotating (synodic) frame,
Earth at (-mu, 0, 0), Moon at (1 - mu, 0, 0), state s = [x, y, z, vx, vy, vz].

Outcome classes (checked in this priority order)
-------------------------------------------------
  3  earth_impact   propagator stopped: r_earth < R_EARTH
  4  moon_impact    propagator stopped: r_moon  < R_MOON
  5  escape         propagator stopped: |r| > R_ESCAPE from the barycenter
  2  lunar_capture  ends inside the Moon's Hill sphere, bound to the Moon
                    (Moon-relative two-body energy < 0), after dwelling there
                    for at least CAPTURE_MIN_DWELL_DAYS
  1  lunar_flyby    entered the Moon's Hill sphere at some point, but was
                    not captured, did not impact, and did not escape
  0  earth_orbit    never reached the Moon's Hill sphere and is still bound
                    to Earth (Earth-relative two-body energy < 0)
  6  unresolved     none of the above by the end of the coast (unbound from
                    Earth but still inside R_ESCAPE). Re-run with a longer
                    coast if too many runs land here.
 -1  failed         the simulation raised an error (row kept for bookkeeping)

Energy regime (independent of the outcome)
------------------------------------------
After the last burn the Jacobi constant C is conserved, so comparing it to the
Lagrange-point values C_L1 > C_L2 > C_L3 > C_L4 = C_L5 says which "gateways"
of the zero-velocity curves are open — i.e. what the spacecraft is even
energetically allowed to do:
  0  closed     C > C_L1        trapped near Earth, cannot reach the Moon
  1  L1_open    C_L2 < C < C_L1 Earth <-> Moon transfer possible, no escape
  2  L2_open    C_L3 < C < C_L2 can leave the system through L2
  3  L3_open    C_L4 < C < C_L3 can also leave through L3
  4  all_open   C < C_L4        no forbidden region in the plane
"""
from math import sqrt

import numpy as np

import dynamics as dyn

MU = dyn.MU

# ---------------------------------------------------------------------------
# Thresholds (nondimensional unless the name says otherwise)
# ---------------------------------------------------------------------------
R_EARTH = dyn.R_EARTH                    # impact radius, Earth surface
R_MOON = dyn.R_MOON                      # impact radius, Moon surface
R_ESCAPE = 4.0                           # LU from barycenter; same as dynamics.propagate / TLIs.R_ESCAPE
R_HILL_MOON = (MU / 3.0) ** (1.0 / 3.0)  # Moon's Hill sphere, ~0.159 LU (~61,300 km)
CAPTURE_MIN_DWELL_DAYS = 1.0             # time inside the Hill sphere before a bound state counts as capture

# ---------------------------------------------------------------------------
# Label tables (ints are what go in the parquet; names are for humans)
# ---------------------------------------------------------------------------
OUTCOMES = {
    -1: "failed",
    0: "earth_orbit",
    1: "lunar_flyby",
    2: "lunar_capture",
    3: "earth_impact",
    4: "moon_impact",
    5: "escape",
    6: "unresolved",
}
OUTCOME_IDS = {name: k for k, name in OUTCOMES.items()}

ENERGY_REGIMES = {0: "closed", 1: "L1_open", 2: "L2_open", 3: "L3_open", 4: "all_open"}


def lagrange_jacobi(mu=MU):
    """Jacobi constant of each Lagrange point (zero velocity): dict L1..L5 -> C."""
    pts = dyn.lagrange_points(mu)
    return {name: float(dyn.jacobi(np.concatenate([p, np.zeros(3)]), mu)) for name, p in pts.items()}


_C_L = lagrange_jacobi(MU)


def two_body_energy(s, body, mu=MU):
    """Specific two-body energy of the spacecraft relative to 'earth' or 'moon'
    (inertial velocity relative to that body, nondim). Negative = bound."""
    x, y, z, vx, vy, vz = s
    if body == "earth":
        bx, gm = -mu, 1.0 - mu
    elif body == "moon":
        bx, gm = 1.0 - mu, mu
    else:
        raise ValueError("body must be 'earth' or 'moon'")
    dx = x - bx
    r = sqrt(dx * dx + y * y + z * z)
    # Both primaries are fixed in the rotating frame, so the inertial velocity
    # relative to a primary is v_rot + w x (r - r_body), w = z_hat.
    vix, viy = vx - y, vy + dx
    return 0.5 * (vix * vix + viy * viy + vz * vz) - gm / r


class Thresholds:
    # The Jacobi constant itself lives in dynamics.jacobi (single source of truth);
    # these methods only interpret it.

    @staticmethod
    def energy_regime(C, mu=MU):
        """Map a Jacobi constant to an ENERGY_REGIMES id (see module docstring)."""
        c = _C_L if mu == MU else lagrange_jacobi(mu)
        if C > c["L1"]:
            return 0
        if C > c["L2"]:
            return 1
        if C > c["L3"]:
            return 2
        if C > c["L4"]:
            return 3
        return 4

    @staticmethod
    def classify(termination, final_state, entered_moon_hill, inside_moon_hill_at_end=False,
                 dwell_days_at_end=0.0, mu=MU):
        """Return (outcome_id, outcome_name) for one finished run.

        termination             None | 'earth_impact' | 'moon_impact' | 'escape'
        final_state             (6,) last state of the coast
        entered_moon_hill       True if the coast ever went inside R_HILL_MOON
        inside_moon_hill_at_end True if the final state is inside R_HILL_MOON
        dwell_days_at_end       days since the last Hill-sphere entry (0 if outside)
        """
        if termination in ("earth_impact", "moon_impact", "escape"):
            k = OUTCOME_IDS[termination]
            return k, OUTCOMES[k]

        if (inside_moon_hill_at_end
                and two_body_energy(final_state, "moon", mu) < 0.0
                and dwell_days_at_end >= CAPTURE_MIN_DWELL_DAYS):
            return 2, OUTCOMES[2]

        if entered_moon_hill:
            return 1, OUTCOMES[1]

        if two_body_energy(final_state, "earth", mu) < 0.0:
            return 0, OUTCOMES[0]

        return 6, OUTCOMES[6]
