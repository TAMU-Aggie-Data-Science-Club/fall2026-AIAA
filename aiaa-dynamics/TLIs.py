
####################################################
#           AIAA Trajectory Simulation             #
#         CODE WRITTEN BY AARYA SHEWALE            #
#   IF NEED HELP, CONTACT "aaryashewale@tamu.edu"  #
#################################################### 


'''

PROGRAM
-------------

Simulates a translunar injection (TLI) sequence based on a seriers of user inputs 


PHYSICS
-------------

Based on the Circular Restricted 3 Body Problem (CR3BP) propagated in the rotating (synodic) frame


INPUTS:
-------------

Parking orbit altitude [km] [Example: 200.0]

Number of intermediate burns before TLI [Example: 2]

Delta-v of each intermediate burn [km/s] [Example: 0.03]

Delta-v of the final TLI burn [km/s] [Example: 3.04]

Starting angle in parking orbit [deg] [Example: 242]
(think top-down 2D Earth view with angle being centered at Earth's core)

MODEL NOTES/SIMPLIFICATIONS
-------------

1.  Motion is modeled in the planar CR3BP (z = 0) by default; `dynamics.py`'s
     equations of motion are fully 3D if you want to extend it.

2.  Burns are impulsive (instantaneous delta-v), applied prograde
    (along the current rotating-frame velocity vector) at each perigee
    (closest approach to Earth) — the classic apogee-raising strategy
     used by multi-burn TLI profiles (e.g. phasing loops).

3. Perigee/apogee period estimates used to size each propagation segment
    come from an instantaneous two-body (Earth-only) vis-viva estimate;
    the *actual* perigee crossing is found from the integrated trajectory
    (local minimum of Earth range), so segment sizing only needs to be
    "long enough," not exact.

3.  mu and the Earth-Moon length/time/velocity units come from
    `dynamics.py` (so this script and `dynamics.py` always agree on units).

4.  The Jacobi constant C = x^2+y^2 + 2(1-mu)/r1 + 2*mu/r2 - v^2 (via
    `dynamics.jacobi`) is tracked throughout: printed before/after every
    burn and at the start/end of the run. It is conserved along any coast
    and jumps only at an impulsive burn, so it also serves as a sanity
    check on the integration.

5.  `dynamics.propagate` also detects Earth impact, Moon impact, and
    system escape (|r| > 4 LU from the barycenter) as integration-terminating
    events; if the post-TLI coast ends in one of these, it's reported.


Run with -h / --help to see all options.

'''


#################
#    IMPORTS    #
#################

import argparse                      # Python's built-in parser
import sys                           # Python's built-in system module 
import numpy as np                   # Math library of commands and functions
from numba import njit               # A tool that directly translates python to machine code for fast computations
import matplotlib                    # Importing matplotlib
matplotlib.use("Agg")                # Converts plot into a seperate PNG
import matplotlib.pyplot as plt      # Importing plotting tools from matplotlib

import dynamics as dyn               # AIAA dynamics file coded by Matt et al. 


##########################
#    System variables    #
##########################

MU = dyn.MU                              # CR3BP mass ratio
LU_KM = dyn.L_UNIT_KM                    # km, Earth-Moon distance (length unit)
TU_S = dyn.T_UNIT_S                      # s, time unit (1/mean motion)
VU_KMS = dyn.V_UNIT_KMS                  # km/s, velocity unit
R_EARTH_KM = dyn.R_EARTH * LU_KM         # km, Earth equatorial radius
R_MOON_KM = dyn.R_MOON * LU_KM           # km, Moon mean radius

MU_EARTH_ND = 1.0 - MU                   # nondimensional Earth GM (in CR3BP units)
GM_EARTH_KM3S2 = MU_EARTH_ND * dyn.GM_EARTH_MOON  # dimensional Earth-only GM [km^3/s^2],
                                                   # derived the same way CR3BP defines mu,
                                                   # so it's consistent with dynamics.py's units


###################
#    FUNCTIONS    #
###################

def earth_range(traj, mu):

    """traj: (N, 6) array -> (N,) distances to Earth.
       Returns spacecraft's non-dimensional position away from Earth """
    
    r1 , _ = dyn.distances(traj[:, 0], traj[:, 1], traj[:, 2], mu)

    return r1

def moon_range(traj, mu):

    """traj: (N, 6) array -> (N,) distances to Moon.
       Returns spacecraft's non-dimensional position away from Moon """

    _ , r2 = dyn.distances(traj[:, 0], traj[:, 1], traj[:, 2], mu)

    return r2

def circular_parking_state(altitude_km, theta0_rad, mu):

    """Returns initial 6 state vector for a circular parking orbit of given altitude around
    Earth, placed at angle theta (starting angle), expressed in the
    CR3BP rotating frame at t = 0. """

    r_km = R_EARTH_KM + altitude_km
    r_nd = r_km / LU_KM
 
    # Position relative to Earth
    rx_rel = r_nd * np.cos(theta0_rad)
    ry_rel = r_nd * np.sin(theta0_rad)

    # Absolute position at Earth (Earth at (Earth at (-mu, 0, 0)))
    x = -mu + rx_rel
    y = ry_rel
    z = 0.0

    # Circular (two-body, Earth-only) inertial speed relative to Earth
    v_circ_km_s = np.sqrt(GM_EARTH_KM3S2 / r_km)
    v_circ_nd = v_circ_km_s / VU_KMS

    # Prograde (meaning moving in the same direction as body) 
    # tangential direction (counterclockwise) relative to Earth
    tx = -np.sin(theta0_rad)
    ty = np.cos(theta0_rad)

    # Rotating-frame velocity compututation 
    omega_cross_rrel_x = -ry_rel
    omega_cross_rrel_y = rx_rel

    # Velocities of the state vector adjusted for rotating frame of motion
    vx = v_circ_nd * tx - omega_cross_rrel_x
    vy = v_circ_nd * ty - omega_cross_rrel_y
    # 0 because orbit is only on x-y plane
    vz = 0.0

    return np.array([x, y, z, vx, vy, vz])

def apply_prograde_burn(state, dv_nd):

    """Add delta-v along the current (rotating-frame) velocity direction 
       that the spacecraft is moving in """

    s = state.copy()
    # Copying velocity components
    v = s[3:6]
    speed = np.linalg.norm(v)
    if speed < 1e-12:
        # Raises error if velocity is zero because no direction would exist
        raise ValueError("Cannot apply a prograde burn: velocity is ~zero.")
    v_hat = v / speed
    s[3:6] = v + dv_nd * v_hat
    return s

def estimate_period_nd(state, mu):

    """Rough two-body (Earth-only) orbital period estimate, nondimensional,
    used only to size the propagation segment between burns."""

    # Assigning variables to be used in calculations
    x, y, z, vx, vy, vz = state
    rx, ry, rz = x + mu, y, z
    r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)

    # Convert rotating-frame velocity back to inertial (relative to Earth)
    vix = vx - ry
    viy = vy + rx
    viz = vz
    v2 = vix ** 2 + viy ** 2 + viz ** 2

    mu_e = MU_EARTH_ND
    eps = 0.5 * v2 - mu_e / r  # specific orbital energy (two-body, Earth-only)
    if eps >= 0:
        # Unbound (hyperbolic/parabolic) two-body estimate -> no periodic
        # orbit; fall back to a generous fixed window.
        return 10.0
    a = -mu_e / (2.0 * eps)
    T = 2.0 * np.pi * np.sqrt(a ** 3 / mu_e)
    # Returns period in non-dimensional time units (1 TU = ~4.94 days)
    return T


#######################################
#    Numba-JIT PROPOGATION KERNALS    #
#   

# (cache = True) saves compiled result to disk to save time
@njit(cache=True)
def eom_nb(s, mu):
    """CR3BP state derivative [vx, vy, vz, ax, ay, az] (compiled)."""
    x, y, z, vx, vy, vz = s
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y ** 2 + z ** 2)
    r13 = r1 ** 3
    r23 = r2 ** 3
    out = np.empty(6)
    out[0] = vx
    out[1] = vy
    out[2] = vz
    out[3] = 2.0 * vy + x - (1.0 - mu) * (x + mu) / r13 - mu * (x - 1.0 + mu) / r23
    out[4] = -2.0 * vx + y - (1.0 - mu) * y / r13 - mu * y / r23
    out[5] = -(1.0 - mu) * z / r13 - mu * z / r23
    return out


@njit(cache=True)
def rk4_step_nb(s, dt, mu):
    """One classical Runge-Kutta 4 step (compiled)."""
    k1 = eom_nb(s, mu)
    k2 = eom_nb(s + 0.5 * dt * k1, mu)
    k3 = eom_nb(s + 0.5 * dt * k2, mu)
    k4 = eom_nb(s + dt * k3, mu)
    return s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@njit(cache=True)
def propagate_nb(s0, dt, nsteps, mu, r_earth, r_moon, r_escape):
    """Fixed-step RK4 propagation with event checks, entirely compiled.

    Returns (traj, n_valid, code): traj has nsteps+1 rows but only the first
    n_valid+1 are meaningful; code = 0 (ran to completion), 1 (Earth
    impact), 2 (Moon impact), 3 (escape). The stopping row is included.
    """
    traj = np.empty((nsteps + 1, 6))
    traj[0] = s0
    s = s0.copy()
    for i in range(nsteps):
        s = rk4_step_nb(s, dt, mu)
        traj[i + 1] = s
        r1 = np.sqrt((s[0] + mu) ** 2 + s[1] ** 2 + s[2] ** 2)
        r2 = np.sqrt((s[0] - 1.0 + mu) ** 2 + s[1] ** 2 + s[2] ** 2)
        rr = np.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2)
        if r1 < r_earth:
            return traj, i + 1, 1
        if r2 < r_moon:
            return traj, i + 1, 2
        if rr > r_escape:
            return traj, i + 1, 3
    return traj, nsteps, 0


@njit(cache=True)
def jacobi_series_nb(traj, mu):
    """Jacobi constant at every row of an (N, 6) trajectory (compiled)."""
    n = traj.shape[0]
    out = np.empty(n)
    for i in range(n):
        x, y, z, vx, vy, vz = traj[i]
        r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
        r2 = np.sqrt((x - 1.0 + mu) ** 2 + y ** 2 + z ** 2)
        out[i] = (x ** 2 + y ** 2 + 2.0 * (1.0 - mu) / r1 + 2.0 * mu / r2
                  - (vx ** 2 + vy ** 2 + vz ** 2))
    return out


_EVENT_NAMES = [None, "earth_impact", "moon_impact", "escape"]
R_ESCAPE = 4.0           # LU from the barycenter, same as dynamics.propagate
MAX_DT_ND = 1.0e-5       # RK4 step ceiling (about 3.75 s); keeps C conserved to ~1e-8
SOLVER = "numba"         # set from --solver in main()


def propagate_segment(state, tf_nd, mu, n_points, rtol=1e-10, atol=1e-10):
    """Propagate `state` forward by tf_nd. Returns (t, traj (N, 6), termination).

    With the numba solver the whole path is computed at a fixed step of at
    most MAX_DT_ND and every step is kept (n_points is ignored); with the
    scipy solver `dynamics.propagate` is used and sampled at n_points.
    """
    if SOLVER == "scipy":
        t, S, termination = dyn.propagate(state, tf_nd, mu=mu, n_points=n_points,
                                           rtol=rtol, atol=atol)
        return t, S.T, termination

    nsteps = max(int(np.ceil(tf_nd / MAX_DT_ND)), 10)
    dt = tf_nd / nsteps
    traj, n_valid, code = propagate_nb(np.asarray(state, dtype=float), dt, nsteps, mu,
                                        dyn.R_EARTH, dyn.R_MOON, R_ESCAPE)
    t = dt * np.arange(n_valid + 1)
    return t, traj[: n_valid + 1], _EVENT_NAMES[code]


def propagate_until_next_perigee(state, mu, window_lo=0.5, window_hi=1.3,
                                  min_period_nd=0.05, points_per_period=1500):
    """Propagate from `state` (assumed at/near perigee) forward by roughly
    one orbit and locate the next perigee as the minimum Earth range within
    a search window of [window_lo, window_hi] x (the two-body period
    estimate). Returns (trajectory_segment, perigee_index, t_segment), where
    t_segment[i] is the nondimensional time elapsed since `state`, for each
    row of trajectory_segment (t_segment[0] == 0).

    Using the global minimum over a time window — rather than scanning for
    the first strict local minimum — sidesteps a subtlety of propagating
    with `dynamics.py`'s high-precision adaptive integrator: a near-circular
    orbit's true (CR3BP-perturbed) Earth-range oscillation can be only
    meters in size, at or below the integrator's own floating-point noise
    floor, which a naive "first local min" scan can mistake for perigee
    after just a handful of samples. Restricting the search to a window
    centered on the expected period (and excluding the immediate vicinity
    of the starting point, which is trivially a local minimum when `state`
    is itself already at perigee) finds the genuine next perigee crossing
    regardless of how small that oscillation is.
    """
    T_est = max(estimate_period_nd(state, mu), min_period_nd)
    total_t = window_hi * T_est
    n_points = max(int(points_per_period * window_hi), 300)

    t, traj, termination = propagate_segment(state, total_t, mu, n_points)
    r = earth_range(traj, mu)

    lo_idx = int(np.searchsorted(t, window_lo * T_est))
    lo_idx = min(max(lo_idx, 1), len(t) - 1)
    idx = lo_idx + int(np.argmin(r[lo_idx:]))

    return traj[: idx + 1], idx, t[: idx + 1]


# ---------------------------------------------------------------------------
# Main simulation driver
# ---------------------------------------------------------------------------
def run_simulation(parking_altitude_km, num_intermediate_burns,
                    intermediate_dv_kms, tli_dv_kms, theta0_deg,
                    coast_days, coast_steps=4000):
    mu = MU

    state0 = circular_parking_state(parking_altitude_km, np.deg2rad(theta0_deg), mu)

    full_traj = [state0.reshape(1, 6)]
    times_nd = [np.array([0.0])]
    cum_t_nd = 0.0

    burn_points = []     # (x, y, label) for plotting, in nondim coords
    burn_log = []
    burn_times_nd = []   # time (nd) at which each burn occurs
    burn_jacobi = []     # (C_before, C_after) for each burn, in that order

    state = state0.copy()

    # The parking orbit starts exactly circular, so it has no genuine
    # perigee to search for yet (the real CR3BP perturbation on a LEO is
    # only meters, well below the adaptive integrator's own noise floor —
    # searching for a "local minimum" there just chases numerical noise).
    # The very first burn is therefore applied directly at state0; once
    # any burn has been applied the orbit is meaningfully eccentric, and
    # perigee search works normally (and reliably) from then on.
    orbit_is_circular = True

    def _record_segment(seg, t_seg):
        """Append a propagated segment (sharing its first point with the
        previous one) to the trajectory/time history, using the actual
        time points dynamics.propagate produced for this segment."""
        nonlocal cum_t_nd
        full_traj.append(seg[1:])
        seg_t = cum_t_nd + t_seg[1:]
        times_nd.append(seg_t)
        cum_t_nd = seg_t[-1] if len(seg_t) > 0 else cum_t_nd

    # --- Intermediate (apogee-raising) burns, each applied at perigee ---
    for i in range(num_intermediate_burns):
        if not orbit_is_circular:
            seg, idx, t_seg = propagate_until_next_perigee(state, mu)
            _record_segment(seg, t_seg)
            state = seg[-1].copy()
        orbit_is_circular = False

        c_before = float(dyn.jacobi(state, mu))
        dv_nd = intermediate_dv_kms / VU_KMS
        burn_points.append((state[0], state[1], f"Burn {i + 1}\n(+{intermediate_dv_kms:.3f} km/s)"))
        state = apply_prograde_burn(state, dv_nd)
        c_after = float(dyn.jacobi(state, mu))

        burn_times_nd.append(cum_t_nd)
        burn_jacobi.append((c_before, c_after))
        burn_log.append(
            f"  Intermediate burn {i + 1}: +{intermediate_dv_kms:.4f} km/s prograde at perigee "
            f"(Earth range = {earth_range(state.reshape(1, 6), mu)[0] * LU_KM:.1f} km)  "
            f"|  Jacobi C: {c_before:.6f} -> {c_after:.6f}  (dC = {c_after - c_before:+.6f})"
        )

    # --- Final TLI burn, also applied at the (next) perigee ---
    if not orbit_is_circular:
        seg, idx, t_seg = propagate_until_next_perigee(state, mu)
        _record_segment(seg, t_seg)
        state = seg[-1].copy()
    orbit_is_circular = False

    c_before = float(dyn.jacobi(state, mu))
    dv_nd = tli_dv_kms / VU_KMS
    burn_points.append((state[0], state[1], f"TLI burn\n(+{tli_dv_kms:.3f} km/s)"))
    state = apply_prograde_burn(state, dv_nd)
    c_after = float(dyn.jacobi(state, mu))

    burn_times_nd.append(cum_t_nd)
    burn_jacobi.append((c_before, c_after))
    burn_log.append(
        f"  TLI burn: +{tli_dv_kms:.4f} km/s prograde at perigee "
        f"(Earth range = {earth_range(state.reshape(1, 6), mu)[0] * LU_KM:.1f} km)  "
        f"|  Jacobi C: {c_before:.6f} -> {c_after:.6f}  (dC = {c_after - c_before:+.6f})"
    )

    # --- Coast phase toward the Moon ---
    coast_t_nd = dyn.days_to_time(coast_days)
    t_coast, coast_traj, coast_termination = propagate_segment(state, coast_t_nd, mu, coast_steps)
    _record_segment(coast_traj, t_coast)

    trajectory = np.vstack(full_traj)
    times_nd_arr = np.concatenate(times_nd)
    final_state = trajectory[-1]

    # Jacobi constant along the whole trajectory (dynamics.jacobi handles a
    # (6, N) trajectory directly)
    jacobi_series = jacobi_series_nb(np.ascontiguousarray(trajectory), mu)

    jacobi_initial = float(dyn.jacobi(state0, mu))
    jacobi_final = float(dyn.jacobi(final_state, mu))

    # Closest lunar approach reached during the coast phase (informational)
    r_moon_coast = moon_range(coast_traj, mu)
    min_idx = int(np.argmin(r_moon_coast))
    closest_approach_km = r_moon_coast[min_idx] * LU_KM - R_MOON_KM
    closest_approach_state = coast_traj[min_idx]

    # Lagrange points (handy reference points for the plot/summary)
    lagrange_pts = dyn.lagrange_points(mu)

    results = {
        "mu": mu,
        "trajectory": trajectory,
        "times_nd": times_nd_arr,
        "times_days": dyn.time_to_days(times_nd_arr),
        "final_state_nd": final_state,
        "final_state_dim": dyn.to_dimensional(final_state.copy()),
        "burn_points": burn_points,
        "burn_log": burn_log,
        "burn_times_days": dyn.time_to_days(np.array(burn_times_nd)),
        "burn_jacobi": burn_jacobi,
        "jacobi_series": jacobi_series,
        "jacobi_initial": jacobi_initial,
        "jacobi_final": jacobi_final,
        "closest_approach_altitude_km": closest_approach_km,
        "closest_approach_state_dim": dyn.to_dimensional(closest_approach_state.copy()),
        "coast_termination": coast_termination,
        "lagrange_points": lagrange_pts,
    }
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_trajectory(results, out_path="translunar_trajectory.png"):
    mu = results["mu"]
    traj = results["trajectory"]

    fig, ax = plt.subplots(figsize=(9, 9))

    # Spacecraft trajectory (nondim coords -> plot directly, label in LU)
    ax.plot(traj[:, 0], traj[:, 1], lw=1.0, color="tab:blue", label="Spacecraft trajectory")

    # Earth and Moon (rotating frame: fixed positions)
    ax.scatter([-mu], [0], s=250, color="tab:green", zorder=5, label="Earth")
    ax.scatter([1 - mu], [0], s=90, color="gray", zorder=5, label="Moon")

    # Collinear Lagrange points (L1, L2), from dynamics.lagrange_points
    lp = results["lagrange_points"]
    for name in ("L1", "L2"):
        px, py, _ = lp[name]
        ax.scatter([px], [py], s=25, marker="x", color="tab:brown", zorder=4)
        ax.annotate(name, (px, py), fontsize=7, color="tab:brown",
                    xytext=(4, -10), textcoords="offset points")

    # Closest lunar approach point
    r_moon = np.sqrt((traj[:, 0] - (1 - mu)) ** 2 + traj[:, 1] ** 2)
    ca_idx = int(np.argmin(r_moon))
    ax.scatter([traj[ca_idx, 0]], [traj[ca_idx, 1]], s=55, marker="*",
               color="tab:orange", zorder=7,
               label=f"Closest approach\n({results['closest_approach_altitude_km']:.0f} km alt.)")

    # Burn points (fan the annotations out so overlapping burns at nearly
    # the same perigee point remain legible)
    # Note: deliberately no `arrowprops` here — on some matplotlib/Python
    # combinations (e.g. older matplotlib on Python 3.14) annotate() with
    # arrowprops triggers a RecursionError inside Path.__deepcopy__. A plain
    # text offset avoids that entirely.
    for i, (bx, by, label) in enumerate(results["burn_points"]):
        ax.scatter([bx], [by], s=45, color="tab:red", zorder=6)
        ax.annotate(label, (bx, by), fontsize=7, xytext=(14, -14 - 30 * i),
                    textcoords="offset points", color="tab:red")

    ax.set_xlabel("x  [LU = 384400 km]")
    ax.set_ylabel("y  [LU = 384400 km]")
    ax.set_title("Translunar Injection Trajectory — Earth-Moon CR3BP (rotating frame)", pad=14)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)

    # Give the burn-label callouts room above the trajectory
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.12 * (ymax - ymin))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


####################################
#    INPUT AND PRINT STATEMENTS    #
####################################

_INPUTS = [

    ("parking_altitude_km", "--parking-altitude-km", float, 300.0,
     "Parking orbit altitude [km] [Example: 200.0]", lambda v: v > 0),

    ("num_intermediate_burns", "--num-intermediate-burns", int, 2,
     "Number of intermediate burns before TLI [Example: 2]", lambda v: v >= 0),

    ("intermediate_dv_kms", "--intermediate-dv-kms", float, 0.03,
     "Delta-v of each intermediate burn [km/s] [Example: 0.03]", lambda v: v >= 0),

    ("tli_dv_kms", "--tli-dv-kms", float, 3.04,
     "Delta-v of the final TLI burn [km/s] [Example: 3.04]", lambda v: v > 0),

    ("theta0_deg", "--theta0-deg", float, 242.0,
     "Starting angle in parking orbit [deg] [Example: 242]\n"
     "(think top-down 2D Earth view with angle being centered at Earth's core)", lambda v: True),

]

# Fixed settings
COAST_DAYS = 6.0
COAST_STEPS = 4000
OUT_FILE = "translunar_trajectory.png"


def prompt_value(text, default, typ, valid):

    """Check and returns only good input and if bad, asks for input again"""
    
    while True:

        raw = input(f"{text} [{default}]: ").strip()

        if raw == "":
            return default
        try:
            val = typ(raw)
        except ValueError:
            print(f"  Please enter a valid {typ.__name__}.")
            continue
        if not valid(val):
            print("  That value is out of range; try again.")
            continue

        return val


def parse_args():
    """Collect inputs. Any value not given as a command-line flag is asked
    for interactively (press Enter to accept the shown default). Pass
    --no-prompt to skip all questions and use defaults/flags only."""
    p = argparse.ArgumentParser(
        description="Simulate a multi-burn translunar injection in the Earth-Moon CR3BP. "
                    "Values not given as flags are prompted for interactively."
    )
    for dest, flag, typ, default, text, _ in _INPUTS:
        p.add_argument(flag, type=typ, default=None, help=f"{text} (default: {default})")
    p.add_argument("--solver", choices=["numba", "scipy"], default="numba",
                   help="Propagator: numba = JIT-compiled fixed-step RK4 (default); "
                        "scipy = dynamics.propagate adaptive DOP853")
    p.add_argument("--no-prompt", action="store_true",
                   help="Do not ask questions; use flags and defaults only")
    args = p.parse_args()
    args.coast_days, args.coast_steps, args.out = COAST_DAYS, COAST_STEPS, OUT_FILE

    interactive = not args.no_prompt and sys.stdin.isatty()
    if interactive:
        print("Enter simulation inputs (press Enter to accept the [default]).\n")
    for dest, flag, typ, default, text, valid in _INPUTS:
        if getattr(args, dest) is None:
            setattr(args, dest, prompt_value(text, default, typ, valid) if interactive else default)
        elif not valid(getattr(args, dest)):
            p.error(f"invalid value for {flag}")
    if interactive:
        print()

    return args


def main():
    global SOLVER
    args = parse_args()
    SOLVER = args.solver

    print("=" * 70)
    print("Earth-Moon CR3BP Translunar Injection Simulation")
    print("(solver: Numba-JIT RK4)" if args.solver == "numba"
          else "(solver: dynamics.py, SciPy DOP853 adaptive integration)")
    print("-" * 70)
    print(f"Final TLI burn       : {args.tli_dv_kms} km/s")
    print(f"Post-TLI coast time  : {args.coast_days} days")
    print("-" * 70)

    results = run_simulation(
        parking_altitude_km=args.parking_altitude_km,
        num_intermediate_burns=args.num_intermediate_burns,
        intermediate_dv_kms=args.intermediate_dv_kms,
        tli_dv_kms=args.tli_dv_kms,
        theta0_deg=args.theta0_deg,
        coast_days=args.coast_days,
        coast_steps=args.coast_steps,
    )

    print("Burn sequence:")
    for line in results["burn_log"]:
        print(line)

    fs_nd = results["final_state_nd"]
    fs_dim = results["final_state_dim"]

    print("-" * 70)
    print("Final state vector (rotating/synodic frame):")
    print(f"  Nondimensional : x={fs_nd[0]: .6f}  y={fs_nd[1]: .6f}  z={fs_nd[2]: .6f}  "
          f"vx={fs_nd[3]: .6f}  vy={fs_nd[4]: .6f}  vz={fs_nd[5]: .6f}")
    print(f"  Dimensional    : r = ({fs_dim[0]: .1f}, {fs_dim[1]: .1f}, {fs_dim[2]: .1f}) km")
    print(f"                   v = ({fs_dim[3]: .4f}, {fs_dim[4]: .4f}, {fs_dim[5]: .4f}) km/s")

    if results["coast_termination"] is not None:
        print("-" * 70)
        print(f"NOTE: coast phase ended early due to event: {results['coast_termination']}")

    print("-" * 70)
    print(f"Closest lunar approach during coast: altitude = "
          f"{results['closest_approach_altitude_km']:.1f} km above the lunar surface")

    print("-" * 70)
    print("Jacobi constant (conserved between burns; jumps only at each impulsive burn):")
    print(f"  Initial (parking orbit) : C = {results['jacobi_initial']:.6f}")
    for i, (bt, (c_before, c_after)) in enumerate(zip(results["burn_times_days"], results["burn_jacobi"])):
        label = "TLI burn" if i == len(results["burn_jacobi"]) - 1 else f"Burn {i + 1}"
        print(f"  {label:<12s} (t={bt:6.2f} d) : C = {c_before:.6f} -> {c_after:.6f}  "
              f"(dC = {c_after - c_before:+.6f})")
    print(f"  Final (end of coast)     : C = {results['jacobi_final']:.6f}")

    print("-" * 70)
    print("Collinear Lagrange points (nondimensional x, rotating frame):")
    for name in ("L1", "L2", "L3"):
        px, _, _ = results["lagrange_points"][name]
        print(f"  {name} : x = {px: .6f}  ({px * LU_KM: .1f} km)")

    out_path = plot_trajectory(results, out_path=args.out)
    print("-" * 70)
    print(f"Trajectory plot saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()