
"""
dynamics.py — Earth-Moon Circular Restricted 3-Body Problem (CR3BP)

Conventions (nondimensional, synodic/rotating frame)
-----------------------------------------------------
- Origin at the Earth-Moon barycenter; frame rotates at unit rate about +z.
- Earth at (-mu, 0, 0), Moon at (1 - mu, 0, 0).
- Units: length = Earth-Moon distance, time = 1 / mean motion, mass = m_E + m_M.
- State s = [x, y, z, vx, vy, vz], velocities measured in the rotating frame.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MU = 0.012150585609624                     # m_moon / (m_earth + m_moon)
GM_EARTH_MOON = 398600.4418 + 4902.8001    # km^3/s^2, G*(m_E + m_M)

L_UNIT_KM = 384400.0                                       # km per length unit
T_UNIT_S = np.sqrt(L_UNIT_KM**3 / GM_EARTH_MOON)           # s per time unit (~375,190 s)
T_UNIT_DAYS = T_UNIT_S / 86400.0                           # ~4.3425 days
V_UNIT_KMS = L_UNIT_KM / T_UNIT_S                          # ~1.0245 km/s

R_EARTH = 6378.137 / L_UNIT_KM     # Earth equatorial radius, nondim
R_MOON = 1737.4 / L_UNIT_KM        # Moon mean radius, nondim

_OMEGA = np.array([[0.0, 2.0, 0.0],
                   [-2.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0]])   # Coriolis block


# ---------------------------------------------------------------------------
# Core dynamics
# ---------------------------------------------------------------------------
def distances(x, y, z, mu=MU):
    """Distances from the spacecraft to Earth (r1) and Moon (r2)."""
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2 + z ** 2)
    return r1, r2


def cr3bp_eom(t, s, mu=MU):
    """Equations of motion: ds/dt for solve_ivp."""
    x, y, z, vx, vy, vz = s
    r1, r2 = distances(x, y, z, mu)
    r13, r23 = r1 ** 3, r2 ** 3

    ax = 2 * vy + x - (1 - mu) * (x + mu) / r13 - mu * (x - 1 + mu) / r23
    ay = -2 * vx + y - (1 - mu) * y / r13 - mu * y / r23
    az = -(1 - mu) * z / r13 - mu * z / r23
    return np.array([vx, vy, vz, ax, ay, az])


def jacobi(s, mu=MU):
    """Jacobi constant C = 2U - v^2 (conserved; higher C = less energy).
    Works on a single state (6,) or a trajectory (6, N)."""
    x, y, z, vx, vy, vz = s
    r1, r2 = distances(x, y, z, mu)
    return x ** 2 + y ** 2 + 2 * (1 - mu) / r1 + 2 * mu / r2 - (vx ** 2 + vy ** 2 + vz ** 2)


def jacobian(s, mu=MU):
    """6x6 Jacobian A = d(ds/dt)/ds. Needed for the STM and the EKF baseline."""
    x, y, z = s[:3]
    r1, r2 = distances(x, y, z, mu)
    dx1, dx2 = x + mu, x - 1 + mu
    a, b = (1 - mu) / r1 ** 3, mu / r2 ** 3          # 1/r^3 terms
    c, d = 3 * (1 - mu) / r1 ** 5, 3 * mu / r2 ** 5  # 3/r^5 terms

    Uxx = 1 - a - b + c * dx1 ** 2 + d * dx2 ** 2
    Uyy = 1 - a - b + (c + d) * y ** 2
    Uzz = -a - b + (c + d) * z ** 2
    Uxy = c * dx1 * y + d * dx2 * y
    Uxz = c * dx1 * z + d * dx2 * z
    Uyz = (c + d) * y * z

    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    A[3:, :3] = [[Uxx, Uxy, Uxz], [Uxy, Uyy, Uyz], [Uxz, Uyz, Uzz]]
    A[3:, 3:] = _OMEGA
    return A


def cr3bp_stm_eom(t, Y, mu=MU):
    """State + state transition matrix (42 elements): dPhi/dt = A Phi."""
    s, Phi = Y[:6], Y[6:].reshape(6, 6)
    return np.concatenate([cr3bp_eom(t, s, mu), (jacobian(s, mu) @ Phi).ravel()])


# ---------------------------------------------------------------------------
# Equilibria (validation targets)
# ---------------------------------------------------------------------------
def lagrange_points(mu=MU):
    """Return dict L1..L5 -> (x, y, z). Collinear points solved on the x-axis."""
    def fx(x):
        return x - (1 - mu) * (x + mu) / abs(x + mu) ** 3 - mu * (x - 1 + mu) / abs(x - 1 + mu) ** 3

    eps = 1e-6
    L1 = brentq(fx, -mu + eps, 1 - mu - eps)    # between Earth and Moon
    L2 = brentq(fx, 1 - mu + eps, 2.0)          # beyond the Moon
    L3 = brentq(fx, -2.0, -mu - eps)            # opposite side of Earth
    h = np.sqrt(3) / 2
    return {
        "L1": np.array([L1, 0, 0]),
        "L2": np.array([L2, 0, 0]),
        "L3": np.array([L3, 0, 0]),
        "L4": np.array([0.5 - mu, h, 0]),
        "L5": np.array([0.5 - mu, -h, 0]),
    }


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------
def _make_events(mu, r_escape):
    def earth_impact(t, s, *args):
        return distances(s[0], s[1], s[2], mu)[0] - R_EARTH
    def moon_impact(t, s, *args):
        return distances(s[0], s[1], s[2], mu)[1] - R_MOON
    def escape(t, s, *args):
        return np.linalg.norm(s[:3]) - r_escape
    for ev, direction in ((earth_impact, -1), (moon_impact, -1), (escape, 1)):
        ev.terminal, ev.direction = True, direction
    return [earth_impact, moon_impact, escape]


_EVENT_NAMES = ["earth_impact", "moon_impact", "escape"]


def propagate(s0, tf, mu=MU, n_points=2000, r_escape=4.0, stm=False,
              rtol=1e-12, atol=1e-12, method="DOP853"):
    """Integrate a state from t=0 to tf (nondim time).

    Stops early on Earth impact, Moon impact, or |r| > r_escape (from barycenter;
    4.0 LU ~ Earth's Hill sphere — the labeling team should own this threshold).

    Returns
    -------
    t : (N,) times
    S : (6, N) states  (or (42, N) if stm=True)
    termination : None | 'earth_impact' | 'moon_impact' | 'escape'
    """
    s0 = np.asarray(s0, dtype=float)
    if stm:
        y0, f = np.concatenate([s0, np.eye(6).ravel()]), cr3bp_stm_eom
    else:
        y0, f = s0, cr3bp_eom

    sol = solve_ivp(f, (0.0, tf), y0, args=(mu,), method=method,
                    t_eval=np.linspace(0.0, tf, n_points),
                    events=_make_events(mu, r_escape), rtol=rtol, atol=atol)
    if sol.status == -1:
        raise RuntimeError(f"Integration failed: {sol.message}")

    termination = None
    t, S = sol.t, sol.y
    for name, te, ye in zip(_EVENT_NAMES, sol.t_events, sol.y_events):
        if len(te):
            termination = name
            t, S = np.append(t, te[0]), np.column_stack([S, ye[0]])  # include impact point
            break
    return t, S, termination


# ---------------------------------------------------------------------------
# Frames & units
# ---------------------------------------------------------------------------
def rotating_to_inertial(t, S, mu=MU, center="earth"):
    """Rotating -> inertial frame (axes aligned at t=0). center: 'earth' or 'barycenter'.
    t: (N,), S: (6, N). Returns (6, N)."""
    t = np.atleast_1d(t)
    S = np.atleast_2d(np.asarray(S).T).T
    x, y, z, vx, vy, vz = S[:6]
    if center == "earth":
        x = x + mu
    c, s = np.cos(t), np.sin(t)
    # v_inertial = R(t) (v_rot + w x r), w = z_hat
    vxr, vyr = vx - y, vy + x
    return np.vstack([c * x - s * y, s * x + c * y, z,
                      c * vxr - s * vyr, s * vxr + c * vyr, vz])


def to_dimensional(S):
    """Nondim state(s) -> [km, km, km, km/s, km/s, km/s]."""
    S = np.asarray(S, dtype=float).copy()
    S[:3] *= L_UNIT_KM
    S[3:6] *= V_UNIT_KMS
    return S


def to_nondimensional(S):
    """[km, km/s] state(s) -> nondim."""
    S = np.asarray(S, dtype=float).copy()
    S[:3] /= L_UNIT_KM
    S[3:6] /= V_UNIT_KMS
    return S


def time_to_days(t):
    return np.asarray(t) * T_UNIT_DAYS


def days_to_time(d):
    return np.asarray(d) / T_UNIT_DAYS
