"""
mission.py — batch CR3BP mission simulator -> parquet dataset for ADSC.

Runs thousands of randomized translunar-injection (TLI) missions in the
Earth-Moon CR3BP and writes one row per run (plan, engine errors, post-TLI
state, energy regime, outcome label, Lagrange-point approaches, summary stats)
to a parquet file. Optionally also writes a second parquet with the
down-sampled post-TLI trajectory of every run.

Pipeline per run
----------------
1. Sample a mission plan (parking altitude, number/size of intermediate burns,
   TLI delta-v, starting angle) from the ranges in SAMPLING_DEFAULTS, or fly
   one fixed plan with --nominal.
2. Dispersion mode (optional): give every burn a random engine error drawn
   from a bell curve (Gaussian): magnitude (m/s), in-plane pointing (deg) and
   timing (s), with sizes set by the --sigma-* flags. --repeats flies each
   plan several times with fresh errors.
3. Build the burn sequence with the exact mission design from TLIs.py
   (circular parking orbit -> prograde burns at perigee -> TLI burn).
4. Coast for --coast-days with a Numba-JIT fixed-step RK4 kernel that tracks
   impacts/escape, lunar approach, Hill-sphere entries and the closest
   approach to each Lagrange point (L1..L5) on the fly.
5. Label the end behavior with class_thresholds.Thresholds.classify.

Runs are spread across CPU cores (ProcessPoolExecutor) and streamed to disk
in row groups, so an interrupted batch still leaves a readable file.

Usage (from aiaa-dynamics/)
---------------------------
    python mission.py --runs 5000
    python mission.py --runs 2000 --coast-days 45 --save-trajectories
    python mission.py --runs 10 --workers 1          # quick smoke test

    # one fixed plan flown 2000 times with 1 m/s burn errors:
    python mission.py --runs 2000 --repeats 2000 --nominal 300 2 0.03 3.047 242 --sigma-dv-ms 1
    # 5000 random plans x 20 repeats, error size drawn per plan between 0 and 3 m/s:
    python mission.py --runs 100000 --repeats 20 --sigma-dv-ms 0 3

Output goes to ../data/processed/ by default (git-ignored — never commit it).
Read it with:  pandas.read_parquet("data/processed/<file>.parquet")
"""
import os

# Parallelism comes from worker processes, so every process gets ONE math-library
# thread. Without this, each worker starts a BLAS thread pool sized to the whole
# CPU (32 threads x 31 workers) and Windows runs out of commit memory
# ("The paging file is too small" / "OpenBLAS: Memory allocation failed").
# Must run before numpy/scipy are imported; workers inherit the environment.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime
from pathlib import Path

import numpy as np
from numba import njit
import pyarrow as pa
import pyarrow.parquet as pq

import dynamics as dyn
import TLIs as tli
from TLIs import rk4_step_nb
from class_thresholds import (Thresholds, OUTCOMES, ENERGY_REGIMES, R_HILL_MOON, R_ESCAPE,
                              R_NEAR_LAGRANGE_KM)

MU = dyn.MU
LU_KM = dyn.L_UNIT_KM
VU_KMS = dyn.V_UNIT_KMS
MAX_DT_ND = tli.MAX_DT_ND        # same RK4 step ceiling as TLIs.py (~3.75 s)
SCHEMA_VERSION = 2               # 2: plan_id/repeat, engine errors, Lagrange-point columns

# Uniform sampling ranges [lo, hi] for each mission input. num_intermediate_burns
# is an integer range (inclusive). Override any of them from the command line.
# TLI range chosen from an 800-run pilot (30-day coast): below ~2.8 km/s almost
# everything stays in Earth orbit, above ~3.15 km/s almost everything escapes.
SAMPLING_DEFAULTS = {
    "parking_altitude_km": (185.0, 1000.0),
    "num_intermediate_burns": (0, 3),
    "intermediate_dv_kms": (0.0, 0.10),
    "tli_dv_kms": (2.80, 3.20),
    "theta0_deg": (0.0, 360.0),
}
PLAN_KEYS = tuple(SAMPLING_DEFAULTS)

# Engine error sizes (1 standard deviation), applied independently to every burn.
SIGMA_KEYS = ("sigma_dv_ms", "sigma_pointing_deg", "sigma_timing_s")

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Each worker loads numpy/scipy/numba/matplotlib (~200-300 MB), so cap the default
# instead of using every core. Raise it with --workers if the machine has RAM to spare.
DEFAULT_MAX_WORKERS = 8
STATE_NAMES = ("x", "y", "z", "vx", "vy", "vz")

# Lagrange points tracked during the coast (nondimensional rotating-frame positions).
LAGRANGE_NAMES = ("l1", "l2", "l3", "l4", "l5")
_LP_ND = np.ascontiguousarray(
    [dyn.lagrange_points(MU)[n.upper()] for n in LAGRANGE_NAMES], dtype=np.float64)


# ---------------------------------------------------------------------------
# Numba-JIT coast kernel
# ---------------------------------------------------------------------------
@njit(cache=True)
def coast_nb(s0, dt, nsteps, mu, r_earth, r_moon, r_escape, r_hill, sample_every, lp, r_near2):
    """Fixed-step RK4 coast with in-loop event detection and statistics.

    lp is an (n, 3) array of Lagrange-point positions; r_near2 is the squared
    "near" radius (nondimensional).

    Returns
    -------
    samples       (k, 7) rows of [t, x, y, z, vx, vy, vz]: t=0, every
                  `sample_every` steps, and the final step
    n_done        steps actually taken
    code          0 completed, 1 Earth impact, 2 Moon impact, 3 escape
    min_r2, i_min_r2   closest Moon distance (center) and its step
    max_r1        farthest Earth distance
    n_entries     number of times the trajectory entered the Moon Hill sphere
    i_first_entry step of first entry (-1 if never)
    inside        inside the Hill sphere at the final step
    i_last_entry  step of the most recent entry (-1 if never)
    min_dl2       (n,) closest squared distance to each Lagrange point
    i_min_l       (n,) step of that closest approach
    i_first_near  (n,) first step within the "near" radius (-1 if never)
    """
    max_samples = nsteps // sample_every + 3
    samples = np.empty((max_samples, 7))
    samples[0, 0] = 0.0
    samples[0, 1:] = s0
    k = 1

    s = s0.copy()
    r1 = np.sqrt((s[0] + mu) ** 2 + s[1] ** 2 + s[2] ** 2)
    r2 = np.sqrt((s[0] - 1.0 + mu) ** 2 + s[1] ** 2 + s[2] ** 2)
    min_r2, i_min_r2, max_r1 = r2, 0, r1
    inside = r2 < r_hill
    n_entries = 1 if inside else 0
    i_first_entry = 0 if inside else -1
    i_last_entry = i_first_entry

    nl = lp.shape[0]
    min_dl2 = np.empty(nl)
    i_min_l = np.zeros(nl, dtype=np.int64)
    i_first_near = np.empty(nl, dtype=np.int64)
    for j in range(nl):
        d2 = (s[0] - lp[j, 0]) ** 2 + (s[1] - lp[j, 1]) ** 2 + (s[2] - lp[j, 2]) ** 2
        min_dl2[j] = d2
        i_first_near[j] = 0 if d2 < r_near2 else -1

    code = 0
    n_done = nsteps
    for i in range(1, nsteps + 1):
        s = rk4_step_nb(s, dt, mu)
        r1 = np.sqrt((s[0] + mu) ** 2 + s[1] ** 2 + s[2] ** 2)
        r2 = np.sqrt((s[0] - 1.0 + mu) ** 2 + s[1] ** 2 + s[2] ** 2)
        rr = np.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2)

        if r2 < min_r2:
            min_r2, i_min_r2 = r2, i
        if r1 > max_r1:
            max_r1 = r1
        if r2 < r_hill:
            if not inside:
                inside = True
                n_entries += 1
                i_last_entry = i
                if i_first_entry < 0:
                    i_first_entry = i
        else:
            inside = False

        for j in range(nl):
            d2 = (s[0] - lp[j, 0]) ** 2 + (s[1] - lp[j, 1]) ** 2 + (s[2] - lp[j, 2]) ** 2
            if d2 < min_dl2[j]:
                min_dl2[j] = d2
                i_min_l[j] = i
            if i_first_near[j] < 0 and d2 < r_near2:
                i_first_near[j] = i

        if r1 < r_earth:
            code = 1
        elif r2 < r_moon:
            code = 2
        elif rr > r_escape:
            code = 3

        if code != 0 or i % sample_every == 0 or i == nsteps:
            samples[k, 0] = i * dt
            samples[k, 1:] = s
            k += 1
        if code != 0:
            n_done = i
            break

    return (samples[:k], n_done, code, min_r2, i_min_r2, max_r1,
            n_entries, i_first_entry, inside, i_last_entry,
            min_dl2, i_min_l, i_first_near)


_TERMINATIONS = [None, "earth_impact", "moon_impact", "escape"]


# ---------------------------------------------------------------------------
# One mission
# ---------------------------------------------------------------------------
def _apply_burn(state, dv_nd, point_err_rad=0.0):
    """Burn along the rotating-frame velocity (prograde, as in TLIs.py), with the
    burn direction rotated in-plane by point_err_rad (pointing error). Identical
    to TLIs.apply_prograde_burn when the pointing error is zero."""
    if point_err_rad == 0.0:
        return tli.apply_prograde_burn(state, dv_nd)
    s = state.copy()
    v = s[3:6]
    speed = np.linalg.norm(v)
    if speed < 1e-12:
        raise ValueError("Cannot apply a burn: velocity is ~zero.")
    ux, uy, uz = v / speed
    c, sn = np.cos(point_err_rad), np.sin(point_err_rad)
    s[3:6] = v + dv_nd * np.array([c * ux - sn * uy, sn * ux + c * uy, uz])
    return s


def _shift_time(state, t, dt_nd, mu=MU):
    """Coast for dt_nd (may be negative) before a burn: models a burn timing
    error (positive = the burn fires late)."""
    if dt_nd == 0.0:
        return state, t
    nsteps = max(int(np.ceil(abs(dt_nd) / MAX_DT_ND)), 1)
    traj, n_valid, _ = tli.propagate_nb(np.ascontiguousarray(state, dtype=np.float64),
                                        dt_nd / nsteps, nsteps, mu,
                                        dyn.R_EARTH, dyn.R_MOON, R_ESCAPE)
    return traj[n_valid].copy(), t + dt_nd


def burn_sequence(parking_altitude_km, num_intermediate_burns, intermediate_dv_kms,
                  tli_dv_kms, theta0_deg, mu=MU, dv_err_kms=None, point_err_rad=None,
                  time_err_nd=None):
    """Parking orbit -> intermediate burns -> TLI, same logic as TLIs.run_simulation.

    The first burn fires directly at the (circular) starting state; every later
    burn, including TLI, fires at the next perigee. The optional error arrays
    (length num_intermediate_burns + 1, TLI last) add a magnitude error [km/s],
    an in-plane pointing error [rad] and a timing error [nondimensional time]
    to each burn. Returns (state_after_tli, t_tli_nd, jacobi_parking).
    """
    n = num_intermediate_burns + 1
    dv_err = np.zeros(n) if dv_err_kms is None else dv_err_kms
    pt_err = np.zeros(n) if point_err_rad is None else point_err_rad
    tm_err = np.zeros(n) if time_err_nd is None else time_err_nd

    state = tli.circular_parking_state(parking_altitude_km, np.deg2rad(theta0_deg), mu)
    c_parking = float(dyn.jacobi(state, mu))
    t = 0.0
    circular = True
    for i in range(num_intermediate_burns):
        if not circular:
            seg, _, t_seg = tli.propagate_until_next_perigee(state, mu)
            state, t = seg[-1].copy(), t + t_seg[-1]
        circular = False
        state, t = _shift_time(state, t, tm_err[i], mu)
        state = _apply_burn(state, (intermediate_dv_kms + dv_err[i]) / VU_KMS, pt_err[i])
    if not circular:
        seg, _, t_seg = tli.propagate_until_next_perigee(state, mu)
        state, t = seg[-1].copy(), t + t_seg[-1]
    state, t = _shift_time(state, t, tm_err[-1], mu)
    state = _apply_burn(state, (tli_dv_kms + dv_err[-1]) / VU_KMS, pt_err[-1])
    return state, t, c_parking


def _total_dv(p):
    return p["num_intermediate_burns"] * p["intermediate_dv_kms"] + p["tli_dv_kms"]


def simulate_one(run_id, p, coast_days, traj_points=0, mu=MU, seed=0,
                 lagrange_radius_km=R_NEAR_LAGRANGE_KM):
    """Simulate one mission. Returns (row dict, trajectory samples or None).

    `p` holds the plan (PLAN_KEYS) and optionally plan_id, repeat and the
    engine error sizes (SIGMA_KEYS). Errors are drawn from a generator seeded
    by (seed, run_id), so a run is reproducible no matter which worker runs it.
    """
    plan = {k: p[k] for k in PLAN_KEYS}
    sig = {k: float(p.get(k, 0.0)) for k in SIGMA_KEYS}
    nb = int(plan["num_intermediate_burns"])
    row = {"run_id": int(run_id), "plan_id": int(p.get("plan_id", run_id)),
           "repeat": int(p.get("repeat", 0)), **plan, "total_dv_kms": _total_dv(plan),
           "coast_days": float(coast_days), **sig}

    dispersed = any(v > 0.0 for v in sig.values())
    if dispersed:
        rng = np.random.default_rng([seed % 2**63, run_id % 2**63])
        dv_err_ms = rng.normal(0.0, sig["sigma_dv_ms"], nb + 1)
        point_err_deg = rng.normal(0.0, sig["sigma_pointing_deg"], nb + 1)
        time_err_s = rng.normal(0.0, sig["sigma_timing_s"], nb + 1)
    else:
        dv_err_ms = point_err_deg = time_err_s = np.zeros(nb + 1)

    s_tli, t_tli, c_parking = burn_sequence(mu=mu, **plan, dv_err_kms=dv_err_ms / 1000.0,
                                            point_err_rad=np.deg2rad(point_err_deg),
                                            time_err_nd=time_err_s / dyn.T_UNIT_S)
    c_tli = float(dyn.jacobi(s_tli, mu))
    c_tli_plan = float(dyn.jacobi(burn_sequence(mu=mu, **plan)[0], mu)) if dispersed else c_tli
    regime = Thresholds.energy_regime(c_tli, mu)

    coast_tu = float(dyn.days_to_time(coast_days))
    nsteps = max(int(np.ceil(coast_tu / MAX_DT_ND)), 10)
    dt = coast_tu / nsteps
    sample_every = max(nsteps // traj_points, 1) if traj_points else nsteps
    r_near_nd = lagrange_radius_km / LU_KM

    (samples, n_done, code, min_r2, i_min_r2, max_r1, n_entries,
     i_first_entry, inside, i_last_entry, min_dl2, i_min_l, i_first_near) = coast_nb(
        np.ascontiguousarray(s_tli, dtype=np.float64), dt, nsteps, mu,
        dyn.R_EARTH, dyn.R_MOON, R_ESCAPE, R_HILL_MOON, sample_every,
        _LP_ND, r_near_nd * r_near_nd)

    s_end = samples[-1, 1:]
    termination = _TERMINATIONS[code]
    dwell_days = float(dyn.time_to_days((n_done - i_last_entry) * dt)) if inside else 0.0
    outcome_id, outcome = Thresholds.classify(termination, s_end, n_entries > 0,
                                              bool(inside), dwell_days, mu)

    row.update({
        "jacobi_tli_plan": c_tli_plan,
        "tli_dv_err_ms": float(dv_err_ms[-1]),
        "tli_point_err_deg": float(point_err_deg[-1]),
        "tli_time_err_s": float(time_err_s[-1]),
        "int_dv_err_ms": [float(v) for v in dv_err_ms[:-1]],
        "int_point_err_deg": [float(v) for v in point_err_deg[:-1]],
        "int_time_err_s": [float(v) for v in time_err_s[:-1]],
        "t_tli_days": float(dyn.time_to_days(t_tli)),
        "jacobi_parking": c_parking,
        "jacobi_tli": c_tli,
        "energy_regime_id": regime,
        "energy_regime": ENERGY_REGIMES[regime],
        **{f"tli_{n}": float(v) for n, v in zip(STATE_NAMES, s_tli)},
        **{f"end_{n}": float(v) for n, v in zip(STATE_NAMES, s_end)},
        "t_end_days": float(dyn.time_to_days(n_done * dt)),
        "termination": termination or "none",
        "min_moon_alt_km": float(min_r2 * LU_KM - dyn.R_MOON * LU_KM),
        "t_min_moon_days": float(dyn.time_to_days(i_min_r2 * dt)),
        "max_earth_dist_km": float(max_r1 * LU_KM),
        "moon_hill_entries": int(n_entries),
        "t_first_moon_hill_days": (float(dyn.time_to_days(i_first_entry * dt))
                                   if i_first_entry >= 0 else float("nan")),
        "moon_hill_dwell_days_at_end": dwell_days,
        "jacobi_drift": float(dyn.jacobi(s_end, mu)) - c_tli,
        "outcome_id": outcome_id,
        "outcome": outcome,
    })
    for j, name in enumerate(LAGRANGE_NAMES):
        row[f"min_dist_{name}_km"] = float(np.sqrt(min_dl2[j]) * LU_KM)
        row[f"t_min_dist_{name}_days"] = float(dyn.time_to_days(i_min_l[j] * dt))
        row[f"near_{name}"] = bool(i_first_near[j] >= 0)
        row[f"t_first_near_{name}_days"] = (float(dyn.time_to_days(i_first_near[j] * dt))
                                            if i_first_near[j] >= 0 else float("nan"))
    traj = samples if traj_points else None
    return row, traj


def _failed_row(run_id, p, coast_days, err):
    plan = {k: p[k] for k in PLAN_KEYS}
    return {"run_id": int(run_id), "plan_id": int(p.get("plan_id", run_id)),
            "repeat": int(p.get("repeat", 0)), **plan, "total_dv_kms": _total_dv(plan),
            "coast_days": float(coast_days), **{k: float(p.get(k, 0.0)) for k in SIGMA_KEYS},
            "outcome_id": -1, "outcome": OUTCOMES[-1], "termination": "none",
            "energy_regime_id": -1, "energy_regime": "unknown", "error": repr(err)}


def simulate_chunk(jobs, coast_days, traj_points, seed=0, lagrange_radius_km=R_NEAR_LAGRANGE_KM):
    """Worker entry point: jobs is a list of (run_id, params)."""
    rows, trajs = [], []
    for run_id, p in jobs:
        try:
            row, traj = simulate_one(run_id, p, coast_days, traj_points, seed=seed,
                                     lagrange_radius_km=lagrange_radius_km)
        except Exception as err:  # keep the batch alive; record the failure
            row, traj = _failed_row(run_id, p, coast_days, err), None
        rows.append(row)
        if traj is not None:
            trajs.append((run_id, traj))
    return rows, trajs


# ---------------------------------------------------------------------------
# Parquet schema
# ---------------------------------------------------------------------------
_F, _I, _S, _B = pa.float64(), pa.int64(), pa.string(), pa.bool_()
_LF = pa.list_(pa.float64())
RUNS_SCHEMA = pa.schema(
    [("run_id", _I), ("plan_id", _I), ("repeat", _I),
     ("parking_altitude_km", _F), ("num_intermediate_burns", _I), ("intermediate_dv_kms", _F),
     ("tli_dv_kms", _F), ("theta0_deg", _F), ("total_dv_kms", _F), ("coast_days", _F),
     ("sigma_dv_ms", _F), ("sigma_pointing_deg", _F), ("sigma_timing_s", _F),
     ("jacobi_tli_plan", _F),
     ("tli_dv_err_ms", _F), ("tli_point_err_deg", _F), ("tli_time_err_s", _F),
     ("int_dv_err_ms", _LF), ("int_point_err_deg", _LF), ("int_time_err_s", _LF),
     ("t_tli_days", _F), ("jacobi_parking", _F), ("jacobi_tli", _F),
     ("energy_regime_id", _I), ("energy_regime", _S)]
    + [(f"tli_{n}", _F) for n in STATE_NAMES]
    + [(f"end_{n}", _F) for n in STATE_NAMES]
    + [("t_end_days", _F), ("termination", _S), ("min_moon_alt_km", _F), ("t_min_moon_days", _F),
       ("max_earth_dist_km", _F), ("moon_hill_entries", _I), ("t_first_moon_hill_days", _F),
       ("moon_hill_dwell_days_at_end", _F), ("jacobi_drift", _F)]
    + [col for name in LAGRANGE_NAMES for col in (
        (f"min_dist_{name}_km", _F), (f"t_min_dist_{name}_days", _F),
        (f"near_{name}", _B), (f"t_first_near_{name}_days", _F))]
    + [("outcome_id", _I), ("outcome", _S), ("error", _S)]
)
TRAJ_SCHEMA = pa.schema([("run_id", _I), ("t_days", _F)] + [(n, _F) for n in STATE_NAMES])


def rows_to_table(rows):
    cols = {f.name: [r.get(f.name) for r in rows] for f in RUNS_SCHEMA}
    return pa.Table.from_pydict(cols, schema=RUNS_SCHEMA)


def trajs_to_table(trajs):
    run_ids = np.concatenate([np.full(len(tr), rid, dtype=np.int64) for rid, tr in trajs])
    arr = np.vstack([tr for _, tr in trajs])
    cols = {"run_id": run_ids, "t_days": dyn.time_to_days(arr[:, 0])}
    for j, n in enumerate(STATE_NAMES):
        cols[n] = arr[:, j + 1]
    return pa.Table.from_pydict(cols, schema=TRAJ_SCHEMA)


# ---------------------------------------------------------------------------
# Sampling + driver
# ---------------------------------------------------------------------------
def sample_params(n_runs, seed, ranges, start_id=0, repeats=1, sigma_ranges=None):
    """Build the job list: n_runs // repeats plans, each flown `repeats` times.

    Plans come from one generator seeded with `seed`, exactly as in schema v1,
    so with repeats == 1 an old seed reproduces the old missions. Engine error
    sizes are drawn once per plan from a second, independent generator,
    uniformly within sigma_ranges (lo == hi means a fixed size).
    """
    n_plans = n_runs // repeats
    rng = np.random.default_rng(seed)
    lo_b, hi_b = ranges["num_intermediate_burns"]
    cols = {
        "parking_altitude_km": rng.uniform(*ranges["parking_altitude_km"], n_plans),
        "num_intermediate_burns": rng.integers(int(lo_b), int(hi_b) + 1, n_plans),
        "intermediate_dv_kms": rng.uniform(*ranges["intermediate_dv_kms"], n_plans),
        "tli_dv_kms": rng.uniform(*ranges["tli_dv_kms"], n_plans),
        "theta0_deg": rng.uniform(*ranges["theta0_deg"], n_plans),
    }
    sigma_ranges = sigma_ranges or {k: (0.0, 0.0) for k in SIGMA_KEYS}
    sig_rng = np.random.default_rng([seed % 2**63, 1])
    sigs = {k: sig_rng.uniform(*sigma_ranges[k], n_plans) for k in SIGMA_KEYS}
    jobs = []
    for j in range(n_plans):
        plan = {k: (int(v[j]) if k == "num_intermediate_burns" else float(v[j])) for k, v in cols.items()}
        sig = {k: float(v[j]) for k, v in sigs.items()}
        for r in range(repeats):
            jobs.append((start_id + j * repeats + r,
                         {**plan, **sig, "plan_id": start_id + j, "repeat": r}))
    return jobs


def parse_args():
    ap = argparse.ArgumentParser(description="Batch Earth-Moon CR3BP TLI simulations -> parquet.")
    ap.add_argument("--runs", type=int, default=1000, help="total number of simulations (default 1000)")
    ap.add_argument("--seed", type=int, default=2026, help="RNG seed for plans and engine errors")
    ap.add_argument("--start-id", type=int, default=0,
                    help="first run_id/plan_id; for a second batch change BOTH --start-id and --seed")
    ap.add_argument("--coast-days", type=float, default=30.0, help="post-TLI coast length [days]")
    ap.add_argument("--workers", type=int,
                    default=max(min((os.cpu_count() or 2) - 1, DEFAULT_MAX_WORKERS), 1),
                    help=f"worker processes (default: cores - 1, capped at {DEFAULT_MAX_WORKERS})")
    ap.add_argument("--chunk-size", type=int, default=20, help="runs per worker task")
    ap.add_argument("--out", type=Path, default=None, help="output .parquet path for the runs table")
    ap.add_argument("--save-trajectories", action="store_true",
                    help="also write <out>_traj.parquet with the down-sampled post-TLI path of each run")
    ap.add_argument("--traj-points", type=int, default=256, help="samples per trajectory (default 256)")
    for key, (lo, hi) in SAMPLING_DEFAULTS.items():
        typ = int if key == "num_intermediate_burns" else float
        ap.add_argument("--" + key.replace("_", "-"), type=typ, nargs=2, metavar=("LO", "HI"),
                        default=(lo, hi), help=f"sampling range (default {lo} {hi})")
    ap.add_argument("--nominal", type=float, nargs=5,
                    metavar=("ALT_KM", "BURNS", "INT_DV_KMS", "TLI_DV_KMS", "THETA_DEG"),
                    help="fly one fixed plan: pins every sampling range to these values")
    ap.add_argument("--repeats", type=int, default=1,
                    help="fly each plan this many times with fresh engine errors "
                         "(--runs must be a multiple of it)")
    ap.add_argument("--sigma-dv-ms", type=float, nargs="+", default=[0.0], metavar="SIGMA",
                    help="burn magnitude error, 1 std. dev. [m/s], applied to every burn. "
                         "One value = fixed size; LO HI = size drawn per plan (default 0 = off)")
    ap.add_argument("--sigma-pointing-deg", type=float, nargs="+", default=[0.0], metavar="SIGMA",
                    help="in-plane burn pointing error, 1 std. dev. [deg] (same rules)")
    ap.add_argument("--sigma-timing-s", type=float, nargs="+", default=[0.0], metavar="SIGMA",
                    help="burn timing error, 1 std. dev. [s], positive = late (same rules)")
    ap.add_argument("--lagrange-radius-km", type=float, default=R_NEAR_LAGRANGE_KM,
                    help=f"radius for the near_l1..near_l5 flags (default {R_NEAR_LAGRANGE_KM:,.0f} km)")
    args = ap.parse_args()

    if args.repeats < 1 or args.runs < 1 or args.runs % args.repeats:
        ap.error("--runs must be a positive multiple of --repeats")
    for k in SIGMA_KEYS:
        v = getattr(args, k)
        if len(v) not in (1, 2) or min(v) < 0 or v[0] > v[-1]:
            ap.error(f"--{k.replace('_', '-')} takes one value or LO HI with 0 <= LO <= HI")
    return args


def main():
    args = parse_args()
    ranges = {k: tuple(getattr(args, k)) for k in SAMPLING_DEFAULTS}
    if args.nominal:
        alt, burns, int_dv, tli_dv, theta = args.nominal
        ranges = {"parking_altitude_km": (alt, alt),
                  "num_intermediate_burns": (int(burns), int(burns)),
                  "intermediate_dv_kms": (int_dv, int_dv),
                  "tli_dv_kms": (tli_dv, tli_dv),
                  "theta0_deg": (theta, theta)}
    sigma_ranges = {k: (getattr(args, k)[0], getattr(args, k)[-1]) for k in SIGMA_KEYS}
    dispersed = any(hi > 0 for _, hi in sigma_ranges.values())
    traj_points = args.traj_points if args.save_trajectories else 0
    n_plans = args.runs // args.repeats

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out or DEFAULT_OUT_DIR / f"cr3bp_runs_{stamp}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    traj_out = out.with_name(out.stem + "_traj.parquet")

    config = {
        "schema_version": SCHEMA_VERSION, "created": stamp, "runs": args.runs, "seed": args.seed,
        "start_id": args.start_id, "coast_days": args.coast_days, "sampling_ranges": ranges,
        "nominal": args.nominal, "repeats": args.repeats, "plans": n_plans,
        "sigma_ranges": sigma_ranges,
        "error_model": ("independent Gaussian error on every burn: magnitude [m/s], in-plane "
                        "pointing [deg] from the rotating-frame velocity, timing [s] (positive = late); "
                        "sizes drawn once per plan when given as LO HI"),
        "lagrange_radius_km": args.lagrange_radius_km,
        "lagrange_points_nd": {n: _LP_ND[j].tolist() for j, n in enumerate(LAGRANGE_NAMES)},
        "mu": MU, "length_unit_km": LU_KM, "time_unit_days": float(dyn.T_UNIT_DAYS),
        "velocity_unit_kms": VU_KMS, "rk4_max_dt_nd": MAX_DT_ND,
        "r_hill_moon_nd": R_HILL_MOON, "r_escape_nd": R_ESCAPE,
        "outcomes": OUTCOMES, "energy_regimes": ENERGY_REGIMES,
        "frame": "Earth-Moon CR3BP rotating (synodic), barycentric, nondimensional states",
    }
    meta = {b"aiaa_sim_config": json.dumps(config).encode()}

    jobs = sample_params(args.runs, args.seed, ranges, args.start_id, args.repeats, sigma_ranges)
    chunks = [jobs[i:i + args.chunk_size] for i in range(0, len(jobs), args.chunk_size)]

    print(f"Simulating {args.runs} runs ({n_plans} plan(s) x {args.repeats}) | "
          f"coast {args.coast_days} d | {args.workers} worker(s)")
    if dispersed:
        print("Engine errors (1 std. dev.): " + ", ".join(
            f"{k} {lo:g}" + (f"-{hi:g}" if hi != lo else "") for k, (lo, hi) in sigma_ranges.items()))
    elif args.repeats > 1:
        print("Note: --repeats > 1 without any --sigma-* error makes identical repeats.")
    print(f"Output: {out}" + (f"\n        {traj_out}" if traj_points else ""))

    # Compile (or load the cached) Numba kernels once before starting workers,
    # so they all load the same on-disk cache instead of compiling in parallel.
    t0 = time.perf_counter()
    simulate_one(-1, jobs[0][1], 0.05, traj_points, seed=args.seed,
                 lagrange_radius_km=args.lagrange_radius_km)
    print(f"JIT warm-up: {time.perf_counter() - t0:.1f} s")

    writer = pq.ParquetWriter(out, RUNS_SCHEMA.with_metadata(meta))
    traj_writer = pq.ParquetWriter(traj_out, TRAJ_SCHEMA.with_metadata(meta)) if traj_points else None
    counts, near_counts, done = {}, {n: 0 for n in LAGRANGE_NAMES}, 0
    t0 = time.perf_counter()

    def handle(rows, trajs):
        nonlocal done
        writer.write_table(rows_to_table(rows))
        if traj_writer is not None and trajs:
            traj_writer.write_table(trajs_to_table(trajs))
        for r in rows:
            counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
            for n in LAGRANGE_NAMES:
                near_counts[n] += bool(r.get(f"near_{n}"))
        done += len(rows)
        rate = done / (time.perf_counter() - t0)
        print(f"\r  {done}/{args.runs} runs  ({rate:.1f} runs/s)", end="", flush=True)

    pool = None
    if args.workers > 1:
        try:
            pool = ProcessPoolExecutor(max_workers=args.workers)
        except OSError as err:
            print(f"Could not start worker processes ({err}); falling back to 1 core.")

    task_args = (args.coast_days, traj_points, args.seed, args.lagrange_radius_km)
    try:
        if pool is None:
            for ch in chunks:
                handle(*simulate_chunk(ch, *task_args))
        else:
            with pool:
                futs = [pool.submit(simulate_chunk, ch, *task_args) for ch in chunks]
                for fut in as_completed(futs):
                    handle(*fut.result())
    except KeyboardInterrupt:
        print("\nInterrupted — keeping the runs finished so far.")
    except BrokenProcessPool:
        print(f"\nA worker process died (usually out of memory). Kept the {done} runs finished so far."
              f"\nRe-run with fewer workers, e.g. --workers {max(args.workers // 2, 1)}, "
              f"and --start-id {args.start_id + args.runs} --seed {args.seed + 1} to avoid duplicate ids.")
    finally:
        writer.close()
        if traj_writer is not None:
            traj_writer.close()

    print(f"\nDone: {done} runs in {time.perf_counter() - t0:.1f} s")
    print("Outcome counts:")
    for name in OUTCOMES.values():
        if name in counts:
            print(f"  {name:<14s} {counts[name]:6d}  ({100 * counts[name] / max(done, 1):5.1f}%)")
    print(f"Came within {args.lagrange_radius_km:,.0f} km of: " + ", ".join(
        f"{n.upper()} {near_counts[n]} ({100 * near_counts[n] / max(done, 1):.1f}%)" for n in LAGRANGE_NAMES))


if __name__ == "__main__":
    main()
