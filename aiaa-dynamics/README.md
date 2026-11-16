# aiaa-dynamics

Earth-Moon Circular Restricted Three-Body Problem (CR3BP) simulator. The AIAA side generates labeled trajectories here; ADSC trains models on the parquet output.

| File | Purpose |
|---|---|
| `dynamics.py` | CR3BP physics: equations of motion, Jacobi constant, STM, Lagrange points, frames/units. |
| `TLIs.py` | Single translunar-injection mission (parking orbit -> prograde burns at perigee -> TLI) with a plot. Numba-JIT RK4 propagator. |
| `class_thresholds.py` | Defines the **end behavior** of an orbit: outcome labels, energy regimes, and the thresholds behind them. |
| `mission.py` | **Batch simulator.** Samples thousands of missions, runs them across CPU cores with Numba-JIT, writes parquet. |

## Setup

```bash
pip install -r aiaa-dynamics/requirements.txt
```

## Generate a dataset

Run from inside `aiaa-dynamics/`:

```bash
python mission.py --runs 10 --workers 1        # smoke test (~10 s including JIT compile)
python mission.py --runs 5000                  # full batch, uses all cores but one
python mission.py --runs 5000 --save-trajectories --traj-points 256
```

Output lands in `data/processed/cr3bp_runs_<timestamp>.parquet` (and `..._traj.parquet` with `--save-trajectories`). `data/` is git-ignored: **share the files with ADSC directly, never commit them.**

Speed: about 7 runs/s per core with the default 30-day coast (fixed RK4 step of about 3.75 s). The first run compiles the kernels; after that they load from Numba's on-disk cache.

Useful flags: `--seed` (reproducible sampling), `--start-id` (give a second batch run_ids that don't collide with the first), `--coast-days`, and one `LO HI` range flag per input, for example `--tli-dv-kms 2.9 3.1`. Run `python mission.py -h` for all of them.

### Sampled inputs (uniform)

| Input | Default range |
|---|---|
| `parking_altitude_km` | 185 – 1000 |
| `num_intermediate_burns` | 0 – 3 (integer) |
| `intermediate_dv_kms` | 0.00 – 0.10 |
| `tli_dv_kms` | 2.80 – 3.20 |
| `theta0_deg` | 0 – 360 |

The full run config (ranges, seed, units, label tables) is stored in the parquet schema metadata under the key `aiaa_sim_config`:

```python
import json, pyarrow.parquet as pq
cfg = json.loads(pq.read_schema("cr3bp_runs_....parquet").metadata[b"aiaa_sim_config"])
```

## Outcome labels (`class_thresholds.py`)

Checked in this priority order. All thresholds are constants at the top of `class_thresholds.py`.

| `outcome_id` | `outcome` | Rule |
|---|---|---|
| 3 | `earth_impact` | distance to Earth center < Earth radius |
| 4 | `moon_impact` | distance to Moon center < Moon radius |
| 5 | `escape` | distance from barycenter > 4 LU (~1.54 million km) |
| 2 | `lunar_capture` | ends inside the Moon's Hill sphere (~61,300 km), bound to the Moon, after ≥ 1 day inside |
| 1 | `lunar_flyby` | entered the Moon's Hill sphere but none of the above |
| 0 | `earth_orbit` | never reached the Hill sphere, still bound to Earth |
| 6 | `unresolved` | unbound from Earth but still in the system at the end; use a longer `--coast-days` |
| -1 | `failed` | the run raised an error (see `error` column) |

Energy regime (from the post-TLI Jacobi constant vs. the Lagrange points): `closed` (C > C_L1, can't reach the Moon), `L1_open`, `L2_open`, `L3_open`, `all_open`.

There is no lunar-orbit-insertion burn in the mission yet, so `lunar_capture` is expected to be rare or absent.

## Runs table schema (one row per run)

States are nondimensional, rotating (synodic) frame, origin at the Earth-Moon barycenter, Earth at (-mu, 0, 0), Moon at (1 - mu, 0, 0). 1 LU = 384,400 km, 1 TU ≈ 4.343 days, 1 VU ≈ 1.0245 km/s (`dynamics.to_dimensional` converts).

| Column | Type | Meaning |
|---|---|---|
| `run_id` | int | unique id within a batch |
| `parking_altitude_km`, `num_intermediate_burns`, `intermediate_dv_kms`, `tli_dv_kms`, `theta0_deg` | inputs | sampled mission inputs |
| `total_dv_kms` | float | intermediate burns + TLI |
| `coast_days` | float | requested post-TLI coast length |
| `t_tli_days` | float | time from start to the TLI burn |
| `jacobi_parking`, `jacobi_tli` | float | Jacobi constant before the first burn and after TLI (conserved during the coast) |
| `energy_regime_id`, `energy_regime` | int, str | see above |
| `tli_x … tli_vz` | float | state right after TLI (model inputs) |
| `end_x … end_vz` | float | last state of the coast |
| `t_end_days` | float | coast time actually flown (shorter than `coast_days` on impact/escape) |
| `termination` | str | `none`, `earth_impact`, `moon_impact`, `escape` |
| `min_moon_alt_km`, `t_min_moon_days` | float | closest lunar approach (altitude) and when, from TLI |
| `max_earth_dist_km` | float | farthest distance from Earth |
| `moon_hill_entries`, `t_first_moon_hill_days` | int, float | Hill-sphere entries and time of the first one (NaN if never) |
| `moon_hill_dwell_days_at_end` | float | time inside the Hill sphere at the end (0 if outside) |
| `jacobi_drift` | float | C(end) − C(TLI); integration-quality check, typically < 1e-8 |
| `outcome_id`, `outcome` | int, str | label (see table above) |
| `error` | str | null unless the run failed |

## Trajectory table schema (`--save-trajectories`)

Long format, one row per sample, about `--traj-points` samples per run, evenly spaced in time from TLI (the final point is always included):

`run_id`, `t_days` (time since TLI), `x`, `y`, `z`, `vx`, `vy`, `vz`. Join to the runs table on `run_id`.
