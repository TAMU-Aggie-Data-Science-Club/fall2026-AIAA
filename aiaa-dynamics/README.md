# aiaa-dynamics

Earth-Moon Circular Restricted Three-Body Problem (CR3BP) simulator. The AIAA side generates labeled trajectories here; ADSC trains models on the parquet output.

| File | Purpose |
|---|---|
| `dynamics.py` | CR3BP physics: equations of motion, Jacobi constant, STM, Lagrange points, frames/units. |
| `TLIs.py` | Single translunar-injection mission (parking orbit -> prograde burns at perigee -> TLI) with a plot. Numba-JIT RK4 propagator. |
| `class_thresholds.py` | Defines the **end behavior** of an orbit: outcome labels, energy regimes, the Lagrange-point "near" radius, and the thresholds behind them. |
| `mission.py` | **Batch simulator.** Samples thousands of missions (optionally with random engine errors), runs them across CPU cores with Numba-JIT, writes parquet. |

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

Speed: about 7 runs/s per core (about 24 runs/s with 8 workers) with the default 30-day coast (fixed RK4 step of about 3.75 s). The first run compiles the kernels; after that they load from Numba's on-disk cache.

Useful flags: `--seed` (reproducible sampling), `--start-id`, `--coast-days`, and one `LO HI` range flag per input, for example `--tli-dv-kms 2.9 3.1`. Run `python mission.py -h` for all of them.

**Adding another batch:** change **both** `--seed` and `--start-id`, for example `--start-id 5000 --seed 2027`. The same seed flies the same missions again, and the same start ID repeats `run_id`s.

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

## Dispersion mode (planned vs. actual burns)

Real engines never deliver exactly the planned burn. Turn on any `--sigma-*` flag and every burn (each intermediate burn and the TLI) gets its own random error, drawn from a bell curve (Gaussian) with that standard deviation:

| Flag | Error on each burn | Units |
|---|---|---|
| `--sigma-dv-ms` | size (delta-v magnitude) | m/s |
| `--sigma-pointing-deg` | direction, rotated in the orbit plane away from prograde | deg |
| `--sigma-timing-s` | fires early or late (positive = late) | s |

- Each flag takes **one value** (the same error size for every plan) or **`LO HI`** (a size drawn once per plan, uniformly in that range, so a model can learn how risk grows with engine error).
- `--repeats N` flies each plan N times with fresh errors; `--runs` must be a multiple of N. All repeats of a plan share its `plan_id` and error sizes.
- `--nominal ALT_KM BURNS INT_DV_KMS TLI_DV_KMS THETA_DEG` flies one fixed plan instead of sampling plans.
- Errors are reproducible: each run's errors come from (`--seed`, `run_id`).

```bash
# One plan flown 2000 times with 1 m/s burn errors ("same mission + noise"):
python mission.py --runs 2000 --repeats 2000 --nominal 300 2 0.03 3.047 242 --sigma-dv-ms 1 --save-trajectories

# Dataset for an outcome-probability model: 5000 random plans x 20 repeats,
# engine error size drawn per plan between 0 and 3 m/s:
python mission.py --runs 100000 --repeats 20 --sigma-dv-ms 0 3
```

In a 120-run test of that first plan (aimed at a lunar flyby of about 1,500 km), 1 m/s errors on every burn gave 33% Moon impact, 39% escape and 28% lunar flyby.

## Lagrange-point outputs

For each of L1 to L5 the coast records the closest distance, when it happened, whether the path came within the "near" radius, and the first time it did (columns below). The radius defaults to 50,000 km (`R_NEAR_LAGRANGE_KM` in `class_thresholds.py`); change it per batch with `--lagrange-radius-km`. These columns describe where a path went; they don't change the outcome label.

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

## Runs table schema (`schema_version` 2, one row per run)

States are nondimensional, rotating (synodic) frame, origin at the Earth-Moon barycenter, Earth at (-mu, 0, 0), Moon at (1 - mu, 0, 0). 1 LU = 384,400 km, 1 TU ≈ 4.343 days, 1 VU ≈ 1.0245 km/s (`dynamics.to_dimensional` converts).

Schema 2 adds `plan_id`/`repeat`, the engine-error columns and the Lagrange-point columns. The 37 schema-1 columns are unchanged and give identical values for the same seed when no errors are used.

**Which columns can be model inputs:** in dispersion mode the "as flown" columns already contain the random engine errors, so a model that predicts outcome probabilities from a plan may only use the **plan** and **error-size** columns. "No-error data only" below means usable as an input only in batches without `--sigma-*` errors.

| Column | Type | Meaning | Model input? |
|---|---|---|---|
| `run_id` | int | unique id (unique across batches when `--start-id` differs) | no |
| `plan_id`, `repeat` | int | which plan the run belongs to, and which repeat (`plan_id` = `run_id` when `--repeats` is 1) | group by `plan_id` |
| `parking_altitude_km`, `num_intermediate_burns`, `intermediate_dv_kms`, `tli_dv_kms`, `theta0_deg` | inputs | the **planned** mission | yes |
| `total_dv_kms` | float | planned intermediate burns + TLI | yes |
| `coast_days` | float | requested post-TLI coast length | yes |
| `sigma_dv_ms`, `sigma_pointing_deg`, `sigma_timing_s` | float | engine error sizes (1 std. dev.); 0 when dispersion is off | yes |
| `jacobi_parking` | float | Jacobi constant of the parking orbit, before any burn | yes |
| `jacobi_tli_plan` | float | Jacobi constant after TLI if every burn were perfect | yes |
| `tli_dv_err_ms`, `tli_point_err_deg`, `tli_time_err_s` | float | the TLI burn's actual random errors | no (hidden truth) |
| `int_dv_err_ms`, `int_point_err_deg`, `int_time_err_s` | list of float | the same, one entry per intermediate burn | no (hidden truth) |
| `t_tli_days` | float | time from start to the TLI burn, as flown | no-error data only |
| `jacobi_tli` | float | Jacobi constant after TLI, as flown (conserved during the coast) | no-error data only |
| `energy_regime_id`, `energy_regime` | int, str | energy regime of `jacobi_tli` (see above) | no-error data only |
| `tli_x … tli_vz` | float | state right after TLI, as flown | no-error data only |
| `end_x … end_vz` | float | last state of the coast | no (answer) |
| `t_end_days` | float | coast time actually flown; the impact or escape time when `termination` is not `none` | no (answer) |
| `termination` | str | `none`, `earth_impact`, `moon_impact`, `escape` | no (answer) |
| `min_moon_alt_km`, `t_min_moon_days` | float | closest lunar approach (altitude) and when it happened (the flyby time), from TLI | no (answer) |
| `max_earth_dist_km` | float | farthest distance from Earth | no (answer) |
| `moon_hill_entries`, `t_first_moon_hill_days` | int, float | Hill-sphere entries and time of the first one (NaN if never) | no (answer) |
| `moon_hill_dwell_days_at_end` | float | time inside the Hill sphere at the end (0 if outside) | no (answer) |
| `jacobi_drift` | float | C(end) − C(TLI); integration-quality check, typically < 1e-8 | no (quality check) |
| `min_dist_l1_km` … `min_dist_l5_km` | float | closest distance to each Lagrange point | no (answer) |
| `t_min_dist_l1_days` … `t_min_dist_l5_days` | float | when that closest approach happened, from TLI | no (answer) |
| `near_l1` … `near_l5` | bool | came within `--lagrange-radius-km` of that point | no (answer) |
| `t_first_near_l1_days` … `t_first_near_l5_days` | float | first time it came that close, from TLI (NaN if never) | no (answer) |
| `outcome_id`, `outcome` | int, str | label (see table above) | no (answer) |
| `error` | str | null unless the run failed | no |

## Trajectory table schema (`--save-trajectories`)

Long format, one row per sample, about `--traj-points` samples per run, evenly spaced in time from TLI (the final point is always included):

`run_id`, `t_days` (time since TLI), `x`, `y`, `z`, `vx`, `vy`, `vz`. Join to the runs table on `run_id`.
