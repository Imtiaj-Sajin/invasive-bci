# Chronic drift and electrode-failure simulator (`ibci.sim`)

A feature-level generative model of how chronic intracortical recordings change over days to years. It transforms a
**real** base session, so behaviour and latent dynamics stay realistic, into simulated sessions `dt` days later. Its
parameters are **calibrated to real long-term data** rather than chosen by hand.

## Model
`sample_drift(Z_base, dt, params, dist, rng, alive)` draws one realization of change. `apply_drift(Z, drift, rng)`
applies it to any segment of the base session (z-scored features, `T × C`). `to_raw` maps back to raw units.

| Component | What it does | Parameters | Source |
|---|---|---|---|
| Mixing | Z ← (I + E) Z; E is Gaussian, partly restricted to grid neighbours | `s_mix0` (session-to-session), `s_mix`, `tau_mix` (slow), `beta_local` | Ladder calibration |
| Turnover | A fraction ρ of each channel's signal is replaced by a new unit with rotated tuning in the session's latent space | `rho0`, `rho_inf`, `tau_rho`, `rho_conc` | Ladder calibration |
| Silencing | Channels alive on the base day (TC > 2 Hz) switch to silent through a two-state Markov chain; a silent channel keeps a fraction of its SBP signal | `h_off`, `h_on`, `silent_signal_frac` | Activity statistics (`failure_stats_xdata.py`); LINK within-channel tuning ratio (0.37) |
| Gain jumps | Heavy-tailed per-channel gain events | `h_jump`, `jump_sd` | Defaults (not used by renormalized decoders) |
| Raw offset/scale walks | Matter only for decoders that are not renormalized | `rw_offset`, `rw_logscale` | Defaults |

Magnitudes grow with elapsed days as `instant + slow * (1 - exp(-dt / tau))`.

## Calibration (`scripts/calibrate_sim.py`)
- The simulator's own **oracle ladder** is matched to the real one at gaps of 1, 7, 30, 120 and 480 days. The three rungs
  matched are renormalized frozen decoder, + per-channel gains, and + full input remap, each as a ratio to the own-day decoder.
- The simulated ladder uses exactly the same correction procedure and regularization as the real data. *An earlier run that
  did not do this failed (see the log, 2026-10-06 04:50).*
- Optimizer: a Sobol global search, then Nelder–Mead, with common random numbers.
- The current fit (`results/sim_calib_split/calibration.json`) uses LINK days < 700 only, so later days are held out
  for validation. The loss is 0.074.

## Validation (`scripts/validate_sim.py`)
Uses held-out time (days ≥ 700), rungs that were not fitted (label-free Procrustes, stable-channel alignment, latent
rotation, ridge-to-prior) and data budgets that were not fitted (20, 50, 100 trials). Results are in `results/sim_validation`.

## Usage
```python
import json, numpy as np
from ibci import sim
from ibci.anatomy import Sess
from ibci.data import link
p = sim.SimParams(**json.load(open("results/sim_calib_split/calibration.json"))["params"])
base = Sess("2020-03-10_CO")
d = sim.sample_drift(base.ztr, 30, p, link.chebyshev_distance(link.electrode_layout()), np.random.default_rng(0),
                     alive=base.alive)
future_test = sim.apply_drift(base.zte, d, np.random.default_rng(1))   # simulated session 30 days later
```
A complete example is `examples/simulate_drift.py`.

## Intended uses and limits
- **Use it for** stress-testing decoders and unsupervised stabilizers under controlled drift and electrode failure, studying
  recalibration policies, and as augmentation (being evaluated in `scripts/sim_augment.py`).
- **Limits:**
  - It is calibrated on one monkey's Utah arrays (spike-band power, finger task). Replication data from Chewie and Mihili exist but
    were not yet used for calibration.
  - The mechanisms reproduce the measured statistics. That does not prove they are the true physical mechanisms.
  - The 30-day point is fitted least well: the simulator decays too slowly there.
